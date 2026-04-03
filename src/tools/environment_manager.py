from __future__ import annotations

from dataclasses import asdict
import logging
import os
from pathlib import Path
import shlex
import tomllib

from core.config import AppConfig
from core.models import ExecutionEnvironment
from reports.serializer import write_json
from tools.command_runner import CommandRunner
from tools.file_store import FileStore


class EnvironmentManager:
    def __init__(
        self,
        *,
        file_store: FileStore | None = None,
        command_runner: CommandRunner | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.file_store = file_store or FileStore()
        self.command_runner = command_runner or CommandRunner()
        self.logger = logger or logging.getLogger("close_devs")

    async def prepare(
        self,
        *,
        report_dir: Path,
        source_repo_root: Path,
        config: AppConfig,
    ) -> ExecutionEnvironment:
        runtime_root = report_dir / "runtime"
        artifacts_root = report_dir / "artifacts"
        base_workspace_root = runtime_root / "base_workspace" / source_repo_root.name
        maintenance_workspace_root = runtime_root / "maintenance_workspace" / source_repo_root.name
        validation_workspace_root = runtime_root / "validation_workspace" / source_repo_root.name
        venv_root = runtime_root / ".venv"
        bin_dir = venv_root / ("Scripts" if os.name == "nt" else "bin")
        python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
        install_log_path = artifacts_root / "install.log"
        environment_json_path = artifacts_root / "environment.json"

        await self.file_store.ensure_dir(runtime_root)
        await self.file_store.ensure_dir(artifacts_root)
        await self.file_store.materialize_workspace_copy(
            source_repo_root,
            destination=base_workspace_root,
        )
        await self.file_store.materialize_workspace_copy(
            base_workspace_root,
            destination=maintenance_workspace_root,
        )
        await self.file_store.materialize_workspace_copy(
            base_workspace_root,
            destination=validation_workspace_root,
        )

        detected_sources = self._detect_dependency_sources(base_workspace_root)
        install_commands: list[str] = []
        install_errors: list[str] = []
        install_logs: list[str] = []
        bootstrap_packages = self._bootstrap_packages(config)

        await self._run_install_command(
            command=(
                f"{shlex.quote(config.environment.python_executable)} -m venv "
                f"{shlex.quote(str(venv_root))}"
            ),
            cwd=runtime_root,
            install_commands=install_commands,
            install_errors=install_errors,
            install_logs=install_logs,
        )

        effective_python = str(python_bin if python_bin.exists() else config.environment.python_executable)
        effective_bin_dir = str(bin_dir if bin_dir.exists() else runtime_root)
        env = ExecutionEnvironment(
            report_dir=str(report_dir),
            runtime_root=str(runtime_root),
            base_workspace_root=str(base_workspace_root),
            maintenance_workspace_root=str(maintenance_workspace_root),
            validation_workspace_root=str(validation_workspace_root),
            venv_root=str(venv_root),
            python_bin=effective_python,
            bin_dir=effective_bin_dir,
            status="ready",
            detected_sources=list(detected_sources),
            install_commands=install_commands,
            install_errors=install_errors,
            bootstrap_packages=bootstrap_packages,
            install_log_path=str(install_log_path),
            environment_json_path=str(environment_json_path),
        )

        if python_bin.exists():
            await self._run_install_command(
                command=(
                    f"{shlex.quote(str(python_bin))} -m pip install --upgrade pip setuptools wheel"
                ),
                cwd=base_workspace_root,
                install_commands=install_commands,
                install_errors=install_errors,
                install_logs=install_logs,
            )
            for source in detected_sources:
                await self._run_install_command(
                    command=self._dependency_install_command(python_bin, source),
                    cwd=base_workspace_root,
                    install_commands=install_commands,
                    install_errors=install_errors,
                    install_logs=install_logs,
                )
            if config.environment.bootstrap_tools and bootstrap_packages:
                await self._run_install_command(
                    command=(
                        f"{shlex.quote(str(python_bin))} -m pip install "
                        + " ".join(shlex.quote(package) for package in bootstrap_packages)
                    ),
                    cwd=base_workspace_root,
                    install_commands=install_commands,
                    install_errors=install_errors,
                    install_logs=install_logs,
                )
        else:
            install_errors.append("virtualenv-python-missing")
            install_logs.append("virtual environment python executable was not created")

        if install_errors:
            env.status = "degraded"

        await self.file_store.write_text(install_log_path, "\n\n".join(install_logs).strip() + "\n")
        await write_json(environment_json_path, self._environment_json(env, source_repo_root))
        return env

    async def refresh_validation_workspace(self, environment: ExecutionEnvironment) -> Path:
        source = Path(environment.maintenance_workspace_root)
        destination = Path(environment.validation_workspace_root)
        await self.file_store.materialize_workspace_copy(source, destination=destination)
        return destination

    def _dependency_install_command(self, python_bin: Path, source: str) -> str:
        if source == "pyproject.toml:project.dependencies":
            return f"{shlex.quote(str(python_bin))} -m pip install -e ."
        return (
            f"{shlex.quote(str(python_bin))} -m pip install -r "
            f"{shlex.quote(source)}"
        )

    def _detect_dependency_sources(self, repo_root: Path) -> list[str]:
        candidates = [
            "src/requirements.txt",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
        ]
        for candidate in candidates:
            if (repo_root / candidate).is_file():
                return [candidate]

        pyproject_path = repo_root / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                with pyproject_path.open("rb") as handle:
                    data = tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                return []
            project = data.get("project", {})
            dependencies = project.get("dependencies", [])
            if isinstance(dependencies, list) and dependencies:
                return ["pyproject.toml:project.dependencies"]
        return []

    def _bootstrap_packages(self, config: AppConfig) -> list[str]:
        packages: list[str] = []
        if config.static_review.ruff_command:
            packages.append("ruff")
        if config.static_review.mypy_command:
            packages.append("mypy")
        if config.static_review.bandit_command:
            packages.append("bandit")
        if config.dynamic_debug.test_commands:
            packages.append("pytest")
        return sorted(set(packages))

    async def _run_install_command(
        self,
        *,
        command: str,
        cwd: Path,
        install_commands: list[str],
        install_errors: list[str],
        install_logs: list[str],
    ) -> None:
        install_commands.append(command)
        result = await self.command_runner.run(
            command=command,
            cwd=cwd,
            timeout_seconds=900,
        )
        install_logs.append(
            "\n".join(
                [
                    f"$ {command}",
                    f"cwd: {cwd}",
                    f"exit_code: {result.returncode}",
                    "--- stdout ---",
                    result.stdout.strip(),
                    "--- stderr ---",
                    result.stderr.strip(),
                ]
            ).strip()
        )
        if result.returncode != 0:
            install_errors.append(
                f"{command} (exit {result.returncode})"
            )

    def _environment_json(
        self,
        environment: ExecutionEnvironment,
        source_repo_root: Path,
    ) -> dict[str, object]:
        data = dict(asdict(environment))
        data["source_repo_root"] = str(source_repo_root)
        data["environment_degraded"] = environment.degraded
        return data
