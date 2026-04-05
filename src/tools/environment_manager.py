from __future__ import annotations

from dataclasses import asdict
import logging
import os
from pathlib import Path
import shutil
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

        detected_sources = self._detect_dependency_sources(base_workspace_root, config)
        install_commands: list[str] = []
        install_errors: list[str] = []
        install_logs: list[str] = []
        bootstrap_packages = self._bootstrap_packages(config)
        installer_summary: dict[str, str] = {}

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
            installer_summary=installer_summary,
            install_log_path=str(install_log_path),
            environment_json_path=str(environment_json_path),
        )

        if python_bin.exists():
            install_env = self._install_env(bin_dir, venv_root)
            await self._run_install_command(
                command=(
                    f"{shlex.quote(str(python_bin))} -m pip install --upgrade pip setuptools wheel"
                ),
                cwd=base_workspace_root,
                install_commands=install_commands,
                install_errors=install_errors,
                install_logs=install_logs,
                env=install_env,
            )
            for source in detected_sources:
                commands, warnings, installer_mode = self._dependency_install_commands(
                    python_bin=python_bin,
                    source=source,
                    env=install_env,
                )
                installer_summary[source] = installer_mode
                install_logs.extend(warnings)
                for command in commands:
                    await self._run_install_command(
                        command=command,
                        cwd=base_workspace_root,
                        install_commands=install_commands,
                        install_errors=install_errors,
                        install_logs=install_logs,
                        env=install_env,
                    )
            if config.environment.bootstrap_tools and bootstrap_packages:
                installer_summary["bootstrap_tools"] = "pip"
                await self._run_install_command(
                    command=(
                        f"{shlex.quote(str(python_bin))} -m pip install "
                        + " ".join(shlex.quote(package) for package in bootstrap_packages)
                    ),
                    cwd=base_workspace_root,
                    install_commands=install_commands,
                    install_errors=install_errors,
                    install_logs=install_logs,
                    env=install_env,
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

    def _dependency_install_commands(
        self,
        *,
        python_bin: Path,
        source: str,
        env: dict[str, str],
    ) -> tuple[list[str], list[str], str]:
        warnings: list[str] = []
        if source in {"pyproject.toml:poetry.dependencies", "poetry.lock"}:
            if self._tool_available("poetry", env):
                return (["poetry install --no-interaction"], warnings, "native:poetry")
            warnings.append(
                "native-installer-unavailable: poetry not found on PATH, falling back to pip editable install"
            )
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e ."], warnings, "fallback:pip")
        if source in {"pyproject.toml:pdm.dependencies", "pdm.lock"}:
            if self._tool_available("pdm", env):
                return (
                    [
                        f"pdm use -f {shlex.quote(str(python_bin))}",
                        "pdm sync" if source == "pdm.lock" else "pdm install",
                    ],
                    warnings,
                    "native:pdm",
                )
            warnings.append(
                "native-installer-unavailable: pdm not found on PATH, falling back to pip editable install"
            )
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e ."], warnings, "fallback:pip")
        if source in {"pyproject.toml:uv.dependencies", "uv.lock"}:
            if self._tool_available("uv", env):
                uv_command = "uv sync --all-groups --all-extras"
                if source == "uv.lock":
                    uv_command = "uv sync --frozen --all-groups --all-extras"
                return ([uv_command], warnings, "native:uv")
            warnings.append(
                "native-installer-unavailable: uv not found on PATH, falling back to pip editable install"
            )
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e ."], warnings, "fallback:pip")
        if source == "pyproject.toml:project.optional-dependencies.test":
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e .[test]"], warnings, "pip")
        if source == "pyproject.toml:project.optional-dependencies.dev":
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e .[dev]"], warnings, "pip")
        if source.startswith("pyproject.toml:") or source in {"poetry.lock", "pdm.lock", "uv.lock"}:
            return ([f"{shlex.quote(str(python_bin))} -m pip install -e ."], warnings, "pip")
        return (
            [
                f"{shlex.quote(str(python_bin))} -m pip install -r "
                f"{shlex.quote(source)}"
            ],
            warnings,
            "pip",
        )

    def _detect_dependency_sources(self, repo_root: Path, config: AppConfig) -> list[str]:
        pyproject_data: dict[str, object] = {}
        pyproject_path = repo_root / "pyproject.toml"
        if pyproject_path.is_file():
            try:
                with pyproject_path.open("rb") as handle:
                    pyproject_data = tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                pyproject_data = {}

        runtime_source: str | None = None
        supplemental_sources: list[str] = []
        for candidate in self._dependency_priority_candidates(config):
            if not self._candidate_exists(repo_root, pyproject_data, candidate):
                continue
            if self._is_supplemental_dependency_source(candidate):
                if candidate not in supplemental_sources:
                    supplemental_sources.append(candidate)
            elif runtime_source is None:
                runtime_source = candidate

        detected: list[str] = []
        if runtime_source is not None:
            detected.append(runtime_source)
        detected.extend(
            source for source in supplemental_sources if source != runtime_source
        )
        return detected

    def _candidate_exists(
        self,
        repo_root: Path,
        pyproject_data: dict[str, object],
        candidate: str,
    ) -> bool:
        if candidate.endswith(".lock") or candidate.endswith(".txt"):
            return (repo_root / candidate).is_file()
        if candidate == "pyproject.toml:project.dependencies":
            dependencies = (
                pyproject_data.get("project", {}).get("dependencies", [])
                if isinstance(pyproject_data, dict)
                else []
            )
            return isinstance(dependencies, list) and bool(dependencies)
        if candidate == "pyproject.toml:project.optional-dependencies.dev":
            return self._has_project_optional_dependencies(pyproject_data, "dev")
        if candidate == "pyproject.toml:project.optional-dependencies.test":
            return self._has_project_optional_dependencies(pyproject_data, "test")
        if candidate == "pyproject.toml:poetry.dependencies":
            dependencies = (
                pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                if isinstance(pyproject_data, dict)
                else {}
            )
            return isinstance(dependencies, dict) and any(key != "python" for key in dependencies)
        if candidate == "pyproject.toml:pdm.dependencies":
            project_section = pyproject_data.get("project", {}) if isinstance(pyproject_data, dict) else {}
            optional_deps = project_section.get("optional-dependencies", {}) if isinstance(project_section, dict) else {}
            tool_section = pyproject_data.get("tool", {}) if isinstance(pyproject_data, dict) else {}
            pdm_section = tool_section.get("pdm", {}) if isinstance(tool_section, dict) else {}
            return bool(optional_deps or pdm_section)
        if candidate == "pyproject.toml:uv.dependencies":
            tool_section = pyproject_data.get("tool", {}) if isinstance(pyproject_data, dict) else {}
            uv_section = tool_section.get("uv", {}) if isinstance(tool_section, dict) else {}
            return bool(uv_section)
        return False

    def _has_project_optional_dependencies(
        self,
        pyproject_data: dict[str, object],
        group: str,
    ) -> bool:
        if not isinstance(pyproject_data, dict):
            return False
        project_section = pyproject_data.get("project", {})
        optional_deps = project_section.get("optional-dependencies", {}) if isinstance(project_section, dict) else {}
        group_deps = optional_deps.get(group, []) if isinstance(optional_deps, dict) else []
        return isinstance(group_deps, list) and bool(group_deps)

    def _is_supplemental_dependency_source(self, candidate: str) -> bool:
        return candidate in {
            "src/requirements-dev.txt",
            "requirements-dev.txt",
            "requirements/dev.txt",
            "src/requirements-test.txt",
            "requirements-test.txt",
            "requirements/test.txt",
            "pyproject.toml:project.optional-dependencies.dev",
            "pyproject.toml:project.optional-dependencies.test",
        }

    def _dependency_priority_candidates(self, config: AppConfig) -> list[str]:
        return [
            str(item)
            for item in (
                config.environment.dependency_sources_priority
                or [
                    "src/requirements.txt",
                    "requirements.txt",
                    "requirements/base.txt",
                    "src/requirements-dev.txt",
                    "requirements-dev.txt",
                    "requirements/dev.txt",
                    "src/requirements-test.txt",
                    "requirements-test.txt",
                    "requirements/test.txt",
                    "pyproject.toml:project.dependencies",
                    "pyproject.toml:project.optional-dependencies.dev",
                    "pyproject.toml:project.optional-dependencies.test",
                    "pyproject.toml:poetry.dependencies",
                    "poetry.lock",
                    "pyproject.toml:pdm.dependencies",
                    "pdm.lock",
                    "pyproject.toml:uv.dependencies",
                    "uv.lock",
                ]
            )
        ]

    def _bootstrap_packages(self, config: AppConfig) -> list[str]:
        packages: list[str] = []
        if config.static_review.ruff_command:
            packages.append("ruff")
        if config.static_review.mypy_command:
            packages.append("mypy")
        if config.static_review.bandit_command:
            packages.append("bandit")
        if config.static_review.dependency_audit_command:
            packages.append("pip-audit")
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
        env: dict[str, str] | None = None,
    ) -> None:
        install_commands.append(command)
        result = await self.command_runner.run(
            command=command,
            cwd=cwd,
            timeout_seconds=900,
            env=env,
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

    def _install_env(self, bin_dir: Path, venv_root: Path) -> dict[str, str]:
        env = dict(os.environ)
        current_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([str(bin_dir), current_path]) if current_path else str(bin_dir)
        env["VIRTUAL_ENV"] = str(venv_root)
        env.setdefault("POETRY_VIRTUALENVS_CREATE", "false")
        return env

    def _tool_available(self, tool_name: str, env: dict[str, str]) -> bool:
        return shutil.which(tool_name, path=env.get("PATH")) is not None

    def _environment_json(
        self,
        environment: ExecutionEnvironment,
        source_repo_root: Path,
    ) -> dict[str, object]:
        data = dict(asdict(environment))
        data["source_repo_root"] = str(source_repo_root)
        data["environment_degraded"] = environment.degraded
        return data
