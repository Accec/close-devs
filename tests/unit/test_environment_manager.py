from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.config import (
    AppConfig,
    DynamicDebugConfig,
    EnvironmentConfig,
    LLMConfig,
    StaticReviewConfig,
)
from tools.command_runner import CommandResult
from tools.environment_manager import EnvironmentManager
from tools.file_store import FileStore
from tests.support import sqlite_database_config


@dataclass(slots=True)
class FakeRunner:
    fail_requirement_install: bool = False

    async def run(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 120,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        if " -m venv " in command:
            bin_dir = cwd / ".venv" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        returncode = 0
        stderr = ""
        if self.fail_requirement_install and " -m pip install -r " in command:
            returncode = 1
            stderr = "dependency install failed"
        return CommandResult(
            command=command,
            returncode=returncode,
            stdout="ok",
            stderr=stderr,
            duration_seconds=0.01,
        )


def _build_config(tmp_path: Path) -> AppConfig:
    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    return AppConfig(
        repo_root=tmp_path / "repo",
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py"],
        exclude=["state/**", "reports/**"],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=["python -c \"print('ok')\""]),
        database=sqlite_database_config(tmp_path),
        environment=EnvironmentConfig(
            enabled=True,
            scope="all_analysis",
            install_mode="auto_detect",
            install_fail_policy="mark_degraded",
            python_executable="python3",
            bootstrap_tools=False,
        ),
    )


@pytest.mark.asyncio
async def test_environment_manager_prefers_src_requirements_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "requirements.txt").write_text("example-package==1.0.0\n", encoding="utf-8")
    (repo_root / "requirements.txt").write_text("ignored-package==1.0.0\n", encoding="utf-8")
    config = _build_config(tmp_path)
    manager = EnvironmentManager(
        file_store=FileStore(),
        command_runner=FakeRunner(),
    )

    environment = await manager.prepare(
        report_dir=tmp_path / "reports" / "run-1",
        source_repo_root=repo_root,
        config=config,
    )

    assert environment.detected_sources == ["src/requirements.txt"]
    assert environment.status == "ready"
    assert Path(environment.environment_json_path or "").exists()
    assert Path(environment.install_log_path or "").exists()
    assert Path(environment.base_workspace_root).exists()
    assert Path(environment.maintenance_workspace_root).exists()
    assert Path(environment.validation_workspace_root).exists()


@pytest.mark.asyncio
async def test_environment_manager_marks_degraded_on_install_failure(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "src" / "requirements.txt").write_text("broken-package==0.0.1\n", encoding="utf-8")
    config = _build_config(tmp_path)
    manager = EnvironmentManager(
        file_store=FileStore(),
        command_runner=FakeRunner(fail_requirement_install=True),
    )

    environment = await manager.prepare(
        report_dir=tmp_path / "reports" / "run-2",
        source_repo_root=repo_root,
        config=config,
    )

    assert environment.status == "degraded"
    assert environment.install_errors
