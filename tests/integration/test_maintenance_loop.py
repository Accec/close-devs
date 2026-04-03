from __future__ import annotations

from pathlib import Path
import sys
import json

import pytest

from core.config import AppConfig, DynamicDebugConfig, EnvironmentConfig, LLMConfig, StaticReviewConfig
from core.orchestrator import Orchestrator
from tests.support import sqlite_database_config


@pytest.mark.asyncio
async def test_maintenance_loop_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = tmp_path / "sample_repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(1, 2) == 4\n",
        encoding="utf-8",
    )

    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py"],
        exclude=["state/**", "reports/**"],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(
            max_complexity=10,
            ruff_command=None,
            mypy_command=None,
            bandit_command=None,
        ),
        dynamic_debug=DynamicDebugConfig(
            smoke_commands=[],
            test_commands=[
                "python -c \"from app import add; import sys; sys.exit(0 if add(1, 2) == 4 else 1)\""
            ],
            timeout_seconds=60,
        ),
        database=sqlite_database_config(tmp_path),
        environment=EnvironmentConfig(
            enabled=True,
            scope="all_analysis",
            install_mode="auto_detect",
            install_fail_policy="mark_degraded",
            python_executable=sys.executable,
            bootstrap_tools=False,
        ),
    )

    orchestrator = await Orchestrator.create(config, ensure_schema=True)
    try:
        report = await orchestrator.run_workflow("maintenance_loop")
    finally:
        await orchestrator.close()

    assert report.static_result is not None
    assert report.dynamic_result is not None
    assert report.maintenance_result is not None
    assert report.maintenance_result.patch is not None
    assert report.maintenance_result.patch.file_patches
    assert (Path(report.report_dir) / "summary.md").exists()
    assert (Path(report.report_dir) / "findings.json").exists()
    assert (Path(report.report_dir) / "runtime" / ".venv").exists()
    assert (Path(report.report_dir) / "artifacts" / "environment.json").exists()
    assert (Path(report.report_dir) / "artifacts" / "install.log").exists()
    assert report.metadata["runtime_root"].startswith(str(Path(report.report_dir) / "runtime"))
    assert report.metadata["validation_workspace_root"].startswith(
        str(Path(report.report_dir) / "runtime" / "validation_workspace")
    )


@pytest.mark.asyncio
async def test_maintenance_loop_installs_src_requirements_into_report_local_env(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    repo_root.mkdir()
    deps_root = repo_root / "deps" / "helper_pkg"
    package_root = deps_root / "helper_pkg"
    package_root.mkdir(parents=True)
    (deps_root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "helper-pkg"
version = "0.1.0"
""".strip(),
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
    (repo_root / "src").mkdir()
    (repo_root / "src" / "requirements.txt").write_text(
        f"-e {deps_root}\n",
        encoding="utf-8",
    )
    (repo_root / "module.py").write_text("def noop():\n    return None\n", encoding="utf-8")

    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py", "src/requirements.txt", "pyproject.toml"],
        exclude=["state/**", "reports/**"],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(
            max_complexity=10,
            ruff_command=None,
            mypy_command=None,
            bandit_command=None,
        ),
        dynamic_debug=DynamicDebugConfig(
            smoke_commands=[],
            test_commands=[
                "python -c \"from helper_pkg import VALUE; import sys; sys.exit(0 if VALUE == 7 else 1)\""
            ],
            timeout_seconds=120,
        ),
        database=sqlite_database_config(tmp_path),
        environment=EnvironmentConfig(
            enabled=True,
            scope="all_analysis",
            install_mode="auto_detect",
            install_fail_policy="mark_degraded",
            python_executable=sys.executable,
            bootstrap_tools=False,
        ),
    )

    orchestrator = await Orchestrator.create(config, ensure_schema=True)
    try:
        report = await orchestrator.run_workflow("maintenance_loop")
    finally:
        await orchestrator.close()

    assert report.dynamic_result is not None
    assert all(finding.rule_id != "command-failed" for finding in report.dynamic_result.findings)
    environment_json = json.loads(
        (Path(report.report_dir) / "artifacts" / "environment.json").read_text(encoding="utf-8")
    )
    assert environment_json["detected_sources"] == ["src/requirements.txt"]
