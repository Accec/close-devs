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
    assert (Path(report.report_dir) / "artifacts" / "dependency_vulnerabilities.json").exists()
    dependency_artifact = json.loads(
        (Path(report.report_dir) / "artifacts" / "dependency_vulnerabilities.json").read_text(encoding="utf-8")
    )
    assert dependency_artifact["status"] in {"executed", "unavailable", "failed", "disabled"}
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


@pytest.mark.asyncio
async def test_maintenance_loop_reports_static_context_summary_for_large_target_sets(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    for index in range(24):
        (repo_root / "src" / "app" / f"module_{index:02d}.py").write_text(
            f"def value_{index}():\n    return {index}\n",
            encoding="utf-8",
        )

    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py", "pyproject.toml"],
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
            test_commands=["python -c \"import sys; sys.exit(0)\""],
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

    assert report.metadata["static_context_summary"]["enabled"] is True
    assert report.metadata["static_context_summary"]["top_target_count"] > 12
    assert report.static_result is not None
    assert report.static_result.artifacts["top_target_count"] > 12


@pytest.mark.asyncio
async def test_maintenance_loop_creates_env_template_for_missing_environment_variable(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")

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
                (
                    "python -c \"from pathlib import Path; import os; "
                    "example = Path('.env.example'); "
                    "env_lines = example.read_text().splitlines() if example.exists() else []; "
                    "has_placeholder = any(line.startswith('APP_SECRET=') for line in env_lines); "
                    "assert 'APP_SECRET' in os.environ or has_placeholder, 'Missing environment variable: APP_SECRET'\""
                )
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

    assert report.maintenance_result is not None
    assert report.maintenance_result.patch is not None
    assert ".env.example" in report.maintenance_result.patch.touched_files
    assert report.metadata["resolution_summary"]["repo_healthy"] is True
    assert report.metadata["auto_fixed_blockers"] == ["config"]
    assert not report.validation_results["dynamic_validation"].findings


@pytest.mark.asyncio
async def test_maintenance_loop_repairs_pytest_bootstrap_for_src_layout(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py", "pyproject.toml"],
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
            test_commands=["python -m pytest -q"],
            timeout_seconds=120,
        ),
        database=sqlite_database_config(tmp_path),
        environment=EnvironmentConfig(
            enabled=True,
            scope="all_analysis",
            install_mode="auto_detect",
            install_fail_policy="mark_degraded",
            python_executable=sys.executable,
            bootstrap_tools=True,
        ),
    )

    orchestrator = await Orchestrator.create(config, ensure_schema=True)
    try:
        report = await orchestrator.run_workflow("maintenance_loop")
    finally:
        await orchestrator.close()

    assert report.maintenance_result is not None
    assert report.maintenance_result.patch is not None
    patched_files = {file_patch.path for file_patch in report.maintenance_result.patch.file_patches}
    assert "pyproject.toml" in patched_files
    assert report.validation_results["dynamic_validation"].status.value == "succeeded"
    assert not report.validation_results["dynamic_validation"].findings
    assert report.metadata["resolution_summary"]["repo_healthy"] is True
    assert report.metadata["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_loop_reports_unresolved_startup_argument_blocker(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")

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
                "python -c \"import argparse; parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True); parser.parse_args()\""
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

    assert report.status.value == "succeeded"
    assert report.metadata["resolution_summary"]["repo_healthy"] is False
    assert report.metadata["resolution_summary"]["workflow_succeeded_but_repo_unhealthy"] is True
    assert "startup" in report.metadata["resolution_summary"]["root_blocker_classes"]
    assert report.metadata["resolution_summary"]["root_blockers_by_class"]["startup"] >= 1
    assert any(item.get("root_cause_class") == "startup" for item in report.metadata["root_blockers"])


@pytest.mark.asyncio
async def test_maintenance_loop_keeps_static_startup_topology_as_advisory_when_runtime_is_clean(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "sample_repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
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
            test_commands=['python -c "print(\'ok\')"'],
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

    assert report.metadata["confirmed_startup_blockers"] == []
    assert report.metadata["advisory_startup_handoffs"]
    assert any(
        item.get("entrypoint_path") == "manage.py"
        for item in report.metadata["advisory_startup_handoffs"]
    )
    assert report.metadata["startup_topology_summary"]["entrypoint_count"] >= 2
    assert "django_manage" in report.metadata["startup_topology_summary"]["contexts"]
    assert "startup" not in report.metadata.get("auto_fixed_blockers", [])
