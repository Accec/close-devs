from __future__ import annotations

from pathlib import Path
import sys

import pytest

from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
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
            test_commands=[f"{sys.executable} -m pytest -q"],
            timeout_seconds=60,
        ),
        database=sqlite_database_config(tmp_path),
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
