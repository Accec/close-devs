from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agents.maintenance import MaintenanceAgent
from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import (
    AgentKind,
    ChangeSet,
    FeedbackBundle,
    Finding,
    RepoSnapshot,
    RunContext,
    Severity,
    Task,
    TaskType,
    utc_now,
)
from memory.state_store import StateStore
from tests.support import sqlite_database_config


@pytest.mark.asyncio
async def test_maintenance_agent_generates_docstring_and_newline_patch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    module_path = repo_root / "module.py"
    module_path.write_text("def add(a, b):    \n    return a + b\n\n\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={"module.py": "hash"},
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["module.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        static_findings=[
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="missing-module-docstring",
                message="Python module is missing a top-level docstring.",
                category="documentation",
                path="module.py",
            ),
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="W291",
                message="trailing whitespace",
                category="style",
                path="module.py",
            ),
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="W391",
                message="too many blank lines at end of file",
                category="style",
                path="module.py",
            ),
        ],
    )
    task = Task(
        task_id="task-1",
        run_id="run-1",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["module.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-1",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches
    new_content = result.patch.file_patches[0].new_content
    assert '"""Maintained by Close-Devs."""' in new_content
    assert new_content.endswith("\n")
    assert "def add(a, b):\n" in new_content
    assert not new_content.endswith("\n\n")
