from __future__ import annotations

from pathlib import Path

import pytest

from core.models import (
    AgentKind,
    AgentResult,
    Finding,
    PatchProposal,
    RepoSnapshot,
    Severity,
    Task,
    TaskStatus,
    TaskType,
    utc_now,
)
from memory.state_store import StateStore
from tests.support import sqlite_database_config


@pytest.mark.asyncio
async def test_state_store_requires_initialized_schema_when_not_ensuring_schema(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="aerich upgrade"):
        await StateStore.create(sqlite_database_config(tmp_path), ensure_schema=False)


@pytest.mark.asyncio
async def test_state_store_persists_snapshot_and_task(tmp_path: Path) -> None:
    store = await StateStore.create(sqlite_database_config(tmp_path), ensure_schema=True)
    try:
        run_id = await store.start_run("maintenance_loop", str(tmp_path))
        snapshot = RepoSnapshot(
            repo_root=str(tmp_path),
            scanned_at=utc_now(),
            revision="abc123",
            file_hashes={"a.py": "hash-a"},
        )
        await store.save_snapshot(run_id, snapshot)
        task = Task(
            task_id="task-1",
            run_id=run_id,
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.STATIC_REVIEW,
            targets=["a.py"],
            payload={},
        )
        await store.save_task(task)
        result = AgentResult(
            task_id=task.task_id,
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.STATIC_REVIEW,
            status=TaskStatus.SUCCEEDED,
            summary="stored",
            findings=[
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.LOW,
                    rule_id="missing-module-docstring",
                    message="Module docstring missing.",
                    category="documentation",
                    path="a.py",
                )
            ],
            patch=PatchProposal(
                summary="patch",
                rationale="test patch",
                suggestions=["follow-up"],
                diff_text="diff",
            ),
        )
        await store.save_agent_result(run_id, task, result)
        await store.upsert_issue(str(tmp_path), result.findings[0], run_id)
        await store.finish_run(run_id, TaskStatus.SUCCEEDED, "done")

        latest_snapshot = await store.get_latest_snapshot(str(tmp_path))
        latest_run = await store.latest_run(str(tmp_path))
        recent_runs = await store.recent_runs(str(tmp_path))
    finally:
        await store.close()

    assert latest_snapshot is not None
    assert latest_snapshot.revision == "abc123"
    assert latest_run is not None
    assert latest_run["run_id"] == run_id
    assert latest_run["status"] == TaskStatus.SUCCEEDED.value
    assert recent_runs[0]["run_id"] == run_id
