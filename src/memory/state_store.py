from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.config import DatabaseConfig
from core.models import AgentResult, Finding, PatchProposal, RepoSnapshot, Task, TaskStatus
from memory.database import close_database, init_database
from memory.orm_models import (
    FindingRecord,
    IssueCatalogRecord,
    PatchRecord,
    RunRecord,
    SnapshotRecord,
    TaskRecord,
)


class StateStore:
    def __init__(self, database: DatabaseConfig) -> None:
        self.database = database

    @classmethod
    async def create(
        cls,
        database: DatabaseConfig,
        *,
        ensure_schema: bool = False,
    ) -> "StateStore":
        store = cls(database)
        await store.initialize(ensure_schema=ensure_schema)
        return store

    async def initialize(self, *, ensure_schema: bool = False) -> None:
        await init_database(self.database, ensure_schema=ensure_schema)

    async def close(self) -> None:
        await close_database()

    async def start_run(self, workflow_name: str, repo_root: str, started_at: datetime | None = None) -> str:
        run_id = uuid4().hex
        await RunRecord.create(
            run_id=run_id,
            workflow_name=workflow_name,
            repo_root=repo_root,
            started_at=started_at or datetime.now(timezone.utc),
            status=TaskStatus.RUNNING.value,
            summary="",
            report_dir=None,
        )
        return run_id

    async def finish_run(
        self,
        run_id: str,
        status: TaskStatus,
        summary: str,
        report_dir: str | None = None,
    ) -> None:
        await RunRecord.filter(run_id=run_id).update(
            finished_at=datetime.now(timezone.utc),
            status=status.value,
            summary=summary,
            report_dir=report_dir,
        )

    async def save_snapshot(self, run_id: str, snapshot: RepoSnapshot) -> None:
        await SnapshotRecord.update_or_create(
            run_id=run_id,
            defaults={
                "repo_root": snapshot.repo_root,
                "scanned_at": snapshot.scanned_at,
                "revision": snapshot.revision,
                "file_hashes": snapshot.file_hashes,
            },
        )

    async def get_latest_snapshot(self, repo_root: str) -> RepoSnapshot | None:
        record = await SnapshotRecord.filter(repo_root=repo_root).order_by("-scanned_at").first()
        if record is None:
            return None
        return RepoSnapshot(
            repo_root=record.repo_root,
            scanned_at=record.scanned_at,
            revision=record.revision,
            file_hashes=dict(record.file_hashes or {}),
        )

    async def save_task(
        self,
        task: Task,
        status: TaskStatus = TaskStatus.PENDING,
        summary: str = "",
    ) -> None:
        payload_summary = {
            "payload_keys": sorted(task.payload.keys()),
            "target_count": len(task.targets),
        }
        await TaskRecord.update_or_create(
            task_id=task.task_id,
            defaults={
                "run_id": task.run_id,
                "agent_kind": task.agent_kind.value,
                "task_type": task.task_type.value,
                "targets": task.targets,
                "payload_summary": payload_summary,
                "created_at": task.created_at,
                "status": status.value,
                "summary": summary,
            },
        )

    async def update_task(self, task_id: str, status: TaskStatus, summary: str) -> None:
        await TaskRecord.filter(task_id=task_id).update(status=status.value, summary=summary)

    async def save_agent_result(self, run_id: str, task: Task, result: AgentResult) -> None:
        await self.update_task(task.task_id, result.status, result.summary)
        await self.save_findings(run_id, task.task_id, result.findings)
        if result.patch is not None:
            await self.save_patch(run_id, result.patch)

    async def save_findings(self, run_id: str, task_id: str, findings: list[Finding]) -> None:
        if not findings:
            return
        records = [
            FindingRecord(
                run_id=run_id,
                task_id=task_id,
                agent_kind=finding.source_agent.value,
                severity=finding.severity.value,
                rule_id=finding.rule_id,
                category=finding.category,
                path=finding.path,
                line=finding.line,
                symbol=finding.symbol,
                message=finding.message,
                fingerprint=finding.fingerprint,
                evidence=finding.evidence,
                state=finding.state,
            )
            for finding in findings
        ]
        await FindingRecord.bulk_create(records)

    async def save_patch(self, run_id: str, patch: PatchProposal) -> None:
        file_patches = [
            {
                "path": file_patch.path,
                "old_content": file_patch.old_content,
                "new_content": file_patch.new_content,
                "diff": file_patch.diff,
            }
            for file_patch in patch.file_patches
        ]
        await PatchRecord.update_or_create(
            run_id=run_id,
            defaults={
                "summary": patch.summary,
                "rationale": patch.rationale,
                "touched_files": patch.touched_files,
                "diff_text": patch.diff_text,
                "suggestions": patch.suggestions,
                "applied": patch.applied,
                "file_patches": file_patches,
            },
        )

    async def upsert_issue(self, repo_root: str, finding: Finding, run_id: str, status: str = "open") -> None:
        record = await IssueCatalogRecord.get_or_none(
            repo_root=repo_root,
            fingerprint=finding.fingerprint,
        )
        if record is None:
            await IssueCatalogRecord.create(
                repo_root=repo_root,
                fingerprint=finding.fingerprint,
                rule_id=finding.rule_id,
                path=finding.path,
                message=finding.message,
                status=status,
                first_seen_run=run_id,
                last_seen_run=run_id,
                occurrences=1,
            )
            return

        record.status = status
        record.last_seen_run = run_id
        record.occurrences += 1
        record.message = finding.message
        record.path = finding.path
        record.rule_id = finding.rule_id
        await record.save()

    async def set_issue_status(self, repo_root: str, fingerprint: str, status: str, run_id: str) -> None:
        await IssueCatalogRecord.filter(
            repo_root=repo_root,
            fingerprint=fingerprint,
        ).update(status=status, last_seen_run=run_id)

    async def latest_run(self, repo_root: str) -> dict[str, Any] | None:
        record = await RunRecord.filter(repo_root=repo_root).order_by("-started_at").first()
        return self._serialize_run(record) if record is not None else None

    async def recent_runs(self, repo_root: str, limit: int = 20) -> list[dict[str, Any]]:
        records = await RunRecord.filter(repo_root=repo_root).order_by("-started_at").limit(limit)
        return [self._serialize_run(record) for record in records]

    def _serialize_run(self, record: RunRecord) -> dict[str, Any]:
        return {
            "run_id": record.run_id,
            "workflow_name": record.workflow_name,
            "repo_root": record.repo_root,
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "status": record.status,
            "summary": record.summary,
            "report_dir": record.report_dir,
        }
