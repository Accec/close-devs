from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.config import DatabaseConfig
from core.models import (
    AgentKind,
    AgentResult,
    AgentReflection,
    AgentSession,
    AgentStep,
    CompletionReason,
    Finding,
    FixRequest,
    PatchProposal,
    RepoSnapshot,
    SkillBinding,
    SkillCandidate,
    SkillCandidateStatus,
    SkillEvaluation,
    SkillPack,
    SkillPolicy,
    SkillSource,
    Task,
    TaskStatus,
    ToolCall,
)
from memory.database import close_database, init_database
from memory.orm_models import (
    AgentHandoffRecord,
    AgentReflectionRecord,
    AgentSessionRecord,
    AgentStepRecord,
    FindingRecord,
    IssueCatalogRecord,
    PatchRecord,
    RunRecord,
    SkillBindingRecord,
    SkillCandidateRecord,
    SkillEvaluationRecord,
    SkillPackRecord,
    SnapshotRecord,
    TaskRecord,
    ToolCallRecord,
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
        handoffs = [
            item for item in result.artifacts.get("handoffs", [])
            if isinstance(item, dict)
        ]
        if handoffs:
            await self.save_handoffs(
                run_id=run_id,
                task_id=task.task_id,
                session_id=str(result.artifacts.get("session_id", "")),
                handoffs=handoffs,
            )
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

    async def start_agent_session(self, session: AgentSession) -> None:
        await AgentSessionRecord.create(
            session_id=session.session_id,
            run_id=session.run_id,
            task_id=session.task_id,
            agent_kind=session.agent_kind.value,
            task_type=session.task_type.value,
            working_repo_root=session.working_repo_root,
            objective=session.objective,
            started_at=session.started_at,
            step_count=0,
            tool_call_count=0,
            active_skill_version=session.active_skill_version,
            candidate_skill_version=session.candidate_skill_version,
            skill_profile_hash=session.skill_profile_hash,
        )

    async def save_agent_step(self, session_id: str, step: AgentStep) -> None:
        await AgentStepRecord.create(
            session_id=session_id,
            step_index=step.step_index,
            decision_summary=step.decision_summary,
            action_type=step.action_type.value,
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            final_response=step.final_response,
            created_at=step.created_at,
        )

    async def save_tool_call(self, session_id: str, tool_call: ToolCall) -> None:
        await ToolCallRecord.create(
            session_id=session_id,
            step_index=tool_call.step_index,
            tool_name=tool_call.tool_name,
            status=tool_call.status,
            tool_input=tool_call.tool_input,
            output=tool_call.output,
            summary=tool_call.summary,
            error=tool_call.error,
            active_skill_version=tool_call.active_skill_version,
            candidate_skill_version=tool_call.candidate_skill_version,
            skill_profile_hash=tool_call.skill_profile_hash,
            started_at=tool_call.started_at,
            finished_at=tool_call.finished_at,
        )

    async def finish_agent_session(
        self,
        session_id: str,
        completion_reason: CompletionReason,
        *,
        summary: str,
        step_count: int,
        tool_call_count: int,
    ) -> None:
        await AgentSessionRecord.filter(session_id=session_id).update(
            finished_at=datetime.now(timezone.utc),
            completion_reason=completion_reason.value,
            summary=summary,
            step_count=step_count,
            tool_call_count=tool_call_count,
        )

    async def save_handoffs(
        self,
        *,
        run_id: str,
        task_id: str,
        session_id: str,
        handoffs: list[dict[str, Any]],
    ) -> None:
        if not handoffs:
            return
        await AgentHandoffRecord.filter(session_id=session_id).delete()
        records = [
            AgentHandoffRecord(
                session_id=session_id,
                run_id=run_id,
                task_id=task_id,
                source_agent=str(item.get("source_agent", "")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                recommended_change=str(item.get("recommended_change", "")),
                severity=str(item.get("severity", "low")),
                affected_files=[str(path) for path in item.get("affected_files", [])],
                evidence=list(item.get("evidence", [])),
                metadata=dict(item.get("metadata", {})),
            )
            for item in handoffs
        ]
        await AgentHandoffRecord.bulk_create(records)

    async def upsert_skill_pack(self, repo_root: str, pack: SkillPack) -> None:
        await SkillPackRecord.update_or_create(
            repo_root=repo_root,
            agent_kind=pack.agent_kind.value,
            version=pack.version,
            defaults={
                "name": pack.name,
                "description": pack.description,
                "status": pack.status,
                "source": pack.source.value,
                "system_prompt": pack.system_prompt,
                "skill_markdown": pack.skill_markdown,
                "examples": pack.examples,
                "policy": self._serialize_skill_policy(pack.policy),
                "profile_hash": pack.profile_hash,
            },
        )

    async def get_skill_pack(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        version: str,
    ) -> SkillPack | None:
        record = await SkillPackRecord.get_or_none(
            repo_root=repo_root,
            agent_kind=agent_kind.value,
            version=version,
        )
        return self._deserialize_skill_pack(record) if record is not None else None

    async def get_skill_binding(
        self,
        repo_root: str,
        agent_kind: AgentKind,
    ) -> SkillBinding | None:
        record = await SkillBindingRecord.get_or_none(
            repo_root=repo_root,
            agent_kind=agent_kind.value,
        )
        return self._deserialize_skill_binding(record) if record is not None else None

    async def set_skill_binding(self, binding: SkillBinding) -> None:
        await SkillBindingRecord.update_or_create(
            repo_root=binding.repo_root,
            agent_kind=binding.agent_kind.value,
            defaults={
                "active_version": binding.active_version,
                "source": binding.source.value,
                "frozen": binding.frozen,
                "updated_at": binding.updated_at,
            },
        )

    async def set_skill_binding_frozen(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        frozen: bool,
    ) -> None:
        await SkillBindingRecord.filter(
            repo_root=repo_root,
            agent_kind=agent_kind.value,
        ).update(
            frozen=frozen,
            updated_at=datetime.now(timezone.utc),
        )

    async def save_skill_candidate(self, candidate: SkillCandidate) -> None:
        await SkillCandidateRecord.update_or_create(
            candidate_id=candidate.candidate_id,
            defaults={
                "repo_root": candidate.repo_root,
                "agent_kind": candidate.agent_kind.value,
                "based_on_version": candidate.based_on_version,
                "version": candidate.version,
                "status": candidate.status.value,
                "shadow_runs": candidate.shadow_runs,
                "notes": candidate.notes,
                "skill_payload": self._serialize_skill_pack(candidate.skill_pack),
                "created_at": candidate.created_at,
            },
        )

    async def get_open_skill_candidate(
        self,
        repo_root: str,
        agent_kind: AgentKind,
    ) -> SkillCandidate | None:
        record = (
            await SkillCandidateRecord.filter(
                repo_root=repo_root,
                agent_kind=agent_kind.value,
                status=SkillCandidateStatus.CANDIDATE.value,
            )
            .order_by("-created_at")
            .first()
        )
        return self._deserialize_skill_candidate(record) if record is not None else None

    async def update_skill_candidate(
        self,
        candidate_id: str,
        *,
        status: SkillCandidateStatus | None = None,
        shadow_runs: int | None = None,
        notes: list[str] | None = None,
    ) -> None:
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status.value
        if shadow_runs is not None:
            updates["shadow_runs"] = shadow_runs
        if notes is not None:
            updates["notes"] = notes
        if updates:
            await SkillCandidateRecord.filter(candidate_id=candidate_id).update(**updates)

    async def save_skill_evaluation(self, evaluation: SkillEvaluation) -> None:
        await SkillEvaluationRecord.update_or_create(
            evaluation_id=evaluation.evaluation_id,
            defaults={
                "repo_root": evaluation.repo_root,
                "agent_kind": evaluation.agent_kind.value,
                "run_id": evaluation.run_id,
                "active_version": evaluation.active_version,
                "candidate_version": evaluation.candidate_version,
                "active_score": evaluation.active_score,
                "candidate_score": evaluation.candidate_score,
                "promoted": evaluation.promoted,
                "reasons": evaluation.reasons,
                "created_at": evaluation.created_at,
            },
        )

    async def recent_skill_evaluations(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        limit: int = 20,
    ) -> list[SkillEvaluation]:
        records = (
            await SkillEvaluationRecord.filter(
                repo_root=repo_root,
                agent_kind=agent_kind.value,
            )
            .order_by("-created_at")
            .limit(limit)
        )
        return [self._deserialize_skill_evaluation(record) for record in records]

    async def save_agent_reflection(self, reflection: AgentReflection) -> None:
        await AgentReflectionRecord.update_or_create(
            reflection_id=reflection.reflection_id,
            defaults={
                "repo_root": reflection.repo_root,
                "run_id": reflection.run_id,
                "task_id": reflection.task_id,
                "session_id": reflection.session_id,
                "agent_kind": reflection.agent_kind.value,
                "skill_version": reflection.skill_version,
                "summary": reflection.summary,
                "metrics": reflection.metrics,
                "upgrade_hints": reflection.upgrade_hints,
                "created_at": reflection.created_at,
            },
        )

    async def recent_agent_reflections(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        limit: int = 20,
    ) -> list[AgentReflection]:
        records = (
            await AgentReflectionRecord.filter(
                repo_root=repo_root,
                agent_kind=agent_kind.value,
            )
            .order_by("-created_at")
            .limit(limit)
        )
        return [self._deserialize_agent_reflection(record) for record in records]

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

    def _serialize_skill_policy(self, policy: SkillPolicy) -> dict[str, Any]:
        return {
            "planning_heuristics": list(policy.planning_heuristics),
            "tool_preferences": list(policy.tool_preferences),
            "forbidden_ordering": list(policy.forbidden_ordering),
            "severity_bias": dict(policy.severity_bias),
            "rule_weights": dict(policy.rule_weights),
            "command_preferences": list(policy.command_preferences),
            "completion_checklist": list(policy.completion_checklist),
            "handoff_style": policy.handoff_style,
            "patch_style": policy.patch_style,
            "recommended_max_steps": policy.recommended_max_steps,
            "recommended_max_tool_calls": policy.recommended_max_tool_calls,
            "recommended_max_wall_time_seconds": policy.recommended_max_wall_time_seconds,
            "allowed_tools": list(policy.allowed_tools),
            "environment_preferences": list(policy.environment_preferences),
            "upgrade_constraints": list(policy.upgrade_constraints),
            "noise_suppression": list(policy.noise_suppression),
        }

    def _deserialize_skill_policy(self, data: dict[str, Any]) -> SkillPolicy:
        return SkillPolicy(
            planning_heuristics=[str(item) for item in data.get("planning_heuristics", [])],
            tool_preferences=[str(item) for item in data.get("tool_preferences", [])],
            forbidden_ordering=[str(item) for item in data.get("forbidden_ordering", [])],
            severity_bias={str(key): float(value) for key, value in dict(data.get("severity_bias", {})).items()},
            rule_weights={str(key): float(value) for key, value in dict(data.get("rule_weights", {})).items()},
            command_preferences=[str(item) for item in data.get("command_preferences", [])],
            completion_checklist=[str(item) for item in data.get("completion_checklist", [])],
            handoff_style=str(data.get("handoff_style", "")),
            patch_style=str(data.get("patch_style", "")),
            recommended_max_steps=int(data["recommended_max_steps"]) if data.get("recommended_max_steps") is not None else None,
            recommended_max_tool_calls=int(data["recommended_max_tool_calls"]) if data.get("recommended_max_tool_calls") is not None else None,
            recommended_max_wall_time_seconds=int(data["recommended_max_wall_time_seconds"]) if data.get("recommended_max_wall_time_seconds") is not None else None,
            allowed_tools=[str(item) for item in data.get("allowed_tools", [])],
            environment_preferences=[str(item) for item in data.get("environment_preferences", [])],
            upgrade_constraints=[str(item) for item in data.get("upgrade_constraints", [])],
            noise_suppression=[str(item) for item in data.get("noise_suppression", [])],
        )

    def _serialize_skill_pack(self, pack: SkillPack) -> dict[str, Any]:
        return {
            "agent_kind": pack.agent_kind.value,
            "name": pack.name,
            "version": pack.version,
            "description": pack.description,
            "status": pack.status,
            "source": pack.source.value,
            "system_prompt": pack.system_prompt,
            "skill_markdown": pack.skill_markdown,
            "examples": list(pack.examples),
            "policy": self._serialize_skill_policy(pack.policy),
            "profile_hash": pack.profile_hash,
        }

    def _deserialize_skill_pack(self, record: SkillPackRecord | dict[str, Any]) -> SkillPack:
        data = record if isinstance(record, dict) else {
            "agent_kind": record.agent_kind,
            "name": record.name,
            "version": record.version,
            "description": record.description,
            "status": record.status,
            "source": record.source,
            "system_prompt": record.system_prompt,
            "skill_markdown": record.skill_markdown,
            "examples": record.examples,
            "policy": record.policy,
            "profile_hash": record.profile_hash,
        }
        return SkillPack(
            agent_kind=AgentKind(str(data["agent_kind"])),
            name=str(data["name"]),
            version=str(data["version"]),
            description=str(data.get("description", "")),
            status=str(data.get("status", "active")),
            source=SkillSource(str(data.get("source", SkillSource.REPO.value))),
            system_prompt=str(data.get("system_prompt", "")),
            skill_markdown=str(data.get("skill_markdown", "")),
            examples=[dict(item) for item in data.get("examples", [])],
            policy=self._deserialize_skill_policy(dict(data.get("policy", {}))),
            profile_hash=str(data.get("profile_hash", "")),
        )

    def _deserialize_skill_binding(self, record: SkillBindingRecord) -> SkillBinding:
        return SkillBinding(
            repo_root=record.repo_root,
            agent_kind=AgentKind(record.agent_kind),
            active_version=record.active_version,
            source=SkillSource(record.source),
            frozen=bool(record.frozen),
            updated_at=record.updated_at,
        )

    def _deserialize_skill_candidate(self, record: SkillCandidateRecord) -> SkillCandidate:
        return SkillCandidate(
            candidate_id=record.candidate_id,
            repo_root=record.repo_root,
            agent_kind=AgentKind(record.agent_kind),
            based_on_version=record.based_on_version,
            version=record.version,
            skill_pack=self._deserialize_skill_pack(dict(record.skill_payload or {})),
            status=SkillCandidateStatus(record.status),
            created_at=record.created_at,
            shadow_runs=int(record.shadow_runs),
            notes=[str(item) for item in record.notes or []],
        )

    def _deserialize_skill_evaluation(self, record: SkillEvaluationRecord) -> SkillEvaluation:
        return SkillEvaluation(
            evaluation_id=record.evaluation_id,
            repo_root=record.repo_root,
            agent_kind=AgentKind(record.agent_kind),
            run_id=record.run_id,
            active_version=record.active_version,
            candidate_version=record.candidate_version,
            active_score=float(record.active_score),
            candidate_score=float(record.candidate_score) if record.candidate_score is not None else None,
            promoted=bool(record.promoted),
            reasons=[str(item) for item in record.reasons or []],
            created_at=record.created_at,
        )

    def _deserialize_agent_reflection(self, record: AgentReflectionRecord) -> AgentReflection:
        return AgentReflection(
            reflection_id=record.reflection_id,
            repo_root=record.repo_root,
            run_id=record.run_id,
            task_id=record.task_id,
            session_id=record.session_id,
            agent_kind=AgentKind(record.agent_kind),
            skill_version=record.skill_version,
            summary=record.summary,
            metrics=dict(record.metrics or {}),
            upgrade_hints=[str(item) for item in record.upgrade_hints or []],
            created_at=record.created_at,
        )
