from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any

from core.models import (
    AgentKind,
    AgentResult,
    ArtifactReference,
    ChangeSet,
    FilePatch,
    Finding,
    PatchProposal,
    PublishContext,
    PublishMode,
    PullRequestContext,
    RepoSnapshot,
    ReviewPayload,
    Severity,
    TaskStatus,
    TaskType,
    WorkflowReport,
    WorkflowTrigger,
)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


async def write_json(path: Path, value: Any) -> None:
    await asyncio.to_thread(_write_json_sync, path, value)


def _write_json_sync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


async def read_json(path: Path) -> Any:
    return await asyncio.to_thread(_read_json_sync, path)


def _read_json_sync(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_datetime(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _artifact_reference_from_dict(data: dict[str, Any]) -> ArtifactReference:
    return ArtifactReference(
        name=str(data["name"]),
        path=str(data["path"]),
        url=str(data["url"]) if data.get("url") else None,
        fallback_url=str(data["fallback_url"]) if data.get("fallback_url") else None,
    )


def _finding_from_dict(data: dict[str, Any]) -> Finding:
    return Finding(
        source_agent=AgentKind(data["source_agent"]),
        severity=Severity(data["severity"]),
        rule_id=str(data["rule_id"]),
        message=str(data["message"]),
        category=str(data["category"]),
        path=str(data["path"]) if data.get("path") else None,
        line=int(data["line"]) if data.get("line") is not None else None,
        symbol=str(data["symbol"]) if data.get("symbol") else None,
        evidence=dict(data.get("evidence", {})),
        fingerprint=str(data.get("fingerprint", "")),
        state=str(data.get("state", "open")),
    )


def _file_patch_from_dict(data: dict[str, Any]) -> FilePatch:
    return FilePatch(
        path=str(data["path"]),
        old_content=str(data["old_content"]),
        new_content=str(data["new_content"]),
        diff=str(data["diff"]),
    )


def _patch_from_dict(data: dict[str, Any]) -> PatchProposal:
    return PatchProposal(
        summary=str(data["summary"]),
        rationale=str(data["rationale"]),
        file_patches=[_file_patch_from_dict(item) for item in data.get("file_patches", [])],
        validation_targets=[str(item) for item in data.get("validation_targets", [])],
        suggestions=[str(item) for item in data.get("suggestions", [])],
        applied=bool(data.get("applied", False)),
        diff_text=str(data.get("diff_text", "")),
    )


def _agent_result_from_dict(data: dict[str, Any]) -> AgentResult:
    return AgentResult(
        task_id=str(data["task_id"]),
        agent_kind=AgentKind(data["agent_kind"]),
        task_type=TaskType(data["task_type"]),
        status=TaskStatus(data["status"]),
        summary=str(data["summary"]),
        findings=[_finding_from_dict(item) for item in data.get("findings", [])],
        patch=_patch_from_dict(data["patch"]) if data.get("patch") else None,
        artifacts=dict(data.get("artifacts", {})),
        errors=[str(item) for item in data.get("errors", [])],
    )


def _repo_snapshot_from_dict(data: dict[str, Any]) -> RepoSnapshot:
    return RepoSnapshot(
        repo_root=str(data["repo_root"]),
        scanned_at=_parse_datetime(str(data["scanned_at"])),
        revision=str(data["revision"]) if data.get("revision") else None,
        file_hashes={str(key): str(value) for key, value in dict(data.get("file_hashes", {})).items()},
    )


def _change_set_from_dict(data: dict[str, Any]) -> ChangeSet:
    return ChangeSet(
        changed_files=[str(item) for item in data.get("changed_files", [])],
        added_files=[str(item) for item in data.get("added_files", [])],
        removed_files=[str(item) for item in data.get("removed_files", [])],
        baseline_revision=str(data["baseline_revision"]) if data.get("baseline_revision") else None,
        current_revision=str(data["current_revision"]) if data.get("current_revision") else None,
        reason=str(data.get("reason", "hash-diff")),
    )


def _pull_request_context_from_dict(data: dict[str, Any]) -> PullRequestContext:
    return PullRequestContext(
        repo_full_name=str(data["repo_full_name"]),
        base_repo_full_name=str(data["base_repo_full_name"]),
        head_repo_full_name=str(data["head_repo_full_name"]),
        pr_number=int(data["pr_number"]),
        title=str(data["title"]),
        html_url=str(data["html_url"]),
        base_branch=str(data["base_branch"]),
        head_branch=str(data["head_branch"]),
        head_sha=str(data["head_sha"]),
        changed_files=[str(item) for item in data.get("changed_files", [])],
        trigger=WorkflowTrigger(data.get("trigger", WorkflowTrigger.PULL_REQUEST.value)),
        event_name=str(data.get("event_name", "pull_request")),
        actor=str(data["actor"]) if data.get("actor") else None,
        comment_body=str(data["comment_body"]) if data.get("comment_body") else None,
        event_path=str(data["event_path"]) if data.get("event_path") else None,
    )


def workflow_report_from_dict(data: dict[str, Any]) -> WorkflowReport:
    validation_results = {
        str(name): _agent_result_from_dict(result)
        for name, result in dict(data.get("validation_results", {})).items()
    }
    return WorkflowReport(
        run_id=str(data["run_id"]),
        workflow_name=str(data["workflow_name"]),
        repo_root=str(data["repo_root"]),
        started_at=_parse_datetime(str(data["started_at"])),
        finished_at=_parse_datetime(str(data["finished_at"])),
        status=TaskStatus(data["status"]),
        snapshot=_repo_snapshot_from_dict(data["snapshot"]),
        change_set=_change_set_from_dict(data["change_set"]),
        static_result=_agent_result_from_dict(data["static_result"]) if data.get("static_result") else None,
        dynamic_result=_agent_result_from_dict(data["dynamic_result"]) if data.get("dynamic_result") else None,
        maintenance_result=_agent_result_from_dict(data["maintenance_result"]) if data.get("maintenance_result") else None,
        validation_results=validation_results,
        report_dir=str(data.get("report_dir", "")),
        metadata=dict(data.get("metadata", {})),
    )


def publish_context_from_dict(data: dict[str, Any]) -> PublishContext:
    return PublishContext(
        run_id=str(data["run_id"]),
        report_dir=str(data["report_dir"]),
        report_path=str(data["report_path"]),
        artifact_name=str(data["artifact_name"]),
        artifact_retention_days=int(data["artifact_retention_days"]),
        pr_context=_pull_request_context_from_dict(data["pr_context"]),
        publish_mode=PublishMode(data["publish_mode"]),
        safe_to_publish=bool(data["safe_to_publish"]),
        branch_name=str(data["branch_name"]) if data.get("branch_name") else None,
        companion_pr_required=bool(data.get("companion_pr_required", False)),
        publish_reasons=[str(item) for item in data.get("publish_reasons", [])],
        artifact_references=[
            _artifact_reference_from_dict(item) for item in data.get("artifact_references", [])
        ],
    )


def review_payload_from_dict(data: dict[str, Any]) -> ReviewPayload:
    return ReviewPayload(
        title=str(data["title"]),
        body=str(data["body"]),
        summary=str(data["summary"]),
        publish_mode=PublishMode(data["publish_mode"]),
        inline_comments=[dict(item) for item in data.get("inline_comments", [])],
        artifact_references=[
            _artifact_reference_from_dict(item) for item in data.get("artifact_references", [])
        ],
    )
