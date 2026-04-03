from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from core.config import AppConfig
    from memory.state_store import StateStore


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentKind(str, Enum):
    MAINTENANCE = "maintenance"
    STATIC_REVIEW = "static_review"
    DYNAMIC_DEBUG = "dynamic_debug"


class TaskType(str, Enum):
    STATIC_REVIEW = "static_review"
    DYNAMIC_DEBUG = "dynamic_debug"
    MAINTENANCE = "maintenance"
    VALIDATION_STATIC = "validation_static"
    VALIDATION_DYNAMIC = "validation_dynamic"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkflowTrigger(str, Enum):
    PULL_REQUEST = "pull_request"
    ISSUE_COMMENT = "issue_comment"


class PublishMode(str, Enum):
    COMPANION_PR = "companion_pr"
    COMMENT_ONLY = "comment_only"
    ARTIFACT_ONLY = "artifact_only"


def build_fingerprint(
    rule_id: str,
    path: str | None,
    symbol: str | None,
    normalized_message: str,
) -> str:
    return "|".join(
        [
            rule_id or "unknown-rule",
            path or "",
            symbol or "",
            " ".join(normalized_message.strip().split()).lower(),
        ]
    )


@dataclass(slots=True)
class RepoSnapshot:
    repo_root: str
    scanned_at: datetime
    revision: str | None
    file_hashes: dict[str, str]

    @property
    def files(self) -> list[str]:
        return sorted(self.file_hashes)


@dataclass(slots=True)
class ChangeSet:
    changed_files: list[str]
    added_files: list[str]
    removed_files: list[str]
    baseline_revision: str | None = None
    current_revision: str | None = None
    reason: str = "hash-diff"

    @property
    def all_touched_files(self) -> list[str]:
        return sorted(set(self.changed_files + self.added_files + self.removed_files))


@dataclass(slots=True)
class Task:
    task_id: str
    run_id: str
    agent_kind: AgentKind
    task_type: TaskType
    targets: list[str]
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class Finding:
    source_agent: AgentKind
    severity: Severity
    rule_id: str
    message: str
    category: str
    path: str | None = None
    line: int | None = None
    symbol: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    state: str = "open"

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = build_fingerprint(
                self.rule_id,
                self.path,
                self.symbol,
                self.message,
            )


@dataclass(slots=True)
class FeedbackBundle:
    snapshot: RepoSnapshot
    change_set: ChangeSet
    static_findings: list[Finding] = field(default_factory=list)
    dynamic_findings: list[Finding] = field(default_factory=list)

    @property
    def all_findings(self) -> list[Finding]:
        return [*self.static_findings, *self.dynamic_findings]


@dataclass(slots=True)
class FilePatch:
    path: str
    old_content: str
    new_content: str
    diff: str


@dataclass(slots=True)
class PatchProposal:
    summary: str
    rationale: str
    file_patches: list[FilePatch] = field(default_factory=list)
    validation_targets: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    applied: bool = False
    diff_text: str = ""

    @property
    def touched_files(self) -> list[str]:
        return [patch.path for patch in self.file_patches]


@dataclass(slots=True)
class AgentResult:
    task_id: str
    agent_kind: AgentKind
    task_type: TaskType
    status: TaskStatus
    summary: str
    findings: list[Finding] = field(default_factory=list)
    patch: PatchProposal | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PullRequestContext:
    repo_full_name: str
    base_repo_full_name: str
    head_repo_full_name: str
    pr_number: int
    title: str
    html_url: str
    base_branch: str
    head_branch: str
    head_sha: str
    changed_files: list[str] = field(default_factory=list)
    trigger: WorkflowTrigger = WorkflowTrigger.PULL_REQUEST
    event_name: str = "pull_request"
    actor: str | None = None
    comment_body: str | None = None
    event_path: str | None = None

    @property
    def is_from_fork(self) -> bool:
        return self.base_repo_full_name != self.head_repo_full_name


@dataclass(slots=True)
class ReviewPayload:
    title: str
    body: str
    summary: str
    publish_mode: PublishMode
    inline_comments: list[dict[str, Any]] = field(default_factory=list)
    artifact_references: list["ArtifactReference"] = field(default_factory=list)


@dataclass(slots=True)
class CompanionPRPayload:
    head_branch: str
    base_branch: str
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    existing_pr_number: int | None = None
    existing_pr_url: str | None = None


@dataclass(slots=True)
class SafeFixPolicy:
    allowed_rule_ids: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "missing-module-docstring",
                "missing-final-newline",
                "trailing-whitespace",
                "excessive-eof-blank-lines",
                "D100",
                "W291",
                "W292",
                "W293",
                "W391",
            }
        )
    )

    def allows(self, rule_id: str) -> bool:
        return rule_id in self.allowed_rule_ids


@dataclass(slots=True)
class ArtifactReference:
    name: str
    path: str
    url: str | None = None
    fallback_url: str | None = None


@dataclass(slots=True)
class GitHubCapabilities:
    can_comment: bool = False
    can_inline_review: bool = False
    can_push_fix_branch: bool = False
    can_open_companion_pr: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PublishContext:
    run_id: str
    report_dir: str
    report_path: str
    artifact_name: str
    artifact_retention_days: int
    pr_context: PullRequestContext
    publish_mode: PublishMode
    safe_to_publish: bool
    branch_name: str | None = None
    companion_pr_required: bool = False
    publish_reasons: list[str] = field(default_factory=list)
    artifact_references: list[ArtifactReference] = field(default_factory=list)


class WorkflowState(TypedDict, total=False):
    run_id: str
    workflow_name: str
    started_at: datetime
    event_path: str
    pr_number_override: int
    snapshot: RepoSnapshot
    change_set: ChangeSet
    static_task: Task
    dynamic_task: Task
    maintenance_task: Task
    validation_static_task: Task
    validation_dynamic_task: Task
    static_result: AgentResult
    dynamic_result: AgentResult
    feedback: FeedbackBundle
    maintenance_result: AgentResult
    validation_static_result: AgentResult
    validation_dynamic_result: AgentResult
    validation_snapshot: RepoSnapshot
    validation_workspace_root: str
    pr_context: PullRequestContext
    pr_repo_root: str
    review_payload: ReviewPayload
    companion_pr_payload: CompanionPRPayload
    publish_context: PublishContext
    publish_mode: PublishMode
    safe_to_publish: bool
    companion_pr_required: bool
    capabilities: GitHubCapabilities
    artifacts: dict[str, Any]
    review_result: dict[str, Any]
    inline_comment_result: dict[str, Any]
    companion_pr_result: dict[str, Any]
    fix_branch_result: dict[str, Any]
    branch_name: str
    report_dir: str
    report: "WorkflowReport"
    errors: list[str]


@dataclass(slots=True)
class WorkflowReport:
    run_id: str
    workflow_name: str
    repo_root: str
    started_at: datetime
    finished_at: datetime
    status: TaskStatus
    snapshot: RepoSnapshot
    change_set: ChangeSet
    static_result: AgentResult | None = None
    dynamic_result: AgentResult | None = None
    maintenance_result: AgentResult | None = None
    validation_results: dict[str, AgentResult] = field(default_factory=dict)
    report_dir: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_findings(self) -> list[Finding]:
        findings: list[Finding] = []
        for result in [self.static_result, self.dynamic_result]:
            if result:
                findings.extend(result.findings)
        return findings


@dataclass(slots=True)
class RunContext:
    run_id: str
    repo_root: "Path"
    working_repo_root: "Path"
    config: "AppConfig"
    state_store: "StateStore"
    logger: "logging.Logger"
    rules: dict[str, Any] = field(default_factory=dict)
