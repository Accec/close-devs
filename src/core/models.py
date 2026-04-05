from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import os
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


class AgentActionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINALIZE = "finalize"


class CompletionReason(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_WALL_TIME = "max_wall_time"
    MAX_CONSECUTIVE_FAILURES = "max_consecutive_failures"
    FAILED = "failed"


class SkillSource(str, Enum):
    REPO = "repo"
    DATABASE = "database"


class SkillCandidateStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FROZEN = "frozen"


class SkillEvaluationMode(str, Enum):
    HEURISTIC = "heuristic"
    REPLAY = "replay"


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
class ExecutionEnvironment:
    report_dir: str
    runtime_root: str
    base_workspace_root: str
    maintenance_workspace_root: str
    validation_workspace_root: str
    venv_root: str
    python_bin: str
    bin_dir: str
    status: str = "ready"
    detected_sources: list[str] = field(default_factory=list)
    install_commands: list[str] = field(default_factory=list)
    install_errors: list[str] = field(default_factory=list)
    bootstrap_packages: list[str] = field(default_factory=list)
    installer_summary: dict[str, str] = field(default_factory=dict)
    install_log_path: str | None = None
    environment_json_path: str | None = None

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    def command_env(self) -> dict[str, str]:
        env = dict(os.environ)
        current_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join(
            [self.bin_dir, current_path] if current_path else [self.bin_dir]
        )
        env["VIRTUAL_ENV"] = self.venv_root
        return env


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
class StartupConfigAnchor:
    path: str
    context: str
    anchor_type: str
    status: str
    repair_hint: str | None = None


@dataclass(slots=True)
class StartupEntrypoint:
    path: str
    context: str
    config_anchor_path: str | None = None
    repair_hint: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StartupTopology:
    repo_root: str
    src_layout: bool = False
    entrypoints: list[StartupEntrypoint] = field(default_factory=list)
    config_anchors: list[StartupConfigAnchor] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LanguageProfile:
    primary_language: str = "unknown"
    languages: list[str] = field(default_factory=list)
    primary_ecosystem: str = "generic"
    ecosystems: list[str] = field(default_factory=list)
    language_file_counts: dict[str, int] = field(default_factory=dict)
    enabled_adapters: list[str] = field(default_factory=list)
    generic_review: bool = False


@dataclass(slots=True)
class ProjectEntrypoint:
    path: str
    context: str
    language: str
    ecosystem: str
    config_anchor_path: str | None = None
    repair_hint: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProjectConfigAnchor:
    path: str
    context: str
    language: str
    ecosystem: str
    anchor_type: str
    status: str
    repair_hint: str | None = None


@dataclass(slots=True)
class ProjectTopology:
    repo_root: str
    languages: list[str] = field(default_factory=list)
    ecosystems: list[str] = field(default_factory=list)
    entrypoints: list[ProjectEntrypoint] = field(default_factory=list)
    config_anchors: list[ProjectConfigAnchor] = field(default_factory=list)
    dependency_manifests: list[str] = field(default_factory=list)
    lockfiles: list[str] = field(default_factory=list)
    package_roots: list[str] = field(default_factory=list)
    test_roots: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LanguageAdapter:
    language: str
    ecosystems: list[str] = field(default_factory=list)
    entrypoint_markers: list[str] = field(default_factory=list)
    config_markers: list[str] = field(default_factory=list)
    manifest_markers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StaticToolAdapter:
    name: str
    language: str
    ecosystem: str
    command: str
    parser: str
    pass_targets: bool = True


@dataclass(slots=True)
class DependencyAuditAdapter:
    ecosystem: str
    command: str
    parser: str


@dataclass(slots=True)
class RepoMapSummary:
    package_roots: list[str] = field(default_factory=list)
    python_package_roots: list[str] = field(default_factory=list)
    test_roots: list[str] = field(default_factory=list)
    src_layout: bool = False
    startup_contexts: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    ecosystems: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    dependency_manifests: list[str] = field(default_factory=list)
    lockfiles: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolCoverageSummary:
    enabled_tools: list[str] = field(default_factory=list)
    unavailable_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    tool_statuses: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class StaticTargetDigest:
    prioritized_targets: list[str] = field(default_factory=list)
    top_targets: list[str] = field(default_factory=list)
    high_signal_targets: list[str] = field(default_factory=list)
    low_signal_targets: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImportAdjacencyDigest:
    related_files: list[str] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class BaselineStaticDigest:
    total_findings: int = 0
    severity_counts: dict[str, int] = field(default_factory=dict)
    noisy_rule_counts: dict[str, int] = field(default_factory=dict)
    top_findings: list[dict[str, Any]] = field(default_factory=list)
    candidate_high_value_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StaticContextBundle:
    startup_topology: StartupTopology
    project_topology: ProjectTopology
    repo_map_summary: RepoMapSummary
    language_profile: LanguageProfile = field(default_factory=LanguageProfile)
    top_targets: list[str] = field(default_factory=list)
    high_signal_targets: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    target_digest: StaticTargetDigest = field(default_factory=StaticTargetDigest)
    import_adjacency_digest: ImportAdjacencyDigest = field(default_factory=ImportAdjacencyDigest)
    config_anchor_digest: list[dict[str, Any]] = field(default_factory=list)
    baseline_static_digest: BaselineStaticDigest = field(default_factory=BaselineStaticDigest)
    baseline_tool_digest: BaselineStaticDigest = field(default_factory=BaselineStaticDigest)
    tool_coverage_summary: ToolCoverageSummary = field(default_factory=ToolCoverageSummary)
    enabled: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "primary_language": self.language_profile.primary_language,
            "languages": list(self.language_profile.languages),
            "primary_ecosystem": self.language_profile.primary_ecosystem,
            "ecosystems": list(self.language_profile.ecosystems),
            "generic_review": self.language_profile.generic_review,
            "startup_context_count": len(
                {
                    entry.context
                    for entry in self.startup_topology.entrypoints
                }
                | {
                    anchor.context
                    for anchor in self.startup_topology.config_anchors
                }
            ),
            "prioritized_target_count": len(self.target_digest.prioritized_targets),
            "top_target_count": len(self.top_targets),
            "high_signal_target_count": len(self.high_signal_targets),
            "related_file_count": len(self.related_files),
            "baseline_total_findings": self.baseline_static_digest.total_findings,
            "baseline_severity_counts": dict(self.baseline_static_digest.severity_counts),
            "baseline_noisy_rule_counts": dict(self.baseline_static_digest.noisy_rule_counts),
            "tool_coverage_summary": {
                "enabled_tools": list(self.tool_coverage_summary.enabled_tools),
                "unavailable_tools": list(self.tool_coverage_summary.unavailable_tools),
                "executed_tools": list(self.tool_coverage_summary.executed_tools),
                "tool_statuses": dict(self.tool_coverage_summary.tool_statuses),
            },
            "project_topology_summary": {
                "languages": list(self.project_topology.languages),
                "ecosystems": list(self.project_topology.ecosystems),
                "entrypoint_count": len(self.project_topology.entrypoints),
                "config_anchor_count": len(self.project_topology.config_anchors),
                "dependency_manifest_count": len(self.project_topology.dependency_manifests),
                "lockfile_count": len(self.project_topology.lockfiles),
            },
        }


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
    root_cause_class: str | None = None
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
class EvidenceArtifact:
    kind: str
    title: str
    summary: str
    path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FixRequest:
    source_agent: AgentKind
    title: str
    description: str
    recommended_change: str
    severity: Severity
    kind: str = "code"
    confidence: float = 0.5
    affected_files: list[str] = field(default_factory=list)
    evidence: list[EvidenceArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillPolicy:
    planning_heuristics: list[str] = field(default_factory=list)
    tool_preferences: list[str] = field(default_factory=list)
    forbidden_ordering: list[str] = field(default_factory=list)
    severity_bias: dict[str, float] = field(default_factory=dict)
    rule_weights: dict[str, float] = field(default_factory=dict)
    command_preferences: list[str] = field(default_factory=list)
    completion_checklist: list[str] = field(default_factory=list)
    handoff_style: str = ""
    patch_style: str = ""
    recommended_max_steps: int | None = None
    recommended_max_tool_calls: int | None = None
    recommended_max_wall_time_seconds: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    environment_preferences: list[str] = field(default_factory=list)
    upgrade_constraints: list[str] = field(default_factory=list)
    noise_suppression: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillPack:
    agent_kind: AgentKind
    name: str
    version: str
    description: str
    status: str
    source: SkillSource
    system_prompt: str
    skill_markdown: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    policy: SkillPolicy = field(default_factory=SkillPolicy)
    profile_hash: str = ""


@dataclass(slots=True)
class SkillBinding:
    repo_root: str
    agent_kind: AgentKind
    active_version: str
    source: SkillSource
    frozen: bool = False
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SkillCandidate:
    candidate_id: str
    repo_root: str
    agent_kind: AgentKind
    based_on_version: str
    version: str
    skill_pack: SkillPack
    status: SkillCandidateStatus = SkillCandidateStatus.CANDIDATE
    created_at: datetime = field(default_factory=utc_now)
    shadow_runs: int = 0
    notes: list[str] = field(default_factory=list)
    cooldown_until: datetime | None = None


@dataclass(slots=True)
class SkillEvaluation:
    evaluation_id: str
    repo_root: str
    agent_kind: AgentKind
    run_id: str
    active_version: str
    candidate_version: str | None
    active_score: float
    candidate_score: float | None
    mode: SkillEvaluationMode = SkillEvaluationMode.HEURISTIC
    promoted: bool = False
    reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentReflection:
    reflection_id: str
    repo_root: str
    run_id: str
    task_id: str
    session_id: str
    agent_kind: AgentKind
    skill_version: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    upgrade_hints: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentMessage:
    role: str
    content: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ToolPermissionSet:
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    allow_write: bool = False

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    requires_write: bool = False


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None


@dataclass(slots=True)
class ToolCall:
    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    active_skill_version: str | None = None
    candidate_skill_version: str | None = None
    skill_profile_hash: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentStep:
    step_index: int
    decision_summary: str
    action_type: AgentActionType
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    final_response: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentSession:
    session_id: str
    run_id: str
    task_id: str
    agent_kind: AgentKind
    task_type: TaskType
    working_repo_root: str
    objective: str
    targets: list[str]
    payload: dict[str, Any]
    messages: list[AgentMessage] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    active_skill_version: str | None = None
    candidate_skill_version: str | None = None
    active_skill: SkillPack | None = None
    candidate_skill: SkillCandidate | None = None
    skill_profile_hash: str | None = None
    completion_reason: CompletionReason | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


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
    metadata: dict[str, Any] = field(default_factory=dict)
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
    execution_environment: ExecutionEnvironment
    active_skills: dict[str, SkillPack]
    candidate_skills: dict[str, SkillCandidate]
    shadow_evaluation_summary: dict[str, Any]
    skill_upgrade_events: list[dict[str, Any]]
    event_path: str
    pr_number_override: int
    snapshot: RepoSnapshot
    change_set: ChangeSet
    startup_topology: StartupTopology
    project_topology: ProjectTopology
    language_profile: LanguageProfile
    static_context: StaticContextBundle
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
    execution_environment: ExecutionEnvironment | None = None
    active_skill: SkillPack | None = None
    candidate_skill: SkillCandidate | None = None
    startup_topology: StartupTopology | None = None
    project_topology: ProjectTopology | None = None
    language_profile: LanguageProfile | None = None
    static_context: StaticContextBundle | None = None
