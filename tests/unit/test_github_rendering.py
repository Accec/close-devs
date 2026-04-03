from __future__ import annotations

from datetime import datetime, timezone

from core.models import (
    AgentKind,
    AgentResult,
    ArtifactReference,
    ChangeSet,
    FilePatch,
    Finding,
    PatchProposal,
    PublishMode,
    PullRequestContext,
    RepoSnapshot,
    ReviewPayload,
    SafeFixPolicy,
    Severity,
    TaskStatus,
    TaskType,
    WorkflowReport,
    WorkflowTrigger,
)
from github.rendering import (
    build_companion_pr_payload,
    build_review_payload,
    inline_comment_marker,
    summary_comment_marker,
)


def _sample_report() -> WorkflowReport:
    snapshot = RepoSnapshot(
        repo_root="/tmp/repo",
        scanned_at=datetime.now(timezone.utc),
        revision="abc123",
        file_hashes={"module.py": "hash"},
    )
    static_result = AgentResult(
        task_id="static-1",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        status=TaskStatus.SUCCEEDED,
        summary="static",
        findings=[
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="missing-module-docstring",
                message="Missing docstring",
                category="documentation",
                path="module.py",
                line=1,
            )
        ],
    )
    dynamic_result = AgentResult(
        task_id="dynamic-1",
        agent_kind=AgentKind.DYNAMIC_DEBUG,
        task_type=TaskType.DYNAMIC_DEBUG,
        status=TaskStatus.SUCCEEDED,
        summary="dynamic",
    )
    maintenance_result = AgentResult(
        task_id="maint-1",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        status=TaskStatus.SUCCEEDED,
        summary="maintenance",
        patch=PatchProposal(
            summary="patched",
            rationale="safe fixes",
            file_patches=[
                FilePatch(
                    path="module.py",
                    old_content="def add():\n    pass\n",
                    new_content='"""Maintained by Close-Devs."""\n\ndef add():\n    pass\n',
                    diff="diff",
                )
            ],
            suggestions=["Review the remaining architectural issues manually."],
            diff_text="diff",
        ),
    )
    return WorkflowReport(
        run_id="run-1",
        workflow_name="pull_request_maintenance",
        repo_root="/tmp/repo",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        status=TaskStatus.SUCCEEDED,
        snapshot=snapshot,
        change_set=ChangeSet(changed_files=["module.py"], added_files=[], removed_files=[]),
        static_result=static_result,
        dynamic_result=dynamic_result,
        maintenance_result=maintenance_result,
        validation_results={},
    )


def _sample_pr_context() -> PullRequestContext:
    return PullRequestContext(
        repo_full_name="acme/repo",
        base_repo_full_name="acme/repo",
        head_repo_full_name="acme/repo",
        pr_number=7,
        title="Fix module",
        html_url="https://github.com/acme/repo/pull/7",
        base_branch="main",
        head_branch="feature",
        head_sha="abc123",
        changed_files=["module.py"],
        trigger=WorkflowTrigger.PULL_REQUEST,
    )


def test_review_payload_includes_markers_artifacts_and_companion_pr() -> None:
    report = _sample_report()
    context = _sample_pr_context()

    payload: ReviewPayload = build_review_payload(
        report=report,
        pr_context=context,
        publish_mode=PublishMode.COMPANION_PR,
        artifact_references=[
            ArtifactReference(
                name="summary_md",
                path="summary.md",
                url="https://github.com/acme/repo/actions/runs/1/artifacts/8",
            ),
            ArtifactReference(
                name="patch_diff",
                path="patch.diff",
                url="https://github.com/acme/repo/actions/runs/1/artifacts/8",
            ),
        ],
        companion_pr_url="https://github.com/acme/repo/pull/9",
        publish_reasons=["validation-passed"],
        inline_comment_limit=5,
    )

    assert summary_comment_marker(context) in payload.body
    assert "Companion PR" in payload.body
    assert "[summary.md]" in payload.body
    assert payload.publish_mode is PublishMode.COMPANION_PR
    assert payload.inline_comments
    assert inline_comment_marker(report.static_result.findings[0].fingerprint, context.head_sha) in payload.inline_comments[0]["body"]

    companion = build_companion_pr_payload(
        pr_context=context,
        branch_name="close-devs/fix/7/run1",
        report=report,
        label="close-devs",
    )
    assert companion.title == "[Close-Devs] Safe autofixes for PR #7"
    assert "module.py" in companion.body
    assert companion.labels == ["close-devs"]


def test_review_payload_skips_duplicate_inline_markers() -> None:
    report = _sample_report()
    context = _sample_pr_context()
    marker = inline_comment_marker(report.static_result.findings[0].fingerprint, context.head_sha)

    payload = build_review_payload(
        report=report,
        pr_context=context,
        publish_mode=PublishMode.COMMENT_ONLY,
        artifact_references=[],
        existing_inline_markers={marker},
        inline_comment_limit=5,
    )

    assert payload.inline_comments == []


def test_safe_fix_policy_allows_only_whitelisted_rules() -> None:
    policy = SafeFixPolicy()

    assert policy.allows("W291") is True
    assert policy.allows("missing-module-docstring") is True
    assert policy.allows("python-traceback") is False
