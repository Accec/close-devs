from __future__ import annotations

from collections import Counter

from core.models import (
    ArtifactReference,
    CompanionPRPayload,
    Finding,
    PublishMode,
    PullRequestContext,
    ReviewPayload,
    SafeFixPolicy,
    WorkflowReport,
)


def summary_comment_marker(pr_context: PullRequestContext) -> str:
    return f"<!-- close-devs:summary pr={pr_context.pr_number} -->"


def inline_comment_marker(fingerprint: str, head_sha: str) -> str:
    return f"<!-- close-devs:inline fingerprint={fingerprint} sha={head_sha} -->"


def _artifact_line(reference: ArtifactReference) -> str:
    target = reference.url or reference.fallback_url
    if target:
        return f"- `{reference.name}`: [{reference.path}]({target})"
    return f"- `{reference.name}`: `{reference.path}`"


def _inline_comment_body(finding: Finding, pr_context: PullRequestContext) -> str:
    location = f"{finding.path}:{finding.line}" if finding.path and finding.line else finding.path or "n/a"
    marker = inline_comment_marker(finding.fingerprint, pr_context.head_sha)
    return (
        f"Close-Devs noticed `{finding.rule_id}` at `{location}`.\n\n"
        f"{finding.message}\n\n"
        f"{marker}"
    )


def build_inline_comments(
    report: WorkflowReport,
    pr_context: PullRequestContext,
    *,
    inline_comment_limit: int,
    existing_inline_markers: set[str] | None = None,
) -> list[dict[str, object]]:
    if inline_comment_limit <= 0 or report.static_result is None:
        return []

    allowed_rule_ids = SafeFixPolicy().allowed_rule_ids
    seen_markers = existing_inline_markers or set()
    comments: list[dict[str, object]] = []
    for finding in report.static_result.findings:
        if len(comments) >= inline_comment_limit:
            break
        if finding.rule_id not in allowed_rule_ids or not finding.path or finding.line is None:
            continue

        marker = inline_comment_marker(finding.fingerprint, pr_context.head_sha)
        if marker in seen_markers:
            continue

        comments.append(
            {
                "path": finding.path,
                "line": finding.line,
                "body": _inline_comment_body(finding, pr_context),
                "fingerprint": finding.fingerprint,
                "marker": marker,
            }
        )
    return comments


def build_review_payload(
    report: WorkflowReport,
    pr_context: PullRequestContext,
    publish_mode: PublishMode,
    artifact_references: list[ArtifactReference],
    companion_pr_url: str | None = None,
    publish_reasons: list[str] | None = None,
    existing_inline_markers: set[str] | None = None,
    inline_comment_limit: int = 0,
) -> ReviewPayload:
    finding_counts = Counter(finding.severity.value for finding in report.all_findings)
    maintenance_patch = report.maintenance_result.patch if report.maintenance_result else None
    lines = [
        summary_comment_marker(pr_context),
        "",
        f"## Close-Devs Summary for PR #{pr_context.pr_number}",
        "",
        f"- Trigger: `{pr_context.trigger.value}`",
        f"- Publish mode: `{publish_mode.value}`",
        f"- Static findings: `{len(report.static_result.findings) if report.static_result else 0}`",
        f"- Dynamic findings: `{len(report.dynamic_result.findings) if report.dynamic_result else 0}`",
        f"- Safe autofix files: `{len(maintenance_patch.touched_files) if maintenance_patch else 0}`",
    ]
    if finding_counts:
        lines.append(
            "- Severity mix: "
            + ", ".join(f"`{level}`={count}" for level, count in sorted(finding_counts.items()))
        )
    if publish_reasons:
        lines.append(
            "- Publish reasons: "
            + ", ".join(f"`{reason}`" for reason in sorted(set(publish_reasons)))
        )
    if companion_pr_url:
        lines.append(f"- Companion PR: {companion_pr_url}")
    lines.extend(["", "### Artifacts", ""])
    for reference in artifact_references:
        lines.append(_artifact_line(reference))

    top_findings = report.all_findings[:10]
    if top_findings:
        lines.extend(["", "### Top Findings", ""])
        for finding in top_findings:
            location = f"{finding.path}:{finding.line}" if finding.path and finding.line else finding.path or "n/a"
            lines.append(
                f"- `{finding.severity.value}` `{finding.rule_id}` at `{location}`: {finding.message}"
            )

    suggestions = (
        report.maintenance_result.patch.suggestions[:8]
        if report.maintenance_result and report.maintenance_result.patch
        else []
    )
    if suggestions:
        lines.extend(["", "### Follow-up", ""])
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")

    inline_comments = build_inline_comments(
        report,
        pr_context,
        inline_comment_limit=inline_comment_limit,
        existing_inline_markers=existing_inline_markers,
    )

    return ReviewPayload(
        title=f"Close-Devs review for PR #{pr_context.pr_number}",
        body="\n".join(lines).strip() + "\n",
        summary=(
            f"Static={len(report.static_result.findings) if report.static_result else 0}, "
            f"Dynamic={len(report.dynamic_result.findings) if report.dynamic_result else 0}, "
            f"PatchFiles={len(maintenance_patch.touched_files) if maintenance_patch else 0}"
        ),
        publish_mode=publish_mode,
        inline_comments=inline_comments,
        artifact_references=artifact_references,
    )


def build_companion_pr_payload(
    pr_context: PullRequestContext,
    branch_name: str,
    report: WorkflowReport,
    label: str,
) -> CompanionPRPayload:
    patch = report.maintenance_result.patch if report.maintenance_result else None
    touched = patch.touched_files if patch else []
    body_lines = [
        f"This PR contains safe Close-Devs autofixes for #{pr_context.pr_number}.",
        "",
        f"- Source PR: {pr_context.html_url}",
        f"- Files patched: `{len(touched)}`",
        f"- Validation status: `{report.validation_results.get('static_validation').status.value if report.validation_results.get('static_validation') else 'n/a'}` / `{report.validation_results.get('dynamic_validation').status.value if report.validation_results.get('dynamic_validation') else 'n/a'}`",
        "",
        "## Autofix Scope",
        "",
    ]
    for path in touched[:20]:
        body_lines.append(f"- `{path}`")
    suggestions = patch.suggestions[:10] if patch else []
    if suggestions:
        body_lines.extend(["", "## Remaining Follow-up", ""])
        for suggestion in suggestions:
            body_lines.append(f"- {suggestion}")

    return CompanionPRPayload(
        head_branch=branch_name,
        base_branch=pr_context.base_branch,
        title=f"[Close-Devs] Safe autofixes for PR #{pr_context.pr_number}",
        body="\n".join(body_lines).strip() + "\n",
        labels=[label],
    )
