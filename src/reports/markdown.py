from __future__ import annotations

import asyncio
from pathlib import Path

from core.models import AgentResult, WorkflowReport


def _render_result(label: str, result: AgentResult | None) -> list[str]:
    if result is None:
        return [f"## {label}", "", "Not executed.", ""]

    lines = [
        f"## {label}",
        "",
        f"- Status: `{result.status.value}`",
        f"- Summary: {result.summary}",
        f"- Findings: {len(result.findings)}",
        "",
    ]
    session_summary = result.artifacts.get("session_summary", {})
    if session_summary:
        lines.extend(
            [
                f"- Agent steps: `{session_summary.get('step_count', 0)}`",
                f"- Tool calls: `{session_summary.get('tool_call_count', 0)}`",
                f"- Completion: `{session_summary.get('completion_reason', 'unknown')}`",
                f"- Budget exhausted: `{bool(session_summary.get('budget_exhausted', False))}`",
                f"- Handoffs: `{len(result.artifacts.get('handoffs', []))}`",
                "",
            ]
        )
    for finding in result.findings[:20]:
        location = f"{finding.path}:{finding.line}" if finding.path and finding.line else finding.path or "n/a"
        lines.append(
            f"- `{finding.severity.value}` `{finding.rule_id}` at `{location}`: {finding.message}"
        )
    if result.patch is not None:
        lines.extend(
            [
                "",
                f"- Patch files: {len(result.patch.touched_files)}",
                f"- Suggestions: {len(result.patch.suggestions)}",
                "",
            ]
        )
    return lines


def render_workflow_report(report: WorkflowReport) -> str:
    lines = [
        f"# Close-Devs Report `{report.run_id}`",
        "",
        f"- Workflow: `{report.workflow_name}`",
        f"- Repo: `{report.repo_root}`",
        f"- Status: `{report.status.value}`",
        f"- Started: `{report.started_at.isoformat()}`",
        f"- Finished: `{report.finished_at.isoformat()}`",
        f"- Change reason: `{report.change_set.reason}`",
        f"- Touched files: `{len(report.change_set.all_touched_files)}`",
        "",
    ]
    environment_status = report.metadata.get("environment_status")
    if environment_status is not None:
        lines.extend(
            [
                "## Execution Environment",
                "",
                f"- Status: `{environment_status}`",
                f"- Degraded: `{bool(report.metadata.get('environment_degraded', False))}`",
                f"- Runtime root: `{report.metadata.get('runtime_root', '')}`",
                f"- Venv: `{report.metadata.get('venv_root', '')}`",
                f"- Dependency sources: `{', '.join(report.metadata.get('dependency_sources', [])) or 'none-detected'}`",
                f"- Install failures: `{len(report.metadata.get('install_failures', []))}`",
                "",
            ]
        )
        for failure in report.metadata.get("install_failures", [])[:10]:
            lines.append(f"- Install error: {failure}")
        if report.metadata.get("install_failures"):
            lines.append("")
    if report.metadata.get("agent_skills"):
        lines.extend(["## Agent Skills", ""])
        active_versions = dict(report.metadata.get("active_skill_versions", {}))
        candidate_versions = dict(report.metadata.get("candidate_skill_versions", {}))
        shadow_summary = dict(report.metadata.get("shadow_evaluation_summary", {}))
        for agent_name, metadata in dict(report.metadata.get("agent_skills", {})).items():
            lines.append(
                f"- `{agent_name}` active=`{active_versions.get(agent_name, metadata.get('active_version', '-'))}` "
                f"candidate=`{candidate_versions.get(agent_name) or '-'}`"
            )
            shadow = shadow_summary.get(agent_name)
            if isinstance(shadow, dict):
                lines.append(
                    f"- `{agent_name}` shadow: active_score=`{shadow.get('active_score', '-')}` "
                    f"candidate_score=`{shadow.get('candidate_score', '-')}` "
                    f"promoted=`{shadow.get('promoted', False)}`"
                )
                reasons = shadow.get("reasons", [])
                if reasons:
                    lines.append(f"- `{agent_name}` reasons: {', '.join(str(item) for item in reasons)}")
        for event in report.metadata.get("skill_upgrade_events", []):
            if isinstance(event, dict):
                lines.append(
                    f"- Upgrade event: `{event.get('agent', 'unknown')}` `{event.get('event', 'unknown')}` "
                    f"candidate=`{event.get('candidate_version', '-')}`"
                )
        lines.append("")
    lines.extend(_render_result("Static Review", report.static_result))
    lines.extend(_render_result("Dynamic Debug", report.dynamic_result))
    lines.extend(_render_result("Maintenance", report.maintenance_result))

    if report.validation_results:
        lines.extend(["## Validation", ""])
        for name, result in report.validation_results.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.extend(_render_result(name, result))
    return "\n".join(lines).strip() + "\n"


async def write_markdown(path: Path, report: WorkflowReport) -> str:
    return await asyncio.to_thread(_write_markdown_sync, path, report)


def _write_markdown_sync(path: Path, report: WorkflowReport) -> str:
    text = render_workflow_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
