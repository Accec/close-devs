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
