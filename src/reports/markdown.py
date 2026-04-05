from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.models import AgentResult, WorkflowReport
from reports.enrichment import SummaryFindingAggregator


def _render_agent_execution_result(label: str, result: AgentResult | None) -> list[str]:
    if result is None:
        return [f"### {label}", "", "Not executed.", ""]

    lines = [
        f"### {label}",
        "",
        f"- Status: `{result.status.value}`",
        f"- Summary: {result.summary}",
        f"- Findings: `{len(result.findings)}`",
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
            ]
        )
    if result.patch is not None:
        lines.extend(
            [
                f"- Patch files: `{len(result.patch.touched_files)}`",
                f"- Suggestions: `{len(result.patch.suggestions)}`",
                f"- Auto-fixed blockers: `{', '.join(result.artifacts.get('auto_fixed_blockers', [])) or '-'}`",
                f"- Unresolved handoffs: `{len(result.artifacts.get('unresolved_handoffs', []))}`",
            ]
        )
    lines.append("")
    return lines


def _render_key_findings(report: WorkflowReport) -> list[str]:
    findings = SummaryFindingAggregator().key_findings(report)
    lines = ["## Key Findings", ""]
    if not findings:
        lines.extend(["No high-signal code findings highlighted in this run.", ""])
        return lines

    for index, item in enumerate(findings, start=1):
        lines.append(f"### Finding {index}")
        lines.append("")
        lines.append(f"- Severity: `{item.get('severity', 'unknown')}`")
        lines.append(f"- Type: `{item.get('rule_id', 'unknown')}`")
        lines.append(f"- Source: `{item.get('source_agent', 'unknown')}`")
        lines.append(f"- Location: `{_location(item)}`")
        if item.get("blocker"):
            lines.append("- Blocker: `true`")
        if item.get("language") or item.get("ecosystem"):
            lines.append(
                f"- Coverage: language=`{item.get('language', '-') or '-'}` ecosystem=`{item.get('ecosystem', '-') or '-'}`"
            )
        if item.get("root_cause_class"):
            lines.append(f"- Root cause: `{item['root_cause_class']}`")
        lines.append(f"- Summary: {item.get('message', '')}")
        snippet = str(item.get("snippet", "") or "").strip()
        if snippet:
            lines.append(
                f"- Snippet: `{item.get('path', '-')}` "
                f"lines `{item.get('snippet_start_line', '-')}`-`{item.get('snippet_end_line', '-')}` "
                f"(highlight `{item.get('highlight_line', '-')}`)"
            )
            lines.append("")
            lines.append(f"```{item.get('snippet_language', 'text')}")
            lines.append(snippet)
            lines.append("```")
        else:
            lines.append("- Snippet: unavailable")
        lines.append("")
    return lines


def _render_dependency_vulnerabilities(report: WorkflowReport) -> list[str]:
    metadata = report.metadata
    status = str(metadata.get("dependency_audit_status", "") or "not-run")
    summary_value = metadata.get("dependency_vulnerability_summary", {})
    summary = dict(summary_value) if isinstance(summary_value, dict) else {}
    vulnerabilities = SummaryFindingAggregator().dependency_vulnerabilities(report)
    lines = [
        "## Dependency Vulnerabilities",
        "",
        f"- Audit status: `{status}`",
        f"- Total: `{metadata.get('dependency_vulnerability_count', 0)}`",
        f"- Blockers: `{metadata.get('dependency_vulnerability_blocker_count', 0)}`",
        f"- Severity counts: `{', '.join(f'{key}={value}' for key, value in summary.get('severity_counts', {}).items()) or '-'}`",
        "",
    ]
    error = str(metadata.get("dependency_audit_error", "")).strip()
    if error:
        lines.append(f"- Audit error: {error}")
        lines.append("")
    if not vulnerabilities:
        if status in {"executed", "disabled"}:
            lines.extend(["No dependency vulnerabilities reported.", ""])
        else:
            lines.extend(["Dependency vulnerability data is unavailable for this run.", ""])
        return lines

    for index, item in enumerate(vulnerabilities, start=1):
        aliases = [str(alias) for alias in item.get("aliases", []) if str(alias)]
        fix_versions = [str(version) for version in item.get("fix_versions", []) if str(version)]
        lines.append(f"### Dependency Vulnerability {index}")
        lines.append("")
        lines.append(f"- Severity: `{item.get('severity', 'medium')}`")
        lines.append(f"- Advisory: `{item.get('vulnerability_id', 'unknown')}`")
        lines.append(f"- Ecosystem: `{item.get('ecosystem', '-')}`")
        lines.append(f"- Package: `{item.get('package_name', '-')}`")
        lines.append(f"- Installed version: `{item.get('installed_version', '-')}`")
        if item.get("blocker"):
            lines.append("- Blocker: `true`")
        lines.append(f"- Summary: {item.get('summary', '')}")
        if aliases:
            lines.append(f"- Aliases: `{', '.join(aliases)}`")
        if fix_versions:
            lines.append(f"- Fixed versions: `{', '.join(fix_versions)}`")
        lines.append("")
    return lines


def _render_manual_follow_up(report: WorkflowReport) -> list[str]:
    items = report.metadata.get("manual_follow_ups")
    if not items:
        return []
    lines = ["## Manual Follow-up", ""]
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- `{item.get('kind', 'unknown')}` `{item.get('reason', 'unknown')}`: {item.get('message', '')}"
        )
        guidance = str(item.get("guidance", "")).strip()
        if guidance:
            lines.append(f"- Guidance: {guidance}")
        if item.get("entrypoint_path") or item.get("config_anchor_path"):
            lines.append(
                f"- Startup mapping: context=`{item.get('startup_context', '-')}` "
                f"entrypoint=`{item.get('entrypoint_path', '-')}` "
                f"anchor=`{item.get('config_anchor_path', '-')}`"
            )
    lines.append("")
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
                f"- Installer summary: `{', '.join(f'{key}={value}' for key, value in dict(report.metadata.get('environment_installer_summary', {})).items()) or '-'}`",
                f"- Install failures: `{len(report.metadata.get('install_failures', []))}`",
                "",
            ]
        )
        for failure in report.metadata.get("install_failures", [])[:10]:
            lines.append(f"- Install error: {failure}")
        if report.metadata.get("install_failures"):
            lines.append("")
    if report.metadata.get("actual_llm_provider") is not None:
        lines.extend(
            [
                "## Runtime Metadata",
                "",
                f"- Provider: `{report.metadata.get('actual_llm_provider', 'unknown')}`",
                f"- Model: `{report.metadata.get('actual_llm_model', 'unknown')}`",
                f"- LLM failure reason: `{report.metadata.get('llm_failure_reason', '') or '-'}`",
                "",
            ]
        )
        for agent_name, provider in dict(report.metadata.get("actual_llm_providers", {})).items():
            lines.append(
                f"- `{agent_name}` provider=`{provider}` model=`{dict(report.metadata.get('actual_llm_models', {})).get(agent_name, '-')}`"
            )
        lines.append("")
    language_profile_summary = report.metadata.get("language_profile_summary")
    if language_profile_summary:
        profile = dict(language_profile_summary)
        tool_coverage = dict(report.metadata.get("tool_coverage_summary", {}))
        lines.extend(
            [
                "## Repo Profile",
                "",
                f"- Primary language: `{profile.get('primary_language', 'unknown')}`",
                f"- Languages: `{', '.join(profile.get('languages', [])) or '-'}`",
                f"- Primary ecosystem: `{profile.get('primary_ecosystem', 'generic')}`",
                f"- Ecosystems: `{', '.join(profile.get('ecosystems', [])) or '-'}`",
                f"- Enabled adapters: `{', '.join(profile.get('enabled_adapters', [])) or '-'}`",
                f"- Generic review mode: `{bool(profile.get('generic_review', False))}`",
                f"- Enabled tools: `{', '.join(tool_coverage.get('enabled_tools', [])) or '-'}`",
                f"- Unavailable tools: `{', '.join(tool_coverage.get('unavailable_tools', [])) or '-'}`",
                f"- Executed tools: `{', '.join(tool_coverage.get('executed_tools', [])) or '-'}`",
                f"- Dependency audit ecosystem: `{report.metadata.get('dependency_audit_ecosystem', '-')}`",
                "",
            ]
        )
    static_context_summary = report.metadata.get("static_context_summary")
    if static_context_summary:
        summary = dict(static_context_summary)
        lines.extend(
            [
                "## Static Context",
                "",
                f"- Enabled: `{bool(summary.get('enabled', False))}`",
                f"- Primary language: `{summary.get('primary_language', 'unknown')}`",
                f"- Languages: `{', '.join(summary.get('languages', [])) or '-'}`",
                f"- Ecosystems: `{', '.join(summary.get('ecosystems', [])) or '-'}`",
                f"- Generic review mode: `{bool(summary.get('generic_review', False))}`",
                f"- Startup contexts: `{summary.get('startup_context_count', 0)}`",
                f"- Prioritized targets: `{summary.get('prioritized_target_count', 0)}`",
                f"- Top targets injected: `{summary.get('top_target_count', 0)}`",
                f"- High-signal targets: `{summary.get('high_signal_target_count', 0)}`",
                f"- Related files: `{summary.get('related_file_count', 0)}`",
                f"- Baseline findings: `{summary.get('baseline_total_findings', 0)}`",
                f"- Baseline severity counts: `{', '.join(f'{key}={value}' for key, value in dict(summary.get('baseline_severity_counts', {})).items()) or '-'}`",
                f"- Baseline noisy rules: `{', '.join(f'{key}={value}' for key, value in dict(summary.get('baseline_noisy_rule_counts', {})).items()) or '-'}`",
                f"- Tool coverage: `{', '.join(f'{key}={value}' for key, value in dict(summary.get('tool_coverage_summary', {}).get('tool_statuses', {})).items()) or '-'}`",
                "",
            ]
        )
    startup_topology_summary = report.metadata.get("startup_topology_summary")
    if startup_topology_summary:
        summary = dict(startup_topology_summary)
        confirmed_startup = [
            item
            for item in report.metadata.get("confirmed_startup_blockers", [])
            if isinstance(item, dict)
        ]
        advisory_startup = [
            item
            for item in report.metadata.get("advisory_startup_handoffs", [])
            if isinstance(item, dict)
        ]
        confirmed_keys = {
            (
                str(item.get("startup_context", "") or ""),
                str(item.get("entrypoint_path", "") or ""),
                str(item.get("config_anchor_path", "") or ""),
            )
            for item in confirmed_startup
        }
        advisory_keys = {
            (
                str(item.get("startup_context", "") or ""),
                str(item.get("entrypoint_path", "") or ""),
                str(item.get("config_anchor_path", "") or ""),
            )
            for item in advisory_startup
        }
        lines.extend(
            [
                "## Startup Topology",
                "",
                f"- Src layout: `{bool(summary.get('src_layout', False))}`",
                f"- Entrypoints: `{summary.get('entrypoint_count', 0)}`",
                f"- Config anchors: `{summary.get('config_anchor_count', 0)}`",
                f"- Confirmed startup blockers: `{len(confirmed_startup)}`",
                f"- Advisory startup handoffs: `{len(advisory_startup)}`",
                f"- Contexts: `{', '.join(summary.get('contexts', [])) or '-'}`",
                "",
            ]
        )
        for entrypoint in summary.get("entrypoints", [])[:20]:
            if not isinstance(entrypoint, dict):
                continue
            key = (
                str(entrypoint.get("context", "") or ""),
                str(entrypoint.get("path", "") or ""),
                str(entrypoint.get("config_anchor_path", "") or ""),
            )
            status = (
                "confirmed"
                if key in confirmed_keys
                else "advisory"
                if key in advisory_keys
                else "discovered"
            )
            lines.append(
                f"- `{status}` context=`{entrypoint.get('context', '-')}` "
                f"entrypoint=`{entrypoint.get('path', '-')}` "
                f"anchor=`{entrypoint.get('config_anchor_path', '-')}` "
                f"repair=`{entrypoint.get('repair_hint', '-')}`"
            )
        if summary.get("entrypoints"):
            lines.append("")
        for item in confirmed_startup[:10]:
            lines.append(
                f"- Confirmed `{item.get('startup_context', '-')}` "
                f"entrypoint=`{item.get('entrypoint_path', '-')}` "
                f"anchor=`{item.get('config_anchor_path', '-')}`: {item.get('message', '')}"
            )
        if confirmed_startup:
            lines.append("")
        for item in advisory_startup[:10]:
            lines.append(
                f"- Advisory `{item.get('startup_context', '-')}` "
                f"entrypoint=`{item.get('entrypoint_path', '-')}` "
                f"anchor=`{item.get('config_anchor_path', '-')}`: {item.get('message', '')}"
            )
        if advisory_startup:
            lines.append("")
    if report.metadata.get("resolution_summary"):
        resolution = dict(report.metadata.get("resolution_summary", {}))
        lines.extend(
            [
                "## Resolution Summary",
                "",
                f"- Resolved: `{resolution.get('resolved', 0)}`",
                f"- Unresolved: `{resolution.get('unresolved', 0)}`",
                f"- Regressed: `{resolution.get('regressed', 0)}`",
                f"- Repo healthy: `{resolution.get('repo_healthy', False)}`",
                f"- Root blocker classes: `{', '.join(resolution.get('root_blocker_classes', [])) or '-'}`",
                f"- Root blockers by class: `{', '.join(f'{key}={value}' for key, value in dict(resolution.get('root_blockers_by_class', {})).items()) or '-'}`",
                f"- Auto-fixed blockers: `{', '.join(report.metadata.get('auto_fixed_blockers', [])) or '-'}`",
                "",
            ]
        )
        if resolution.get("explanation"):
            lines.append(f"- Explanation: {resolution['explanation']}")
            lines.append("")

    lines.extend(_render_key_findings(report))
    lines.extend(_render_dependency_vulnerabilities(report))
    lines.extend(_render_manual_follow_up(report))

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
                    f"mode=`{shadow.get('mode', '-')}` promoted=`{shadow.get('promoted', False)}`"
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

    lines.extend(["## Agent Execution", ""])
    lines.extend(_render_agent_execution_result("Static Review", report.static_result))
    lines.extend(_render_agent_execution_result("Dynamic Debug", report.dynamic_result))
    lines.extend(_render_agent_execution_result("Maintenance", report.maintenance_result))

    if report.validation_results:
        lines.extend(["## Validation", ""])
        for name, result in report.validation_results.items():
            lines.extend(_render_agent_execution_result(name, result))
    return "\n".join(lines).strip() + "\n"


def _location(item: dict[str, Any]) -> str:
    path = str(item.get("path", "") or "n/a")
    line = item.get("line")
    return f"{path}:{line}" if path and line else path


async def write_markdown(path: Path, report: WorkflowReport) -> str:
    return await asyncio.to_thread(_write_markdown_sync, path, report)


def _write_markdown_sync(path: Path, report: WorkflowReport) -> str:
    text = render_workflow_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
