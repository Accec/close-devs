from __future__ import annotations

from core.models import (
    AgentKind,
    AgentResult,
    ChangeSet,
    Finding,
    PatchProposal,
    RepoSnapshot,
    Severity,
    TaskStatus,
    TaskType,
    WorkflowReport,
    utc_now,
)
from reports.markdown import render_workflow_report


def test_render_workflow_report_includes_manual_follow_up_guidance() -> None:
    now = utc_now()
    maintenance_result = AgentResult(
        task_id="task-maint",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        status=TaskStatus.SUCCEEDED,
        summary="Prepared a maintenance patch with one unresolved startup blocker.",
        patch=PatchProposal(
            summary="Patch summary",
            rationale="Patch rationale",
            file_patches=[],
            validation_targets=[],
            suggestions=[],
            metadata={"repair_scope": ["startup"]},
            applied=False,
            diff_text="",
        ),
        artifacts={
            "auto_fixed_blockers": ["startup"],
            "unresolved_handoffs": [
                {
                    "kind": "startup",
                    "reason": "uvicorn-app-dir-unsupported",
                    "message": "Uvicorn still needs manual startup configuration review.",
                    "guidance": "Adjust uvicorn app-dir or equivalent Python path bootstrap.",
                }
            ],
        },
    )
    report = WorkflowReport(
        run_id="run-1",
        workflow_name="maintenance_loop",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"pyproject.toml": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["pyproject.toml"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        maintenance_result=maintenance_result,
        metadata={
            "resolution_summary": {
                "resolved": 0,
                "unresolved": 1,
                "regressed": 0,
                "repo_healthy": False,
                "root_blocker_classes": ["startup"],
                "root_blockers_by_class": {"startup": 1},
                "workflow_succeeded_but_repo_unhealthy": True,
                "explanation": "Workflow execution completed, but the repository is still unhealthy.",
            },
            "manual_follow_ups": [
                {
                    "kind": "startup",
                    "reason": "uvicorn-app-dir-unsupported",
                    "message": "Uvicorn still needs manual startup configuration review.",
                    "guidance": "Adjust uvicorn app-dir or equivalent Python path bootstrap.",
                }
            ],
            "auto_fixed_blockers": ["startup"],
            "dependency_audit_status": "unavailable",
            "dependency_vulnerability_summary": {"total": 0, "blockers": 0, "severity_counts": {}, "packages": []},
            "dependency_vulnerability_count": 0,
            "dependency_vulnerability_blocker_count": 0,
        },
    )

    rendered = render_workflow_report(report)

    assert "## Manual Follow-up" in rendered
    assert "uvicorn-app-dir-unsupported" in rendered
    assert "Adjust uvicorn app-dir or equivalent Python path bootstrap." in rendered
    assert "## Agent Execution" in rendered
    assert "## Root Blockers" not in rendered


def test_render_workflow_report_includes_key_findings_with_snippet() -> None:
    now = utc_now()
    finding = Finding(
        source_agent=AgentKind.STATIC_REVIEW,
        severity=Severity.HIGH,
        rule_id="swallowed-exception",
        message="Exception handler swallows failures and returns silently.",
        category="security",
        root_cause_class="application",
        path="src/app.py",
        line=12,
        evidence={
            "snippet": "try:\n    risky()\nexcept Exception:\n    return None",
            "snippet_start_line": 9,
            "snippet_end_line": 12,
            "highlight_line": 12,
            "snippet_language": "python",
        },
    )
    static_result = AgentResult(
        task_id="task-static",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        status=TaskStatus.SUCCEEDED,
        summary="Static review surfaced one high-signal security finding.",
        findings=[finding],
        artifacts={"session_summary": {"step_count": 3, "tool_call_count": 2, "completion_reason": "completed"}},
    )
    report = WorkflowReport(
        run_id="run-snippet",
        workflow_name="static_review",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"src/app.py": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app.py"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        static_result=static_result,
        metadata={
            "root_blockers": [
                {
                    "fingerprint": finding.fingerprint,
                    "source_agent": "static_review",
                    "severity": "high",
                    "rule_id": finding.rule_id,
                    "message": finding.message,
                    "root_cause_class": "application",
                }
            ],
            "dependency_audit_status": "disabled",
            "dependency_vulnerability_summary": {"total": 0, "blockers": 0, "severity_counts": {}, "packages": []},
            "dependency_vulnerability_count": 0,
            "dependency_vulnerability_blocker_count": 0,
        },
    )

    rendered = render_workflow_report(report)

    assert "## Key Findings" in rendered
    assert "### Finding 1" in rendered
    assert "- Severity: `high`" in rendered
    assert "- Type: `swallowed-exception`" in rendered
    assert "- Source: `static_review`" in rendered
    assert "- Location: `src/app.py:12`" in rendered
    assert "- Snippet: `src/app.py` lines `9`-`12` (highlight `12`)" in rendered
    assert "```python" in rendered
    assert "except Exception:" in rendered


def test_render_workflow_report_includes_dependency_vulnerabilities() -> None:
    now = utc_now()
    report = WorkflowReport(
        run_id="run-deps",
        workflow_name="maintenance_loop",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"requirements.txt": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["requirements.txt"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        metadata={
            "dependency_audit_status": "executed",
            "dependency_vulnerability_summary": {
                "total": 1,
                "blockers": 1,
                "severity_counts": {"critical": 1},
                "packages": ["django"],
            },
            "dependency_vulnerability_count": 1,
            "dependency_vulnerability_blocker_count": 1,
            "dependency_vulnerabilities": [
                {
                    "ecosystem": "python",
                    "package_name": "django",
                    "installed_version": "4.2.0",
                    "vulnerability_id": "PYSEC-2026-1",
                    "aliases": ["GHSA-xxxx-yyyy"],
                    "fix_versions": ["4.2.11"],
                    "summary": "SQL injection in admin search.",
                    "severity": "critical",
                    "blocker": True,
                }
            ],
        },
    )

    rendered = render_workflow_report(report)

    assert "## Dependency Vulnerabilities" in rendered
    assert "- Audit status: `executed`" in rendered
    assert "### Dependency Vulnerability 1" in rendered
    assert "- Severity: `critical`" in rendered
    assert "- Advisory: `PYSEC-2026-1`" in rendered
    assert "- Ecosystem: `python`" in rendered
    assert "- Package: `django`" in rendered
    assert "- Installed version: `4.2.0`" in rendered
    assert "Fixed versions: `4.2.11`" in rendered


def test_render_workflow_report_includes_repo_profile_section() -> None:
    now = utc_now()
    report = WorkflowReport(
        run_id="run-profile",
        workflow_name="maintenance_loop",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"package.json": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["package.json"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        metadata={
            "language_profile_summary": {
                "primary_language": "typescript",
                "languages": ["typescript", "javascript"],
                "primary_ecosystem": "node",
                "ecosystems": ["node"],
                "enabled_adapters": ["typescript", "javascript"],
                "generic_review": False,
            },
            "tool_coverage_summary": {
                "enabled_tools": ["eslint", "tsc"],
                "unavailable_tools": ["eslint"],
                "executed_tools": ["tsc"],
                "tool_statuses": {"eslint": "missing", "tsc": "executed"},
            },
            "dependency_audit_ecosystem": "node",
        },
    )

    rendered = render_workflow_report(report)

    assert "## Repo Profile" in rendered
    assert "Primary language: `typescript`" in rendered
    assert "Enabled adapters: `typescript, javascript`" in rendered
    assert "Dependency audit ecosystem: `node`" in rendered


def test_render_workflow_report_includes_startup_topology_section() -> None:
    now = utc_now()
    report = WorkflowReport(
        run_id="run-topology",
        workflow_name="maintenance_loop",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"manage.py": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["manage.py"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        metadata={
            "startup_topology_summary": {
                "src_layout": True,
                "entrypoint_count": 1,
                "config_anchor_count": 1,
                "contexts": ["django_manage"],
                "entrypoints": [
                    {
                        "context": "django_manage",
                        "path": "manage.py",
                        "config_anchor_path": "manage.py",
                        "repair_hint": "managepy-sys-path-src",
                    }
                ],
                "config_anchors": [
                    {
                        "context": "django_manage",
                        "path": "manage.py",
                        "anchor_type": "python_src_bootstrap",
                        "status": "missing",
                        "repair_hint": "managepy-sys-path-src",
                    }
                ],
                "repair_hints": ["managepy-sys-path-src"],
                "confirmed_count": 0,
                "advisory_count": 1,
            },
            "confirmed_startup_blockers": [],
            "advisory_startup_handoffs": [
                {
                    "kind": "startup",
                    "reason": "startup-topology-advisory",
                    "message": "Static review found a missing startup anchor for manage.py.",
                    "guidance": "Add deterministic src bootstrap to manage.py.",
                    "startup_context": "django_manage",
                    "entrypoint_path": "manage.py",
                    "config_anchor_path": "manage.py",
                    "repair_hint": "managepy-sys-path-src",
                }
            ],
        },
    )

    rendered = render_workflow_report(report)

    assert "## Startup Topology" in rendered
    assert "entrypoint=`manage.py`" in rendered
    assert "`advisory` context=`django_manage`" in rendered
    assert "Advisory `django_manage` entrypoint=`manage.py` anchor=`manage.py`" in rendered


def test_render_workflow_report_includes_static_context_section() -> None:
    now = utc_now()
    report = WorkflowReport(
        run_id="run-static-context",
        workflow_name="maintenance_loop",
        repo_root="/tmp/repo",
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root="/tmp/repo",
            scanned_at=now,
            revision=None,
            file_hashes={"manage.py": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["manage.py"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        metadata={
            "static_context_summary": {
                "enabled": True,
                "startup_context_count": 3,
                "prioritized_target_count": 27,
                "top_target_count": 27,
                "high_signal_target_count": 8,
                "related_file_count": 5,
                "baseline_total_findings": 19,
                "baseline_severity_counts": {"high": 1, "medium": 2, "low": 16},
                "baseline_noisy_rule_counts": {"missing-module-docstring": 14},
            }
        },
    )

    rendered = render_workflow_report(report)

    assert "## Static Context" in rendered
    assert "- Enabled: `True`" in rendered
    assert "- Prioritized targets: `27`" in rendered
    assert "- Baseline severity counts: `high=1, medium=2, low=16`" in rendered
