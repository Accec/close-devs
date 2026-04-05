from __future__ import annotations

from pathlib import Path

import pytest

from core.models import (
    AgentKind,
    AgentResult,
    ChangeSet,
    Finding,
    RepoSnapshot,
    Severity,
    TaskStatus,
    TaskType,
    WorkflowReport,
    utc_now,
)
from reports.enrichment import enrich_report_snippets
from tools.file_store import FileStore


@pytest.mark.asyncio
async def test_enrich_report_snippets_hydrates_location_from_message_and_symbol(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "src" / "infra" / "security" / "passwords.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            [
                "import hashlib",
                "",
                "def hash_password(password: str) -> str:",
                "    return hashlib.sha256(password.encode()).hexdigest()",
                "",
                "def verify_password(password: str) -> str:",
                "    return hashlib.md5(password.encode()).hexdigest()",
            ]
        ),
        encoding="utf-8",
    )

    finding = Finding(
        source_agent=AgentKind.STATIC_REVIEW,
        severity=Severity.HIGH,
        rule_id="agent-observation",
        message=(
            "src/infra/security/passwords.py still relies on MD5 in security-sensitive flows. "
            "verify_password() hashes incoming credentials with hashlib.md5()."
        ),
        category="security",
        root_cause_class="application",
    )
    static_result = AgentResult(
        task_id="task-static",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        status=TaskStatus.SUCCEEDED,
        summary="Static review identified a weak hashing implementation.",
        findings=[finding],
    )
    now = utc_now()
    report = WorkflowReport(
        run_id="run-enrichment",
        workflow_name="static_review",
        repo_root=str(tmp_path),
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root=str(tmp_path),
            scanned_at=now,
            revision=None,
            file_hashes={"src/infra/security/passwords.py": "hash"},
        ),
        change_set=ChangeSet(
            changed_files=["src/infra/security/passwords.py"],
            added_files=[],
            removed_files=[],
            reason="initial-scan",
        ),
        static_result=static_result,
    )

    await enrich_report_snippets(
        report,
        analysis_root=tmp_path,
        maintenance_root=None,
        validation_root=None,
        file_store=FileStore(),
    )

    assert finding.path == "src/infra/security/passwords.py"
    assert finding.line == 6
    assert finding.evidence.get("snippet_language") == "python"
    assert "def verify_password" in str(finding.evidence.get("snippet", ""))
    assert finding.evidence.get("highlight_line") == 6
