from __future__ import annotations

import json

from core.models import Severity
from tools.command_runner import CommandResult
from tools.static_tooling import StaticTooling


def test_parse_bandit_output_returns_issue_level_findings() -> None:
    payload = {
        "results": [
            {
                "filename": "./src/app.py",
                "line_number": 12,
                "issue_severity": "HIGH",
                "issue_confidence": "HIGH",
                "issue_text": "Use of assert detected.",
                "test_id": "B101",
                "test_name": "assert_used",
                "more_info": "https://bandit.readthedocs.io/",
                "code": "assert user.is_admin\n",
                "issue_cwe": {"id": 703, "link": "https://cwe.mitre.org/data/definitions/703.html"},
            }
        ]
    }
    result = CommandResult(
        command="bandit -q -r -f json .",
        returncode=1,
        stdout=json.dumps(payload),
        stderr="",
        duration_seconds=0.1,
    )

    findings = StaticTooling()._parse_external_output("bandit", result)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "B101"
    assert finding.path == "src/app.py"
    assert finding.line == 12
    assert finding.severity == Severity.HIGH
    assert finding.message == "Use of assert detected."
    assert finding.evidence["tool"] == "bandit"
    assert finding.evidence["test_name"] == "assert_used"
    assert finding.evidence["cwe"]["id"] == 703


def test_parse_bandit_output_returns_single_parse_error_for_invalid_json() -> None:
    result = CommandResult(
        command="bandit -q -r -f json .",
        returncode=1,
        stdout="{not-json",
        stderr="",
        duration_seconds=0.1,
    )

    findings = StaticTooling()._parse_external_output("bandit", result)

    assert len(findings) == 1
    assert findings[0].rule_id == "bandit-parse-error"
    assert findings[0].category == "tooling"


def test_parse_eslint_output_returns_issue_level_findings() -> None:
    payload = [
        {
            "filePath": "src/app.ts",
            "messages": [
                {
                    "ruleId": "no-console",
                    "severity": 2,
                    "message": "Unexpected console statement.",
                    "line": 7,
                    "column": 5,
                }
            ],
        }
    ]
    result = CommandResult(
        command="eslint --format json .",
        returncode=1,
        stdout=json.dumps(payload),
        stderr="",
        duration_seconds=0.1,
    )

    findings = StaticTooling()._parse_external_output("eslint", result)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "no-console"
    assert finding.path == "src/app.ts"
    assert finding.line == 7
    assert finding.category == "lint"
    assert finding.evidence["tool"] == "eslint"
    assert finding.evidence["language"] == "typescript"
