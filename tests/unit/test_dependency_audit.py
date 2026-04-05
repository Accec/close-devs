from __future__ import annotations

import json

from tools.dependency_audit import (
    parse_cargo_audit_output,
    parse_govulncheck_output,
    parse_npm_audit_output,
    parse_pip_audit_output,
    summarize_dependency_vulnerabilities,
)


def test_parse_pip_audit_output_normalizes_vulnerabilities() -> None:
    payload = {
        "dependencies": [
            {
                "name": "django",
                "version": "4.2.0",
                "vulns": [
                    {
                        "id": "PYSEC-2026-1",
                        "aliases": ["GHSA-xxxx-yyyy"],
                        "fix_versions": ["4.2.11"],
                        "summary": "SQL injection in admin search.",
                        "severity": "critical",
                    }
                ],
            }
        ]
    }

    vulnerabilities = parse_pip_audit_output(json.dumps(payload))

    assert vulnerabilities == [
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
    ]


def test_parse_pip_audit_output_defaults_unknown_severity_to_medium() -> None:
    payload = {
        "dependencies": [
            {
                "name": "requests",
                "version": "2.31.0",
                "vulns": [
                    {
                        "id": "PYSEC-2026-2",
                        "description": "Redirect issue.",
                        "severity": None,
                    }
                ],
            }
        ]
    }

    vulnerabilities = parse_pip_audit_output(json.dumps(payload))
    summary = summarize_dependency_vulnerabilities(vulnerabilities)

    assert vulnerabilities[0]["severity"] == "medium"
    assert vulnerabilities[0]["blocker"] is False
    assert summary["blockers"] == 0
    assert summary["severity_counts"] == {"medium": 1}


def test_parse_npm_audit_output_normalizes_node_vulnerabilities() -> None:
    payload = {
        "vulnerabilities": {
            "lodash": {
                "name": "lodash",
                "severity": "high",
                "range": "4.17.0",
                "via": [
                    {
                        "source": "GHSA-35jh-r3h4-6jhm",
                        "name": "lodash",
                        "title": "Prototype Pollution",
                        "severity": "high",
                        "url": "https://github.com/advisories/GHSA-35jh-r3h4-6jhm",
                    }
                ],
                "fixAvailable": {"name": "lodash", "version": "4.17.21"},
            }
        }
    }

    vulnerabilities = parse_npm_audit_output(json.dumps(payload))

    assert vulnerabilities == [
        {
            "ecosystem": "node",
            "package_name": "lodash",
            "installed_version": "4.17.0",
            "vulnerability_id": "GHSA-35jh-r3h4-6jhm",
            "aliases": ["https://github.com/advisories/GHSA-35jh-r3h4-6jhm"],
            "fix_versions": ["4.17.21"],
            "summary": "Prototype Pollution",
            "severity": "high",
            "blocker": True,
        }
    ]


def test_parse_cargo_audit_output_normalizes_rust_vulnerabilities() -> None:
    payload = {
        "vulnerabilities": {
            "list": [
                {
                    "advisory": {
                        "id": "RUSTSEC-2020-0071",
                        "aliases": ["CVE-2020-26235"],
                        "title": "Potential segfault",
                        "description": "Unsound API usage.",
                        "severity": "critical",
                    },
                    "package": {"name": "time", "version": "0.1.44"},
                    "versions": {"patched": ["0.1.45"]},
                }
            ]
        }
    }

    vulnerabilities = parse_cargo_audit_output(json.dumps(payload))

    assert vulnerabilities[0]["ecosystem"] == "rust"
    assert vulnerabilities[0]["package_name"] == "time"
    assert vulnerabilities[0]["vulnerability_id"] == "RUSTSEC-2020-0071"
    assert vulnerabilities[0]["fix_versions"] == ["0.1.45"]
    assert vulnerabilities[0]["severity"] == "critical"


def test_parse_govulncheck_output_defaults_unknown_severity_to_medium() -> None:
    payload = "\n".join(
        [
            json.dumps(
                {
                    "osv": {
                        "id": "GO-2024-1234",
                        "summary": "Unsafe parsing bug",
                        "aliases": ["CVE-2024-0001"],
                        "affected": [
                            {
                                "ranges": [
                                    {"events": [{"introduced": "0"}, {"fixed": "v1.2.3"}]}
                                ]
                            }
                        ],
                    }
                }
            ),
            json.dumps(
                {
                    "finding": {
                        "osv": "GO-2024-1234",
                        "trace": [
                            {"module": {"path": "github.com/example/lib", "version": "v1.0.0"}}
                        ],
                    }
                }
            ),
        ]
    )

    vulnerabilities = parse_govulncheck_output(payload)

    assert vulnerabilities == [
        {
            "ecosystem": "go",
            "package_name": "github.com/example/lib",
            "installed_version": "v1.0.0",
            "vulnerability_id": "GO-2024-1234",
            "aliases": ["CVE-2024-0001"],
            "fix_versions": ["v1.2.3"],
            "summary": "Unsafe parsing bug",
            "severity": "medium",
            "blocker": False,
        }
    ]
