from __future__ import annotations

import json
from typing import Any


def parse_dependency_audit_output(parser: str, text: str) -> list[dict[str, Any]]:
    if parser == "pip-audit":
        return parse_pip_audit_output(text)
    if parser == "npm":
        return parse_npm_audit_output(text)
    if parser == "cargo-audit":
        return parse_cargo_audit_output(text)
    if parser == "govulncheck":
        return parse_govulncheck_output(text)
    if parser == "dependency-check":
        return []
    raise ValueError(f"Unsupported dependency audit parser: {parser}")


def parse_pip_audit_output(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    dependencies = _dependencies(payload)
    findings: list[dict[str, Any]] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        package_name = str(dependency.get("name", "")).strip()
        installed_version = str(dependency.get("version", "")).strip()
        for vulnerability in dependency.get("vulns", []) or []:
            if not isinstance(vulnerability, dict):
                continue
            findings.append(
                _normalize_vulnerability(
                    ecosystem="python",
                    package_name=package_name,
                    installed_version=installed_version,
                    vulnerability=vulnerability,
                )
            )
    return findings


def parse_npm_audit_output(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    vulnerabilities = payload.get("vulnerabilities", {}) if isinstance(payload, dict) else {}
    findings: list[dict[str, Any]] = []
    if not isinstance(vulnerabilities, dict):
        return findings
    for package_name, item in vulnerabilities.items():
        if not isinstance(item, dict):
            continue
        installed_version = str(item.get("range", "")).strip()
        via_items = item.get("via", []) or []
        for vulnerability in via_items:
            if isinstance(vulnerability, str):
                findings.append(
                    _normalize_vulnerability(
                        ecosystem="node",
                        package_name=str(package_name),
                        installed_version=installed_version,
                        vulnerability={
                            "id": vulnerability,
                            "summary": vulnerability,
                            "severity": item.get("severity"),
                            "fix_versions": _npm_fix_versions(item.get("fixAvailable")),
                        },
                    )
                )
                continue
            if not isinstance(vulnerability, dict):
                continue
            findings.append(
                _normalize_vulnerability(
                    ecosystem="node",
                    package_name=str(vulnerability.get("name") or package_name),
                    installed_version=installed_version,
                    vulnerability={
                        "id": vulnerability.get("source") or vulnerability.get("url") or vulnerability.get("name"),
                        "aliases": [vulnerability.get("url")] if vulnerability.get("url") else [],
                        "summary": vulnerability.get("title") or vulnerability.get("overview"),
                        "details": vulnerability.get("recommendation"),
                        "severity": vulnerability.get("severity") or item.get("severity"),
                        "fix_versions": _npm_fix_versions(item.get("fixAvailable")),
                    },
                )
            )
    return findings


def parse_cargo_audit_output(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    vulnerabilities = payload.get("vulnerabilities", {}) if isinstance(payload, dict) else {}
    advisories = vulnerabilities.get("list", []) if isinstance(vulnerabilities, dict) else []
    findings: list[dict[str, Any]] = []
    for item in advisories:
        if not isinstance(item, dict):
            continue
        advisory = item.get("advisory", {}) if isinstance(item.get("advisory"), dict) else {}
        package = item.get("package", {}) if isinstance(item.get("package"), dict) else {}
        versions = item.get("versions", {}) if isinstance(item.get("versions"), dict) else {}
        findings.append(
            _normalize_vulnerability(
                ecosystem="rust",
                package_name=str(package.get("name", "")).strip(),
                installed_version=str(package.get("version", "")).strip(),
                vulnerability={
                    "id": advisory.get("id"),
                    "aliases": advisory.get("aliases", []),
                    "summary": advisory.get("title"),
                    "details": advisory.get("description"),
                    "severity": advisory.get("severity"),
                    "fix_versions": versions.get("patched", []),
                },
            )
        )
    return findings


def parse_govulncheck_output(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    osv_index: dict[str, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("osv"), dict):
            osv = payload["osv"]
            osv_id = str(osv.get("id", "")).strip()
            if osv_id:
                osv_index[osv_id] = osv
            continue
        finding = payload.get("finding")
        if not isinstance(finding, dict):
            continue
        osv_id = str(finding.get("osv", "")).strip()
        trace = finding.get("trace", []) or []
        module = ""
        version = ""
        if isinstance(trace, list):
            for frame in trace:
                if not isinstance(frame, dict):
                    continue
                module_info = frame.get("module")
                if isinstance(module_info, dict):
                    module = module or str(module_info.get("path", "")).strip()
                    version = version or str(module_info.get("version", "")).strip()
        osv_payload = osv_index.get(osv_id, {})
        findings.append(
            _normalize_vulnerability(
                ecosystem="go",
                package_name=module,
                installed_version=version,
                vulnerability={
                    "id": osv_id,
                    "aliases": osv_payload.get("aliases", []),
                    "summary": osv_payload.get("summary"),
                    "details": osv_payload.get("details"),
                    "severity": _govulncheck_severity(osv_payload),
                    "fix_versions": _govulncheck_fix_versions(osv_payload),
                },
            )
        )
    return findings


def summarize_dependency_vulnerabilities(
    vulnerabilities: list[dict[str, Any]],
) -> dict[str, Any]:
    severity_counts: dict[str, int] = {}
    blocker_count = 0
    ecosystems: dict[str, int] = {}
    for item in vulnerabilities:
        severity = str(item.get("severity", "medium"))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if bool(item.get("blocker", False)):
            blocker_count += 1
        ecosystem = str(item.get("ecosystem", "")).strip()
        if ecosystem:
            ecosystems[ecosystem] = ecosystems.get(ecosystem, 0) + 1
    return {
        "total": len(vulnerabilities),
        "blockers": blocker_count,
        "severity_counts": severity_counts,
        "packages": sorted(
            {
                str(item.get("package_name", ""))
                for item in vulnerabilities
                if item.get("package_name")
            }
        ),
        "ecosystems": ecosystems,
    }


def _dependencies(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        dependencies = payload.get("dependencies")
        if isinstance(dependencies, list):
            return [item for item in dependencies if isinstance(item, dict)]
    return []


def _normalize_vulnerability(
    *,
    ecosystem: str,
    package_name: str,
    installed_version: str,
    vulnerability: dict[str, Any],
) -> dict[str, Any]:
    aliases = [
        str(item).strip()
        for item in vulnerability.get("aliases", []) or []
        if str(item).strip()
    ]
    vulnerability_id = str(
        vulnerability.get("id")
        or (aliases[0] if aliases else "unknown-vulnerability")
    ).strip()
    cvss = vulnerability.get("cvss")
    cvss_severity = None
    if isinstance(cvss, dict):
        cvss_severity = cvss.get("severity")
    severity = _normalize_severity(
        vulnerability.get("severity")
        or vulnerability.get("cvss_v3_severity")
        or cvss_severity
    )
    fix_versions = [
        str(item).strip()
        for item in vulnerability.get("fix_versions", []) or []
        if str(item).strip()
    ]
    summary = str(
        vulnerability.get("description")
        or vulnerability.get("summary")
        or vulnerability.get("details")
        or vulnerability_id
    ).strip()
    return {
        "ecosystem": ecosystem,
        "package_name": package_name,
        "installed_version": installed_version,
        "vulnerability_id": vulnerability_id,
        "aliases": aliases,
        "fix_versions": fix_versions,
        "summary": summary,
        "severity": severity,
        "blocker": severity in {"high", "critical"},
    }


def _npm_fix_versions(value: object) -> list[str]:
    if isinstance(value, dict):
        version = str(value.get("version", "")).strip()
        return [version] if version else []
    if isinstance(value, bool):
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _govulncheck_fix_versions(payload: dict[str, Any]) -> list[str]:
    affected = payload.get("affected", []) or []
    fixed_versions: list[str] = []
    for item in affected:
        if not isinstance(item, dict):
            continue
        ranges = item.get("ranges", []) or []
        for version_range in ranges:
            if not isinstance(version_range, dict):
                continue
            for event in version_range.get("events", []) or []:
                if not isinstance(event, dict):
                    continue
                fixed = str(event.get("fixed", "")).strip()
                if fixed:
                    fixed_versions.append(fixed)
    return fixed_versions


def _govulncheck_severity(payload: dict[str, Any]) -> str:
    database_specific = payload.get("database_specific", {})
    if isinstance(database_specific, dict):
        severity = database_specific.get("severity")
        if severity:
            return _normalize_severity(severity)
    return "medium"


def _normalize_severity(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"critical", "high", "medium", "low"}:
        return normalized
    if normalized in {"moderate"}:
        return "medium"
    return "medium"
