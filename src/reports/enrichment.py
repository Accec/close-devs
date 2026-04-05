from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from core.models import AgentResult, Finding, Severity, WorkflowReport
from tools.file_store import FileStore
from tools.language_support import detect_snippet_language


class SummaryFindingAggregator:
    def key_findings(self, report: WorkflowReport) -> list[dict[str, Any]]:
        sources = [
            result
            for result in report.validation_results.values()
            if result is not None and result.findings
        ]
        if not sources:
            sources = [
                result
                for result in (report.static_result, report.dynamic_result)
                if result is not None
            ]

        blocker_fingerprints = {
            str(item.get("fingerprint", ""))
            for item in report.metadata.get("root_blockers", [])
            if isinstance(item, dict) and item.get("fingerprint")
        }
        findings: list[dict[str, Any]] = []
        seen: set[str] = set()
        for result in sources:
            for finding in result.findings:
                if bool(finding.evidence.get("advisory", False)):
                    continue
                if not self._is_key_finding(finding):
                    continue
                key = finding.fingerprint or self._fallback_key(finding)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "fingerprint": finding.fingerprint,
                        "source_agent": finding.source_agent.value,
                        "severity": finding.severity.value,
                        "rule_id": finding.rule_id,
                        "message": finding.message,
                        "path": finding.path,
                        "line": finding.line,
                        "root_cause_class": finding.root_cause_class,
                        "language": finding.evidence.get("language"),
                        "ecosystem": finding.evidence.get("ecosystem"),
                        "blocker": finding.fingerprint in blocker_fingerprints,
                        "snippet": str(finding.evidence.get("snippet", "")).strip() or None,
                        "snippet_start_line": finding.evidence.get("snippet_start_line"),
                        "snippet_end_line": finding.evidence.get("snippet_end_line"),
                        "highlight_line": finding.evidence.get("highlight_line"),
                        "snippet_language": finding.evidence.get("snippet_language"),
                    }
                )
        findings.sort(key=self._sort_key)
        return findings[:10]

    def dependency_vulnerabilities(self, report: WorkflowReport) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        vulnerabilities: list[dict[str, Any]] = []
        for item in report.metadata.get("dependency_vulnerabilities", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("package_name", "")),
                str(item.get("installed_version", "")),
                str(item.get("vulnerability_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            vulnerabilities.append(dict(item))
        vulnerabilities.sort(
            key=lambda item: (
                -self._severity_rank(str(item.get("severity", "medium"))),
                str(item.get("package_name", "")),
                str(item.get("vulnerability_id", "")),
            )
        )
        return vulnerabilities[:20]

    def _is_key_finding(self, finding: Finding) -> bool:
        if finding.rule_id == "startup-topology-anchor-missing":
            return False
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}:
            return True
        if finding.severity == Severity.MEDIUM and (
            finding.category in {"security", "correctness", "architecture", "runtime", "typing"}
            or finding.root_cause_class in {"config", "startup", "runtime", "application"}
        ):
            return True
        return False

    def _fallback_key(self, finding: Finding) -> str:
        return "|".join(
            [
                finding.rule_id,
                finding.path or "",
                str(finding.line or 0),
                finding.message,
            ]
        )

    def _sort_key(self, item: dict[str, Any]) -> tuple[int, int, str, int]:
        return (
            0 if item.get("blocker") else 1,
            -self._severity_rank(str(item.get("severity", "low"))),
            str(item.get("path", "") or ""),
            int(item.get("line") or 0),
        )

    def _severity_rank(self, value: str) -> int:
        return {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.get(value, 0)


async def enrich_report_snippets(
    report: WorkflowReport,
    *,
    analysis_root: Path,
    maintenance_root: Path | None,
    validation_root: Path | None,
    file_store: FileStore,
) -> None:
    cache: dict[str, list[str]] = {}

    async def enrich_result(result: AgentResult | None, root: Path | None) -> None:
        if result is None or root is None:
            return
        for finding in result.findings:
            await _enrich_finding(finding, root, cache, file_store)

    await enrich_result(report.static_result, analysis_root)
    await enrich_result(report.dynamic_result, analysis_root)
    await enrich_result(report.maintenance_result, maintenance_root)
    for result in report.validation_results.values():
        await enrich_result(result, validation_root)


async def _enrich_finding(
    finding: Finding,
    root: Path,
    cache: dict[str, list[str]],
    file_store: FileStore,
) -> None:
    await _hydrate_finding_location(finding, root, cache, file_store)
    if not finding.path or finding.line is None or finding.line < 1:
        return
    if finding.evidence.get("snippet"):
        return
    path = root / finding.path
    if not path.is_file():
        return
    cache_key = str(path)
    if cache_key not in cache:
        try:
            text = await file_store.read_text(path)
        except (OSError, UnicodeDecodeError):
            return
        cache[cache_key] = text.splitlines()
    lines = cache.get(cache_key, [])
    if not lines or finding.line > len(lines):
        return
    start_line = max(1, finding.line - 4)
    end_line = min(len(lines), finding.line + 4)
    snippet_lines = [_truncate_line(line) for line in lines[start_line - 1 : end_line]]
    finding.evidence.update(
        {
            "snippet": "\n".join(snippet_lines),
            "snippet_start_line": start_line,
            "snippet_end_line": end_line,
            "highlight_line": finding.line,
            "snippet_language": _snippet_language(path),
            "language": finding.evidence.get("language") or _snippet_language(path),
        }
    )


def _truncate_line(line: str, *, max_length: int = 240) -> str:
    if len(line) <= max_length:
        return line
    return line[: max_length - 3] + "..."


def _snippet_language(path: Path) -> str:
    return detect_snippet_language(path)


_PATH_PATTERN = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?)"
)
_CALL_SYMBOL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)")
_BACKTICK_IDENTIFIER_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


async def _hydrate_finding_location(
    finding: Finding,
    root: Path,
    cache: dict[str, list[str]],
    file_store: FileStore,
) -> None:
    if finding.path and finding.line is not None and finding.line > 0:
        return
    candidate_paths = _candidate_paths(finding, root)
    if not candidate_paths:
        return
    candidate_symbols = _candidate_symbols(finding)
    for relative_path in candidate_paths:
        path = root / relative_path
        if not path.is_file():
            continue
        lines = await _read_lines(path, cache, file_store)
        if not lines:
            continue
        line = _locate_symbol_line(lines, candidate_symbols)
        if line is None:
            continue
        finding.path = relative_path
        finding.line = line
        if candidate_symbols and not finding.symbol:
            finding.symbol = candidate_symbols[0]
        finding.evidence.setdefault("location_hydrated", True)
        return


def _candidate_paths(finding: Finding, root: Path) -> list[str]:
    candidates: list[str] = []
    fragments: list[str] = []
    if finding.path:
        fragments.append(finding.path)
    fragments.append(finding.message)
    summary = finding.evidence.get("summary")
    if isinstance(summary, str):
        fragments.append(summary)
    for key in ("path", "entrypoint_path", "config_anchor_path"):
        value = finding.evidence.get(key)
        if isinstance(value, str):
            fragments.append(value)
    inspected_files = finding.evidence.get("inspected_files")
    if isinstance(inspected_files, list):
        fragments.extend(str(item) for item in inspected_files if isinstance(item, str))
    seen: set[str] = set()
    for fragment in fragments:
        for match in _PATH_PATTERN.finditer(fragment):
            normalized = _normalize_candidate_path(match.group("path"), root)
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)
        normalized_fragment = _normalize_candidate_path(fragment, root)
        if normalized_fragment and normalized_fragment not in seen:
            seen.add(normalized_fragment)
            candidates.append(normalized_fragment)
    return candidates


def _normalize_candidate_path(value: str, root: Path) -> str | None:
    text = value.strip().strip("`'\"()[]{}<>.,:;")
    if not text or "://" in text:
        return None
    if any(character.isspace() for character in text):
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            return None
    if "/" not in text and "." not in Path(text).name:
        return None
    return text


def _candidate_symbols(finding: Finding) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = value.strip().strip("`'\"")
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        symbols.append(cleaned)

    if finding.symbol:
        add(finding.symbol)
    symbol_value = finding.evidence.get("symbol")
    if isinstance(symbol_value, str):
        add(symbol_value)
    fragments = [finding.message]
    summary = finding.evidence.get("summary")
    if isinstance(summary, str):
        fragments.append(summary)
    for fragment in fragments:
        for match in _CALL_SYMBOL_PATTERN.finditer(fragment):
            add(match.group(1))
        for match in _BACKTICK_IDENTIFIER_PATTERN.finditer(fragment):
            identifier = match.group(1)
            if "." not in identifier and "/" not in identifier:
                add(identifier)
    return symbols


async def _read_lines(
    path: Path,
    cache: dict[str, list[str]],
    file_store: FileStore,
) -> list[str]:
    cache_key = str(path)
    if cache_key not in cache:
        try:
            text = await file_store.read_text(path)
        except (OSError, UnicodeDecodeError):
            return []
        cache[cache_key] = text.splitlines()
    return cache.get(cache_key, [])


def _locate_symbol_line(lines: list[str], symbols: list[str]) -> int | None:
    if not symbols:
        return None
    for symbol in symbols:
        definition_pattern = re.compile(
            rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b"
        )
        for index, line in enumerate(lines, start=1):
            if definition_pattern.search(line):
                return index
    for symbol in symbols:
        word_pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        for index, line in enumerate(lines, start=1):
            if word_pattern.search(line):
                return index
    return None
