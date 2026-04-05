from __future__ import annotations

import asyncio
import ast
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any

from core.config import AppConfig
from core.models import (
    BaselineStaticDigest,
    ExecutionEnvironment,
    Finding,
    ImportAdjacencyDigest,
    RepoMapSummary,
    StartupTopology,
    StaticContextBundle,
    StaticTargetDigest,
    ToolCoverageSummary,
)
from tools.agent_toolkit import AgentToolkitFactory
from tools.language_support import (
    build_project_topology,
    build_static_tool_adapters,
    detect_language_profile,
)
from tools.static_tooling import StaticTooling


LOW_VALUE_RULE_IDS = frozenset(
    {
        "missing-module-docstring",
        "missing-final-newline",
        "trailing-whitespace",
        "excessive-eof-blank-lines",
        "ruff-output",
    }
)
PRIORITY_KEYWORDS = (
    "bootstrap",
    "container",
    "settings",
    "config",
    "entry",
    "manage",
    "asgi",
    "wsgi",
    "gunicorn",
    "alembic",
    "pytest",
    "celery",
    "runtime",
    "main",
    "dependency",
    "package",
    "vite",
    "next",
    "cargo",
    "go.mod",
    "pom",
    "gradle",
    "tsconfig",
)
TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".kts",
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "state",
    "reports",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    ".gradle",
    "coverage",
}
IMPORT_LINE_PATTERN = re.compile(r"^\s*import\s+([A-Za-z0-9_./:-]+)", flags=re.MULTILINE)
FROM_LINE_PATTERN = re.compile(r"^\s*from\s+([A-Za-z0-9_./:-]+)", flags=re.MULTILINE)
REQUIRE_PATTERN = re.compile(r"""require\(["']([^"']+)["']\)""")
JS_IMPORT_PATTERN = re.compile(r"""from\s+["']([^"']+)["']""")
GO_IMPORT_PATTERN = re.compile(r'"([^"]+)"')
RUST_USE_PATTERN = re.compile(r"^\s*(?:use|mod)\s+([A-Za-z0-9_:]+)", flags=re.MULTILINE)
JAVA_IMPORT_PATTERN = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)", flags=re.MULTILINE)


class StaticContextBuilder:
    def __init__(
        self,
        *,
        toolkit_factory: AgentToolkitFactory,
        static_tooling: StaticTooling,
    ) -> None:
        self.toolkit_factory = toolkit_factory
        self.static_tooling = static_tooling

    async def build(
        self,
        *,
        repo_root: Path,
        targets: list[str],
        config: AppConfig,
        rules: dict[str, Any],
        execution_environment: ExecutionEnvironment | None = None,
    ) -> StaticContextBundle:
        normalized_targets = sorted({item for item in targets if item})
        language_profile = await asyncio.to_thread(
            detect_language_profile,
            repo_root,
            enabled_languages=config.static_review.language_adapters_enabled,
        )
        startup_topology = await self._build_startup_topology(repo_root, language_profile)
        project_topology = await asyncio.to_thread(
            build_project_topology,
            repo_root,
            language_profile=language_profile,
            startup_topology=startup_topology,
        )
        tool_adapters = await asyncio.to_thread(
            build_static_tool_adapters,
            repo_root,
            language_profile=language_profile,
            python_commands={
                "ruff": config.static_review.ruff_command,
                "mypy": config.static_review.mypy_command,
                "bandit": config.static_review.bandit_command,
            },
        )
        env = execution_environment.command_env() if execution_environment is not None else None
        baseline_findings, baseline_artifacts = await self.static_tooling.review(
            repo_root,
            normalized_targets,
            config,
            rules,
            env=env,
            language_profile=language_profile,
            tool_adapters=tool_adapters,
        )
        repo_map_summary = await asyncio.to_thread(
            self._build_repo_map_summary,
            repo_root,
            startup_topology,
            project_topology,
            language_profile,
        )
        baseline_static_digest = await asyncio.to_thread(
            self._build_baseline_static_digest,
            baseline_findings,
        )
        target_digest = await asyncio.to_thread(
            self._build_target_digest,
            repo_root,
            normalized_targets,
            project_topology,
            baseline_findings,
        )
        import_adjacency_digest = await asyncio.to_thread(
            self._build_import_adjacency_digest,
            repo_root,
            target_digest.high_signal_targets or target_digest.top_targets,
        )
        config_anchor_digest = await asyncio.to_thread(
            self._build_config_anchor_digest,
            project_topology,
        )
        tool_coverage_summary = await asyncio.to_thread(
            self._build_tool_coverage_summary,
            tool_adapters,
            baseline_artifacts,
        )
        related_files = list(import_adjacency_digest.related_files)
        return StaticContextBundle(
            startup_topology=startup_topology,
            project_topology=project_topology,
            repo_map_summary=repo_map_summary,
            language_profile=language_profile,
            top_targets=list(target_digest.top_targets),
            high_signal_targets=list(target_digest.high_signal_targets),
            related_files=related_files,
            target_digest=target_digest,
            import_adjacency_digest=import_adjacency_digest,
            config_anchor_digest=config_anchor_digest,
            baseline_static_digest=baseline_static_digest,
            baseline_tool_digest=baseline_static_digest,
            tool_coverage_summary=tool_coverage_summary,
            enabled=True,
        )

    async def _build_startup_topology(
        self,
        repo_root: Path,
        language_profile,
    ) -> StartupTopology:
        if "python" not in language_profile.ecosystems and "python" not in language_profile.languages:
            return StartupTopology(
                repo_root=str(repo_root),
                src_layout=repo_root.joinpath("src").is_dir(),
            )
        return await self.toolkit_factory.discover_startup_topology(repo_root)

    def _build_repo_map_summary(
        self,
        repo_root: Path,
        startup_topology: StartupTopology,
        project_topology,
        language_profile,
    ) -> RepoMapSummary:
        python_package_roots: set[str] = set()
        for path in repo_root.rglob("*.py"):
            if self._should_skip(path, repo_root):
                continue
            relative_path = path.relative_to(repo_root)
            if relative_path.name == "__init__.py":
                python_package_roots.add(relative_path.parent.as_posix())
        startup_contexts = sorted(
            {
                entry.context
                for entry in startup_topology.entrypoints
            }
            | {
                anchor.context
                for anchor in startup_topology.config_anchors
            }
        )
        config_files = sorted(
            {
                anchor.path
                for anchor in project_topology.config_anchors
                if anchor.path and repo_root.joinpath(anchor.path).exists()
            }
            | set(project_topology.dependency_manifests)
            | set(project_topology.lockfiles)
        )[:40]
        return RepoMapSummary(
            package_roots=sorted(item for item in project_topology.package_roots if item)[:40],
            python_package_roots=sorted(item for item in python_package_roots if item)[:20],
            test_roots=sorted(item for item in project_topology.test_roots if item)[:40],
            src_layout=startup_topology.src_layout or repo_root.joinpath("src").is_dir(),
            startup_contexts=startup_contexts,
            languages=list(language_profile.languages),
            ecosystems=list(language_profile.ecosystems),
            config_files=config_files,
            dependency_manifests=list(project_topology.dependency_manifests)[:30],
            lockfiles=list(project_topology.lockfiles)[:20],
        )

    def _build_baseline_static_digest(
        self,
        findings: list[Finding],
    ) -> BaselineStaticDigest:
        severity_counts = Counter(finding.severity.value for finding in findings)
        noisy_rule_counts = Counter(
            finding.rule_id for finding in findings if finding.rule_id in LOW_VALUE_RULE_IDS
        )
        file_scores: Counter[str] = Counter()
        for finding in findings:
            if not finding.path:
                continue
            file_scores[finding.path] += self._finding_weight(finding)
        candidate_high_value_files = [
            path
            for path, _ in file_scores.most_common(20)
            if path
        ]
        top_findings = [
            {
                "rule_id": finding.rule_id,
                "severity": finding.severity.value,
                "message": finding.message,
                "category": finding.category,
                "root_cause_class": finding.root_cause_class,
                "path": finding.path,
                "line": finding.line,
                "language": str(finding.evidence.get("language", "")) or None,
                "ecosystem": str(finding.evidence.get("ecosystem", "")) or None,
            }
            for finding in self._prioritize_findings(findings)[:50]
        ]
        return BaselineStaticDigest(
            total_findings=len(findings),
            severity_counts=dict(severity_counts),
            noisy_rule_counts=dict(noisy_rule_counts),
            top_findings=top_findings,
            candidate_high_value_files=candidate_high_value_files,
        )

    def _build_target_digest(
        self,
        repo_root: Path,
        targets: list[str],
        project_topology,
        findings: list[Finding],
    ) -> StaticTargetDigest:
        target_set = set(targets)
        candidate_paths = set(targets)
        candidate_paths.update(
            item.path
            for item in project_topology.entrypoints
            if item.path and repo_root.joinpath(item.path).exists()
        )
        candidate_paths.update(
            item.path
            for item in project_topology.config_anchors
            if item.path and repo_root.joinpath(item.path).exists()
        )
        candidate_paths.update(project_topology.dependency_manifests)
        candidate_paths.update(project_topology.lockfiles)

        findings_by_path: defaultdict[str, list[Finding]] = defaultdict(list)
        for finding in findings:
            if finding.path:
                findings_by_path[finding.path].append(finding)

        entrypoint_paths = {item.path for item in project_topology.entrypoints if item.path}
        anchor_paths = {item.path for item in project_topology.config_anchors if item.path}
        manifest_paths = set(project_topology.dependency_manifests)
        lockfile_paths = set(project_topology.lockfiles)
        scored: list[tuple[int, str]] = []
        for path in candidate_paths:
            score = 0
            lowered = path.lower()
            if path in target_set:
                score += 200
            if path in entrypoint_paths:
                score += 160
            if path in anchor_paths:
                score += 150
            if path in manifest_paths:
                score += 140
            if path in lockfile_paths:
                score += 120
            if any(keyword in lowered for keyword in PRIORITY_KEYWORDS):
                score += 80
            if lowered.endswith(("__init__.py", "index.ts", "index.js")):
                score -= 10
            if lowered.endswith((".md", ".txt")):
                score -= 40
            path_findings = findings_by_path.get(path, [])
            if path_findings:
                score += sum(self._finding_weight(finding) for finding in path_findings)
                if self._is_docstring_only(path_findings):
                    score -= 120
            scored.append((score, path))

        prioritized_targets = [
            path for _, path in sorted(scored, key=lambda item: (-item[0], item[1]))
        ]
        high_signal_targets = [
            path
            for score, path in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score >= 120
        ][:20]
        low_signal_targets = [
            path
            for score, path in sorted(scored, key=lambda item: (item[0], item[1]))
            if score <= 60
        ][:20]
        return StaticTargetDigest(
            prioritized_targets=prioritized_targets,
            top_targets=prioritized_targets[:40],
            high_signal_targets=high_signal_targets,
            low_signal_targets=low_signal_targets,
        )

    def _build_import_adjacency_digest(
        self,
        repo_root: Path,
        candidate_paths: list[str],
    ) -> ImportAdjacencyDigest:
        module_index = self._build_module_index(repo_root)
        edges: list[dict[str, Any]] = []
        related_files: list[str] = []
        seen_related: set[str] = set()
        for relative_path in candidate_paths[:20]:
            path = repo_root / relative_path
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            resolved_imports: list[str] = []
            for module_name in self._import_names(path, content):
                resolved = self._resolve_local_module(
                    repo_root,
                    module_index,
                    relative_path,
                    module_name,
                )
                if not resolved or resolved == relative_path:
                    continue
                resolved_imports.append(resolved)
                if resolved not in seen_related:
                    seen_related.add(resolved)
                    related_files.append(resolved)
            if resolved_imports:
                edges.append(
                    {
                        "source": relative_path,
                        "resolved_imports": sorted(set(resolved_imports))[:10],
                    }
                )
        return ImportAdjacencyDigest(
            related_files=related_files[:20],
            edges=edges[:20],
        )

    def _build_config_anchor_digest(
        self,
        project_topology,
    ) -> list[dict[str, Any]]:
        return [
            {
                "path": anchor.path,
                "context": anchor.context,
                "language": anchor.language,
                "ecosystem": anchor.ecosystem,
                "anchor_type": anchor.anchor_type,
                "status": anchor.status,
                "repair_hint": anchor.repair_hint,
            }
            for anchor in project_topology.config_anchors
        ]

    def _build_tool_coverage_summary(
        self,
        tool_adapters,
        artifacts: dict[str, Any],
    ) -> ToolCoverageSummary:
        statuses = dict(artifacts.get("external_tools", {}))
        enabled_tools = [adapter.name for adapter in tool_adapters]
        unavailable_tools = [
            name for name, data in statuses.items()
            if isinstance(data, dict) and data.get("status") == "missing"
        ]
        executed_tools = [
            name for name, data in statuses.items()
            if isinstance(data, dict) and data.get("status") == "executed"
        ]
        tool_statuses = {
            name: str(data.get("status", "unknown"))
            for name, data in statuses.items()
            if isinstance(data, dict)
        }
        for name in enabled_tools:
            tool_statuses.setdefault(name, "planned")
        return ToolCoverageSummary(
            enabled_tools=enabled_tools,
            unavailable_tools=unavailable_tools,
            executed_tools=executed_tools,
            tool_statuses=tool_statuses,
        )

    def _build_module_index(self, repo_root: Path) -> dict[str, str]:
        index: dict[str, str] = {}
        for path in repo_root.rglob("*"):
            if not path.is_file() or self._should_skip(path, repo_root):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            relative = path.relative_to(repo_root)
            normalized = relative.as_posix()
            keys = {
                relative.stem,
                normalized,
                normalized.rsplit(".", 1)[0],
                normalized.replace("/", ".").rsplit(".", 1)[0],
            }
            if relative.parts and relative.parts[0] == "src" and len(relative.parts) > 1:
                without_src = Path(*relative.parts[1:])
                keys.add(without_src.as_posix())
                keys.add(without_src.as_posix().rsplit(".", 1)[0])
                keys.add(without_src.as_posix().replace("/", ".").rsplit(".", 1)[0])
            for key in keys:
                if key:
                    index.setdefault(key, normalized)
        return index

    def _resolve_local_module(
        self,
        repo_root: Path,
        module_index: dict[str, str],
        relative_path: str,
        module_name: str,
    ) -> str | None:
        cleaned = module_name.strip().strip('"').strip("'")
        if not cleaned:
            return None
        if cleaned in module_index:
            return module_index[cleaned]
        if cleaned.startswith(("./", "../")):
            base_dir = (repo_root / relative_path).parent
            candidate_base = (base_dir / cleaned).resolve()
            for suffix in ("", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt"):
                candidate = candidate_base if not suffix else Path(f"{candidate_base}{suffix}")
                if candidate.is_file():
                    return candidate.relative_to(repo_root).as_posix()
            for child in ("__init__.py", "index.ts", "index.js"):
                candidate = candidate_base / child
                if candidate.is_file():
                    return candidate.relative_to(repo_root).as_posix()
        last_segment = cleaned.split("/")[-1].split(".")[-1].split("::")[-1]
        if last_segment in module_index:
            return module_index[last_segment]
        direct_candidates = [
            Path(cleaned.replace(".", "/") + suffix)
            for suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt")
        ]
        direct_candidates.extend(
            [
                Path(cleaned.replace(".", "/")) / "__init__.py",
                Path(cleaned.replace(".", "/")) / "index.ts",
                Path(cleaned.replace(".", "/")) / "index.js",
                Path("src") / Path(cleaned.replace(".", "/") + ".py"),
                Path("src") / Path(cleaned.replace(".", "/") + ".ts"),
                Path("src") / Path(cleaned.replace(".", "/") + ".js"),
            ]
        )
        for candidate in direct_candidates:
            if repo_root.joinpath(candidate).exists():
                return candidate.as_posix()
        return None

    def _import_names(self, path: Path, content: str) -> list[str]:
        suffix = path.suffix.lower()
        if suffix == ".py":
            try:
                tree = ast.parse(content or "\n")
            except SyntaxError:
                return []
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
            return names
        if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
            return [
                *REQUIRE_PATTERN.findall(content),
                *JS_IMPORT_PATTERN.findall(content),
            ]
        if suffix == ".go":
            return GO_IMPORT_PATTERN.findall(content)
        if suffix == ".rs":
            return RUST_USE_PATTERN.findall(content)
        if suffix in {".java", ".kt", ".kts"}:
            return JAVA_IMPORT_PATTERN.findall(content)
        return [
            *IMPORT_LINE_PATTERN.findall(content),
            *FROM_LINE_PATTERN.findall(content),
        ]

    def _prioritize_findings(self, findings: list[Finding]) -> list[Finding]:
        severity_rank = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }
        return sorted(
            findings,
            key=lambda finding: (
                -severity_rank.get(finding.severity.value, 0),
                finding.rule_id in LOW_VALUE_RULE_IDS,
                finding.path or "",
                finding.line or 0,
            ),
        )

    def _finding_weight(self, finding: Finding) -> int:
        severity_weight = {
            "critical": 120,
            "high": 90,
            "medium": 50,
            "low": 10,
        }
        weight = severity_weight.get(finding.severity.value, 0)
        if finding.rule_id in LOW_VALUE_RULE_IDS:
            weight -= 20
        if finding.category in {"correctness", "architecture", "config", "dependency", "security", "typing"}:
            weight += 25
        if finding.root_cause_class in {"startup", "config", "dependency", "application"}:
            weight += 20
        return weight

    def _is_docstring_only(self, findings: list[Finding]) -> bool:
        return bool(findings) and all(finding.rule_id in LOW_VALUE_RULE_IDS for finding in findings)

    def _should_skip(self, path: Path, repo_root: Path) -> bool:
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            return True
        return any(part in SKIP_PARTS for part in relative.parts)
