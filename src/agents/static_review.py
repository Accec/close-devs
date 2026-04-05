from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent
from core.config import AgentRuntimeConfig
from core.models import AgentKind, AgentResult, CompletionReason, EvidenceArtifact, Finding, FixRequest, RunContext, Severity, StartupConfigAnchor, StartupEntrypoint, StartupTopology, StaticContextBundle, Task, TaskStatus, TaskType, ToolPermissionSet
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient
from tools.agent_toolkit import AgentToolkitFactory


class StaticReviewAgent(BaseAgent):
    kind = AgentKind.STATIC_REVIEW
    allowed_task_types = frozenset({TaskType.STATIC_REVIEW, TaskType.VALIDATION_STATIC})
    default_tools = frozenset(
        {
            "read_file",
            "search_repo",
            "git_diff",
            "ast_summary",
            "run_static_review",
            "shell_readonly",
        }
    )
    low_value_rule_ids = frozenset(
        {
            "missing-module-docstring",
            "missing-final-newline",
            "trailing-whitespace",
            "excessive-eof-blank-lines",
        }
    )

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        runtime_config: AgentRuntimeConfig | None = None,
        toolkit_factory: AgentToolkitFactory | None = None,
        permissions: ToolPermissionSet | None = None,
    ) -> None:
        runtime = runtime_config or AgentRuntimeConfig()
        permission_set = permissions or ToolPermissionSet(
            allowed_tools=frozenset(runtime.allowed_tools or self.default_tools),
            allow_write=False,
        )
        super().__init__(
            llm_client=llm_client or MockLLMClient(),
            runtime_config=runtime,
            permissions=permission_set,
        )
        self.toolkit_factory = toolkit_factory or AgentToolkitFactory()

    async def run(self, task: Task, context: RunContext) -> AgentResult:
        self.ensure_task_type(task)
        filtered_targets, skipped_targets = self._filter_existing_targets(task.targets, context)
        session = self.build_session(
            task=task,
            context=context,
            objective=(
                "Autonomously review the target code like a senior static analysis agent. "
                "Use multiple tools to inspect code, then produce findings and fix requests."
            ),
        )
        session.targets = filtered_targets
        tools = self.toolkit_factory.build_static_toolkit(self.permissions, context=context)
        kernel_result = await self.kernel.run_session(
            session=session,
            tools=tools,
            permissions=self.permissions,
            context=context,
        )
        static_context = self._resolve_static_context(task, context)
        startup_topology = (
            static_context.startup_topology
            if static_context is not None
            else await self.toolkit_factory.discover_startup_topology(Path(context.working_repo_root))
        )
        startup_advisory_enabled = task.task_type == TaskType.STATIC_REVIEW

        deterministic_findings = self._tool_findings(kernel_result)
        semantic_findings = [
            self.finding_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("findings", [])
            if isinstance(item, dict)
        ]
        startup_findings = (
            self._startup_topology_findings(startup_topology)
            if startup_advisory_enabled
            else []
        )
        fallback_findings = self._fallback_semantic_findings(
            kernel_result=kernel_result,
            deterministic_findings=deterministic_findings,
            semantic_findings=semantic_findings,
            task=task,
        )
        findings = self._prioritize_findings(
            self._merge_findings(
                deterministic_findings,
                semantic_findings,
                startup_findings,
                fallback_findings,
            )
        )
        explicit_handoffs = [
            self.fix_request_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("fix_requests", [])
            if isinstance(item, dict)
        ]
        derived_handoffs = self._derive_fix_requests_from_findings(findings)
        startup_handoffs = (
            self._startup_topology_handoffs(startup_topology)
            if startup_advisory_enabled
            else []
        )
        handoffs = self._merge_handoffs(explicit_handoffs, derived_handoffs, startup_handoffs)
        artifacts = self.base_artifacts(kernel_result, handoffs=handoffs)
        artifacts["deterministic_findings"] = len(deterministic_findings)
        artifacts["semantic_findings"] = len(semantic_findings) + len(fallback_findings)
        artifacts["high_value_findings"] = len(
            [finding for finding in findings if self._is_high_value_finding(finding)]
        )
        artifacts["startup_topology"] = startup_topology
        artifacts["project_topology"] = (
            static_context.project_topology if static_context is not None else None
        )
        artifacts["language_profile"] = (
            static_context.language_profile if static_context is not None else None
        )
        artifacts["tool_coverage_summary"] = (
            static_context.tool_coverage_summary if static_context is not None else None
        )
        artifacts["entrypoint_count"] = len(startup_topology.entrypoints)
        artifacts["config_anchor_count"] = len(startup_topology.config_anchors)
        artifacts["startup_handoffs"] = [self.fix_request_to_dict(item) for item in startup_handoffs]
        if static_context is not None:
            artifacts["static_context_summary"] = static_context.summary()
            artifacts["project_topology_summary"] = {
                "languages": list(static_context.project_topology.languages),
                "ecosystems": list(static_context.project_topology.ecosystems),
                "entrypoint_count": len(static_context.project_topology.entrypoints),
                "config_anchor_count": len(static_context.project_topology.config_anchors),
                "dependency_manifest_count": len(static_context.project_topology.dependency_manifests),
                "lockfile_count": len(static_context.project_topology.lockfiles),
            }
            artifacts["top_target_count"] = len(static_context.top_targets)
            artifacts["baseline_digest_counts"] = {
                "total_findings": static_context.baseline_static_digest.total_findings,
                "severity_counts": dict(static_context.baseline_static_digest.severity_counts),
                "noisy_rule_counts": dict(static_context.baseline_static_digest.noisy_rule_counts),
            }
            artifacts["generic_language_review"] = bool(static_context.language_profile.generic_review)
        artifacts["reviewed_files"] = list(filtered_targets)
        artifacts["skipped_missing_targets"] = skipped_targets
        inspected_paths = self._inspected_paths(kernel_result)
        artifacts["investigation_depth"] = {
            "requested_targets": len(task.targets),
            "existing_targets": len(filtered_targets),
            "inspected_files": len(inspected_paths),
            "ast_summaries": len(
                [call for call in kernel_result.session.tool_calls if call.tool_name == "ast_summary"]
            ),
            "repo_searches": len(
                [call for call in kernel_result.session.tool_calls if call.tool_name == "search_repo"]
            ),
            "cross_file_investigation": len(set(inspected_paths)) > 1,
        }

        summary = str(
            kernel_result.final_response.get(
                "summary",
                (
                    f"Autonomous static review completed with {len(findings)} findings "
                    f"and {len(handoffs)} handoffs."
                ),
            )
        )
        status = TaskStatus.SUCCEEDED
        if kernel_result.session.completion_reason == CompletionReason.FAILED:
            status = TaskStatus.FAILED
        return AgentResult(
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            status=status,
            summary=summary,
            findings=findings,
            artifacts=artifacts,
            errors=list(kernel_result.errors),
        )

    def _resolve_static_context(
        self,
        task: Task,
        context: RunContext,
    ) -> StaticContextBundle | None:
        if context.static_context is not None:
            return context.static_context
        value = task.payload.get("static_context")
        return value if isinstance(value, StaticContextBundle) else None

    def _tool_findings(self, kernel_result: "AgentKernelResult") -> list[Finding]:
        findings: list[Finding] = []
        for call in kernel_result.session.tool_calls:
            if call.tool_name != "run_static_review" or call.status != "succeeded":
                continue
            for item in call.output.get("findings", []):
                if not isinstance(item, dict):
                    continue
                findings.append(self.finding_from_dict(item, default_source_agent=self.kind))
        return findings

    def _merge_findings(
        self,
        *finding_groups: list[Finding],
    ) -> list[Finding]:
        findings: list[Finding] = []
        seen: set[str] = set()
        for group in finding_groups:
            for finding in group:
                if finding.fingerprint in seen:
                    continue
                seen.add(finding.fingerprint)
                findings.append(finding)
        return findings

    def _fallback_semantic_findings(
        self,
        *,
        kernel_result: "AgentKernelResult",
        deterministic_findings: list[Finding],
        semantic_findings: list[Finding],
        task: Task,
    ) -> list[Finding]:
        summary = str(kernel_result.final_response.get("summary", "")).strip()
        if not summary:
            return []
        if semantic_findings or any(self._is_high_value_finding(item) for item in deterministic_findings):
            return []
        if not self._summary_implies_high_value_issue(summary):
            return []

        candidate_path = self._primary_review_path(kernel_result, task.targets)
        severity = Severity.HIGH if any(
            keyword in summary.lower()
            for keyword in ("correctness", "dependency", "bootstrap", "initialization", "failure")
        ) else Severity.MEDIUM
        category = "architecture" if "bootstrap" in summary.lower() or "initialization" in summary.lower() else "correctness"
        return [
            Finding(
                source_agent=self.kind,
                severity=severity,
                rule_id="semantic-review-observation",
                message=summary,
                category=category,
                root_cause_class="application" if category == "correctness" else "config",
                path=candidate_path,
                evidence={
                    "summary": summary,
                    "inspected_files": self._inspected_paths(kernel_result),
                },
            )
        ]

    def _derive_fix_requests_from_findings(
        self,
        findings: list[Finding],
    ) -> list[FixRequest]:
        requests: list[FixRequest] = []
        for finding in findings:
            if not self._is_high_value_finding(finding):
                continue
            requests.append(
                FixRequest(
                    source_agent=self.kind,
                    title=f"Investigate {finding.rule_id}",
                    description=finding.message,
                    recommended_change=self._recommended_change(finding),
                    severity=finding.severity,
                    kind=self._fix_request_kind(finding),
                    confidence=self._fix_request_confidence(finding),
                    affected_files=[finding.path] if finding.path else [],
                    evidence=[
                        EvidenceArtifact(
                            kind="finding",
                            title=finding.rule_id,
                            summary=finding.message,
                            path=finding.path,
                            data={"line": finding.line, "category": finding.category},
                        )
                    ],
                    metadata={
                        "rule_id": finding.rule_id,
                        "category": finding.category,
                        "root_cause_class": finding.root_cause_class,
                    },
                )
            )
            if len(requests) >= 6:
                break
        return requests

    def _startup_topology_findings(self, topology: StartupTopology) -> list[Finding]:
        findings: list[Finding] = []
        for anchor in topology.config_anchors:
            if anchor.status != "missing" or not anchor.repair_hint or anchor.anchor_type == "env_template":
                continue
            entrypoint = self._entrypoint_for_anchor(topology, anchor)
            findings.append(
                Finding(
                    source_agent=self.kind,
                    severity=Severity.MEDIUM,
                    rule_id="startup-topology-anchor-missing",
                    message=self._startup_topology_message(entrypoint, anchor),
                    category="config",
                    root_cause_class="startup",
                    path=anchor.path or (entrypoint.path if entrypoint is not None else None),
                    evidence={
                        "advisory": True,
                        "startup_context": anchor.context,
                        "entrypoint_path": entrypoint.path if entrypoint is not None else None,
                        "config_anchor_path": anchor.path,
                        "repair_hint": anchor.repair_hint,
                    },
                )
            )
        return findings

    def _startup_topology_handoffs(self, topology: StartupTopology) -> list[FixRequest]:
        handoffs: list[FixRequest] = []
        for anchor in topology.config_anchors:
            if anchor.status != "missing" or not anchor.repair_hint or anchor.anchor_type == "env_template":
                continue
            entrypoint = self._entrypoint_for_anchor(topology, anchor)
            handoffs.append(
                FixRequest(
                    source_agent=self.kind,
                    title=f"Review {anchor.context} startup topology",
                    description=self._startup_topology_message(entrypoint, anchor),
                    recommended_change=self._startup_topology_recommended_change(entrypoint, anchor),
                    severity=Severity.MEDIUM,
                    kind="startup",
                    confidence=0.82,
                    affected_files=[
                        item
                        for item in [entrypoint.path if entrypoint is not None else None, anchor.path]
                        if item
                    ],
                    evidence=[
                        EvidenceArtifact(
                            kind="startup-topology",
                            title=anchor.context,
                            summary=self._startup_topology_message(entrypoint, anchor),
                            path=entrypoint.path if entrypoint is not None else anchor.path,
                            data={
                                "advisory": True,
                                "startup_context": anchor.context,
                                "entrypoint_path": entrypoint.path if entrypoint is not None else None,
                                "config_anchor_path": anchor.path,
                                "repair_hint": anchor.repair_hint,
                            },
                        )
                    ],
                    metadata={
                        "root_cause_class": "startup",
                        "startup_context": anchor.context,
                        "entrypoint_path": entrypoint.path if entrypoint is not None else None,
                        "config_anchor_path": anchor.path,
                        "repair_hint": anchor.repair_hint,
                        "advisory": True,
                    },
                )
            )
        return handoffs

    def _merge_handoffs(
        self,
        explicit_handoffs: list[FixRequest],
        derived_handoffs: list[FixRequest],
        startup_handoffs: list[FixRequest],
    ) -> list[FixRequest]:
        merged: list[FixRequest] = []
        seen: set[tuple[str, str, str]] = set()
        for request in [*explicit_handoffs, *derived_handoffs, *startup_handoffs]:
            key = (
                request.title,
                request.recommended_change,
                ",".join(request.affected_files),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(request)
        return merged

    def _prioritize_findings(self, findings: list[Finding]) -> list[Finding]:
        severity_rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        return sorted(
            findings,
            key=lambda finding: (
                0 if self._is_high_value_finding(finding) else 1,
                severity_rank[finding.severity],
                1 if finding.rule_id in self.low_value_rule_ids else 0,
                finding.path or "",
                finding.line or 0,
                finding.rule_id,
            ),
        )

    def _is_high_value_finding(self, finding: Finding) -> bool:
        if finding.rule_id in self.low_value_rule_ids:
            return False
        if finding.category in {"correctness", "architecture", "security", "runtime", "typing", "dependency"}:
            return True
        return finding.severity in {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}

    def _summary_implies_high_value_issue(self, summary: str) -> bool:
        lowered = summary.lower()
        if "no functional" in lowered or "only low-severity documentation" in lowered:
            return False
        return any(
            keyword in lowered
            for keyword in (
                "higher-value",
                "correctness",
                "operability",
                "bootstrap",
                "initialization",
                "dependency",
                "contract",
                "unsafe",
                "failure",
                "broken",
                "misconfiguration",
            )
        )

    def _primary_review_path(
        self,
        kernel_result: "AgentKernelResult",
        targets: list[str],
    ) -> str | None:
        inspected = self._inspected_paths(kernel_result)
        if inspected:
            return inspected[-1]
        for target in targets:
            if target.endswith(".py"):
                return target
        return targets[0] if targets else None

    def _inspected_paths(self, kernel_result: "AgentKernelResult") -> list[str]:
        paths: list[str] = []
        for call in kernel_result.session.tool_calls:
            if call.tool_name not in {"read_file", "ast_summary"}:
                continue
            path = str(call.output.get("path") or call.tool_input.get("path") or "")
            if path:
                paths.append(path)
        return paths

    def _recommended_change(self, finding: Finding) -> str:
        path = finding.path or "the affected module"
        if finding.rule_id == "semantic-review-observation":
            return f"Inspect {path} and convert the reported static review concern into a concrete code or configuration fix."
        if finding.rule_id in {"bare-except", "broad-exception-catch", "swallowed-exception"}:
            return f"Narrow exception handling in {path}, preserve the original failure signal, and add explicit logging or re-raise behavior."
        if finding.rule_id == "mutable-default-argument":
            return f"Replace mutable defaults in {path} with None sentinels or immutable values and initialize state inside the function."
        if finding.rule_id == "assert-used-for-runtime-validation":
            return f"Replace runtime asserts in {path} with explicit validation and domain-specific exceptions."
        return f"Update {path} to address {finding.rule_id}."

    def _filter_existing_targets(
        self,
        targets: list[str],
        context: RunContext,
    ) -> tuple[list[str], list[str]]:
        existing: list[str] = []
        skipped: list[str] = []
        for target in targets:
            path = context.working_repo_root / target
            if path.exists():
                existing.append(target)
            else:
                skipped.append(target)
        return existing, skipped

    def _fix_request_kind(self, finding: Finding) -> str:
        if finding.category == "dependency":
            return "dependency"
        if finding.rule_id == "startup-topology-anchor-missing":
            return "startup"
        if finding.category in {"architecture", "config"} or finding.root_cause_class == "config":
            return "config"
        if finding.category == "runtime":
            return "runtime"
        return "code"

    def _fix_request_confidence(self, finding: Finding) -> float:
        if finding.rule_id == "startup-topology-anchor-missing":
            return 0.82
        if finding.rule_id == "semantic-review-observation":
            return 0.7
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}:
            return 0.88
        if finding.severity == Severity.MEDIUM:
            return 0.78
        return 0.65

    def _entrypoint_for_anchor(
        self,
        topology: StartupTopology,
        anchor: StartupConfigAnchor,
    ) -> StartupEntrypoint | None:
        return next(
            (
                item
                for item in topology.entrypoints
                if item.config_anchor_path == anchor.path or item.path == anchor.path
            ),
            None,
        )

    def _startup_topology_message(
        self,
        entrypoint: StartupEntrypoint | None,
        anchor: StartupConfigAnchor,
    ) -> str:
        if entrypoint is not None:
            return (
                f"Static review identified `{entrypoint.path}` as a `{anchor.context}` startup entrypoint, "
                f"but the expected configuration/bootstrap anchor `{anchor.path}` is missing or not aligned "
                "with the repository src layout."
            )
        return (
            f"Static review identified `{anchor.path}` as a `{anchor.context}` startup/config anchor, "
            "but it is missing the expected src-layout bootstrap configuration."
        )

    def _startup_topology_recommended_change(
        self,
        entrypoint: StartupEntrypoint | None,
        anchor: StartupConfigAnchor,
    ) -> str:
        target = anchor.path or (entrypoint.path if entrypoint is not None else "the startup configuration")
        return (
            f"Add the expected `{anchor.anchor_type}` bootstrap at `{target}` so the `{anchor.context}` startup "
            "chain resolves the repository src layout deterministically."
        )


from core.agent_kernel import AgentKernelResult  # noqa: E402
