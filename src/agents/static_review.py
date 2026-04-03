from __future__ import annotations

from agents.base import BaseAgent
from core.config import AgentRuntimeConfig
from core.models import AgentKind, AgentResult, CompletionReason, EvidenceArtifact, Finding, FixRequest, RunContext, Severity, Task, TaskStatus, TaskType, ToolPermissionSet
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
        session = self.build_session(
            task=task,
            context=context,
            objective=(
                "Autonomously review the target code like a senior static analysis agent. "
                "Use multiple tools to inspect code, then produce findings and fix requests."
            ),
        )
        tools = self.toolkit_factory.build_static_toolkit(self.permissions, context=context)
        kernel_result = await self.kernel.run_session(
            session=session,
            tools=tools,
            permissions=self.permissions,
            context=context,
        )

        deterministic_findings = self._tool_findings(kernel_result)
        semantic_findings = [
            self.finding_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("findings", [])
            if isinstance(item, dict)
        ]
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
                fallback_findings,
            )
        )
        explicit_handoffs = [
            self.fix_request_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("fix_requests", [])
            if isinstance(item, dict)
        ]
        derived_handoffs = self._derive_fix_requests_from_findings(findings)
        handoffs = self._merge_handoffs(explicit_handoffs, derived_handoffs)
        artifacts = self.base_artifacts(kernel_result, handoffs=handoffs)
        artifacts["deterministic_findings"] = len(deterministic_findings)
        artifacts["semantic_findings"] = len(semantic_findings) + len(fallback_findings)
        artifacts["high_value_findings"] = len(
            [finding for finding in findings if self._is_high_value_finding(finding)]
        )
        artifacts["reviewed_files"] = list(task.targets)

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
                    metadata={"rule_id": finding.rule_id, "category": finding.category},
                )
            )
            if len(requests) >= 6:
                break
        return requests

    def _merge_handoffs(
        self,
        explicit_handoffs: list[FixRequest],
        derived_handoffs: list[FixRequest],
    ) -> list[FixRequest]:
        merged: list[FixRequest] = []
        seen: set[tuple[str, str, str]] = set()
        for request in [*explicit_handoffs, *derived_handoffs]:
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


from core.agent_kernel import AgentKernelResult  # noqa: E402
