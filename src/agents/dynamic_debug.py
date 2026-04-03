from __future__ import annotations

from agents.base import BaseAgent
from core.agent_kernel import AgentKernelResult
from core.config import AgentRuntimeConfig
from core.models import AgentKind, AgentResult, CompletionReason, Finding, RunContext, Task, TaskStatus, TaskType, ToolPermissionSet
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient
from tools.agent_toolkit import AgentToolkitFactory


class DynamicDebugAgent(BaseAgent):
    kind = AgentKind.DYNAMIC_DEBUG
    allowed_task_types = frozenset({TaskType.DYNAMIC_DEBUG, TaskType.VALIDATION_DYNAMIC})
    default_tools = frozenset(
        {
            "read_file",
            "search_repo",
            "run_test_command",
            "parse_traceback",
            "shell_readonly",
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
                "Autonomously debug the repository like a senior runtime agent. "
                "Choose commands, parse failures, and produce fix requests without writing files."
            ),
        )
        tools = self.toolkit_factory.build_dynamic_toolkit(self.permissions, context=context)
        kernel_result = await self.kernel.run_session(
            session=session,
            tools=tools,
            permissions=self.permissions,
            context=context,
        )
        runtime_findings = self._runtime_findings(kernel_result)
        diagnosis_findings = [
            self.finding_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("findings", [])
            if isinstance(item, dict)
        ]
        findings = self._merge_findings(runtime_findings, diagnosis_findings)
        handoffs = [
            self.fix_request_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("fix_requests", [])
            if isinstance(item, dict)
        ]
        artifacts = self.base_artifacts(kernel_result, handoffs=handoffs)
        artifacts["commands"] = [
            call.output
            for call in kernel_result.session.tool_calls
            if call.tool_name == "run_test_command"
        ]
        artifacts["diagnosis_findings"] = len(diagnosis_findings)
        if kernel_result.final_response.get("suggestions"):
            artifacts["agent_diagnosis"] = {
                "summary": str(kernel_result.final_response.get("summary", "")),
                "suggestions": list(kernel_result.final_response.get("suggestions", [])),
                "extra_findings": len(diagnosis_findings),
            }
        summary = str(
            kernel_result.final_response.get(
                "summary",
                (
                    f"Autonomous dynamic debugging completed with {len(findings)} findings "
                    f"and {len(handoffs)} handoffs."
                ),
            )
        )
        status = (
            TaskStatus.FAILED
            if kernel_result.session.completion_reason == CompletionReason.FAILED
            else TaskStatus.SUCCEEDED
        )
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

    def _runtime_findings(self, kernel_result: AgentKernelResult) -> list[Finding]:
        findings: list[Finding] = []
        for call in kernel_result.session.tool_calls:
            if call.tool_name != "run_test_command" or call.status != "succeeded":
                continue
            if call.output.get("timed_out"):
                findings.append(
                    Finding(
                        source_agent=self.kind,
                        severity=self._severity_from_output(call.output, timeout=True),
                        rule_id="command-timeout",
                        message=f"Command timed out: {call.output.get('command', '')}",
                        category="runtime",
                        evidence={"command": call.output.get("command", "")},
                    )
                )
            elif int(call.output.get("returncode", 0) or 0) != 0:
                findings.append(
                    Finding(
                        source_agent=self.kind,
                        severity=self._severity_from_output(call.output, timeout=False),
                        rule_id="command-failed",
                        message=(
                            f"Command failed with exit code {call.output.get('returncode', 1)}: "
                            f"{call.output.get('command', '')}"
                        ),
                        category="runtime",
                        evidence={"command": call.output.get("command", "")},
                    )
                )
        return findings

    def _severity_from_output(self, output: dict[str, object], *, timeout: bool) -> "Severity":
        from core.models import Severity

        if timeout:
            return Severity.HIGH
        stderr = str(output.get("stderr", ""))
        stdout = str(output.get("stdout", ""))
        if "Traceback" in stderr or "Traceback" in stdout:
            return Severity.HIGH
        return Severity.MEDIUM

    def _merge_findings(
        self,
        runtime_findings: list[Finding],
        diagnosis_findings: list[Finding],
    ) -> list[Finding]:
        findings: list[Finding] = []
        seen: set[str] = set()
        for finding in [*runtime_findings, *diagnosis_findings]:
            if finding.fingerprint in seen:
                continue
            seen.add(finding.fingerprint)
            findings.append(finding)
        return findings
