from __future__ import annotations

from pathlib import Path
import re

from agents.base import BaseAgent
from core.agent_kernel import AgentKernelResult
from core.config import AgentRuntimeConfig
from core.models import AgentKind, AgentResult, CompletionReason, EvidenceArtifact, Finding, FixRequest, RunContext, Severity, Task, TaskStatus, TaskType, ToolPermissionSet
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
        runtime_findings = self._runtime_findings(
            kernel_result,
            repo_root=Path(context.working_repo_root),
            startup_topology=context.startup_topology,
        )
        diagnosis_findings = [
            self.finding_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("findings", [])
            if isinstance(item, dict)
        ]
        findings = self._merge_findings(runtime_findings, diagnosis_findings)
        explicit_handoffs = [
            self.fix_request_from_dict(item, default_source_agent=self.kind)
            for item in kernel_result.final_response.get("fix_requests", [])
            if isinstance(item, dict)
        ]
        derived_handoffs = self._derive_fix_requests(runtime_findings, kernel_result)
        handoffs = self._merge_handoffs(explicit_handoffs, derived_handoffs)
        artifacts = self.base_artifacts(kernel_result, handoffs=handoffs)
        artifacts["commands"] = [
            call.output
            for call in kernel_result.session.tool_calls
            if call.tool_name == "run_test_command"
        ]
        artifacts["diagnosis_findings"] = len(diagnosis_findings)
        artifacts["investigation_depth"] = {
            "commands_executed": len(
                [call for call in kernel_result.session.tool_calls if call.tool_name == "run_test_command"]
            ),
            "tracebacks_parsed": len(
                [call for call in kernel_result.session.tool_calls if call.tool_name == "parse_traceback"]
            ),
            "root_cause_classes": sorted(
                {
                    finding.root_cause_class
                    for finding in findings
                    if finding.root_cause_class
                }
            ),
        }
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

    def _runtime_findings(
        self,
        kernel_result: AgentKernelResult,
        *,
        repo_root: Path,
        startup_topology=None,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for call in kernel_result.session.tool_calls:
            if call.tool_name != "run_test_command" or call.status != "succeeded":
                continue
            root_cause_class = self._classify_runtime_output(
                call.output,
                repo_root=repo_root,
            )
            evidence = self._runtime_evidence(
                call.output,
                root_cause_class=root_cause_class,
                startup_topology=startup_topology,
            )
            module_path = self._local_module_path(
                self._extract_missing_module_from_text(str(evidence.get("combined_excerpt", ""))),
                repo_root,
            )
            if module_path is not None:
                evidence["module_path"] = module_path
            if call.output.get("timed_out"):
                findings.append(
                    Finding(
                        source_agent=self.kind,
                        severity=self._severity_from_output(call.output, timeout=True),
                        rule_id="command-timeout",
                        message=f"Command timed out: {call.output.get('command', '')}",
                        category="runtime",
                        root_cause_class=root_cause_class,
                        evidence=evidence,
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
                        category=self._finding_category(root_cause_class),
                        root_cause_class=root_cause_class,
                        evidence=evidence,
                    )
                )
        return findings

    def _derive_fix_requests(
        self,
        findings: list[Finding],
        kernel_result: AgentKernelResult,
    ) -> list[FixRequest]:
        requests: list[FixRequest] = []
        seen: set[tuple[str, str]] = set()
        latest_command = next(
            (
                call.output
                for call in reversed(kernel_result.session.tool_calls)
                if call.tool_name == "run_test_command"
            ),
            {},
        )
        for finding in findings:
            if finding.severity not in {Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM}:
                continue
            key = (finding.root_cause_class or "runtime", finding.rule_id)
            if key in seen:
                continue
            seen.add(key)
            requests.append(
                FixRequest(
                    source_agent=self.kind,
                    title=self._handoff_title(finding),
                    description=finding.message,
                    recommended_change=self._recommended_change(finding),
                    severity=finding.severity,
                    kind=self._fix_request_kind(finding.root_cause_class),
                    confidence=self._confidence_for_finding(finding),
                    affected_files=self._affected_files_from_evidence(finding),
                    evidence=[
                        EvidenceArtifact(
                            kind="runtime-output",
                            title=finding.rule_id,
                            summary=finding.message,
                            path=None,
                            data={
                                "root_cause_class": finding.root_cause_class,
                                "command": latest_command.get("command", ""),
                                "stderr_excerpt": str(finding.evidence.get("stderr_excerpt", "")),
                                "startup_context": str(finding.evidence.get("startup_context", "")),
                                "matched_entrypoint": str(finding.evidence.get("matched_entrypoint", "")),
                                "matched_config_anchor": str(finding.evidence.get("matched_config_anchor", "")),
                                "repair_hint": str(finding.evidence.get("repair_hint", "")),
                            },
                        )
                    ],
                    metadata={
                        "rule_id": finding.rule_id,
                        "root_cause_class": finding.root_cause_class,
                        "startup_context": str(finding.evidence.get("startup_context", "")) or None,
                        "entrypoint_path": str(finding.evidence.get("matched_entrypoint", "")) or None,
                        "config_anchor_path": str(finding.evidence.get("matched_config_anchor", "")) or None,
                        "repair_hint": str(finding.evidence.get("repair_hint", "")) or None,
                    },
                )
            )
        return requests[:6]

    def _merge_handoffs(
        self,
        explicit_handoffs: list[FixRequest],
        derived_handoffs: list[FixRequest],
    ) -> list[FixRequest]:
        merged: list[FixRequest] = []
        seen: set[tuple[str, str, str]] = set()
        for request in [*explicit_handoffs, *derived_handoffs]:
            key = (
                request.kind,
                request.title,
                request.recommended_change,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(request)
        return merged

    def _severity_from_output(self, output: dict[str, object], *, timeout: bool) -> Severity:
        if timeout:
            return Severity.HIGH
        stderr = str(output.get("stderr", ""))
        stdout = str(output.get("stdout", ""))
        if "Traceback" in stderr or "Traceback" in stdout:
            return Severity.HIGH
        return Severity.MEDIUM

    def _classify_runtime_output(
        self,
        output: dict[str, object],
        *,
        repo_root: Path,
    ) -> str:
        if output.get("timed_out"):
            return "environment"
        text = "\n".join([str(output.get("stdout", "")), str(output.get("stderr", ""))])
        lowered = text.lower()
        missing_module = self._extract_missing_module_from_text(text)
        asgi_module = self._extract_asgi_module_from_text(text)
        if self._is_missing_env_text(text):
            return "config"
        if self._is_startup_argument_mismatch(lowered):
            return "startup"
        if asgi_module and self._local_module_path(asgi_module, repo_root) is not None:
            return "startup"
        if missing_module:
            if self._local_module_path(missing_module, repo_root) is not None:
                return "startup"
            return "dependency"
        if "importerror while importing test module" in lowered and repo_root.joinpath("src").exists():
            return "startup"
        if "fixture" in lowered and "not found" in lowered:
            return "test"
        if any(marker in lowered for marker in ("permission denied", "connection refused", "address already in use")):
            return "environment"
        return "application"

    def _finding_category(self, root_cause_class: str) -> str:
        mapping = {
            "dependency": "dependency",
            "config": "config",
            "startup": "runtime",
            "test": "test",
            "environment": "runtime",
            "application": "runtime",
        }
        return mapping.get(root_cause_class, "runtime")

    def _runtime_evidence(
        self,
        output: dict[str, object],
        *,
        root_cause_class: str = "application",
        startup_topology=None,
    ) -> dict[str, object]:
        combined = "\n".join([str(output.get("stdout", "")), str(output.get("stderr", ""))]).strip()
        evidence = {
            "command": output.get("command", ""),
            "returncode": output.get("returncode", 0),
            "timed_out": output.get("timed_out", False),
            "stdout_excerpt": str(output.get("stdout", ""))[:600],
            "stderr_excerpt": str(output.get("stderr", ""))[:600],
            "combined_excerpt": combined[:800],
            "startup_context": self._detect_startup_context(
                "\n".join([str(output.get("command", "")), combined])
            ),
        }
        if startup_topology is not None:
            evidence.update(
                self.toolkit_factory.match_runtime_to_startup_topology(
                    {
                        "command": str(output.get("command", "")),
                        "stdout": str(output.get("stdout", "")),
                        "stderr": str(output.get("stderr", "")),
                    },
                    topology=startup_topology,
                    root_cause_class=root_cause_class,
                )
            )
        return evidence

    def _handoff_title(self, finding: Finding) -> str:
        root_cause_class = finding.root_cause_class or "runtime"
        return f"Address {root_cause_class} blocker"

    def _recommended_change(self, finding: Finding) -> str:
        root_cause_class = finding.root_cause_class or "runtime"
        module_name = self._extract_missing_module_name(finding)
        env_var = self._extract_missing_env_var(finding)
        if root_cause_class == "dependency" and module_name:
            return f"Declare and install the package providing `{module_name}` before rerunning tests."
        if root_cause_class == "config" and env_var:
            return f"Document and supply the required environment variable `{env_var}` in the runtime configuration."
        if root_cause_class == "startup":
            startup_context = self._startup_context(finding)
            if module_name:
                if startup_context == "uvicorn":
                    return (
                        f"Repair the Python startup/bootstrap path so `{module_name}` resolves for the uvicorn "
                        "app import path before rerunning validation."
                    )
                if startup_context == "celery":
                    return (
                        f"Repair the Python startup/bootstrap path so `{module_name}` resolves for the celery "
                        "worker app import before rerunning validation."
                    )
                return (
                    f"Repair the Python startup/bootstrap path so `{module_name}` resolves from the repository layout "
                    "before rerunning validation."
                )
            if self._is_startup_argument_mismatch(self._finding_text(finding).lower()):
                return (
                    "Review the startup or test command contract: required CLI arguments are missing or mismatched "
                    "and should be fixed manually."
                )
            return "Repair the Python startup/bootstrap configuration before rerunning validation."
        if root_cause_class == "test":
            return "Fix the failing test setup or fixture contract before continuing runtime validation."
        if root_cause_class == "environment":
            return "Stabilize the execution environment or runtime dependencies before rerunning the reproduction command."
        return "Inspect the failing runtime path, reproduce locally, and patch the underlying application defect."

    def _fix_request_kind(self, root_cause_class: str | None) -> str:
        if root_cause_class in {"dependency", "config", "startup", "runtime", "application"}:
            return "runtime" if root_cause_class == "application" else root_cause_class
        return "runtime"

    def _confidence_for_finding(self, finding: Finding) -> float:
        if finding.root_cause_class == "dependency" and self._extract_missing_module_name(finding):
            return 0.95
        if finding.root_cause_class == "config" and self._extract_missing_env_var(finding):
            return 0.9
        if finding.root_cause_class == "startup":
            if self._extract_missing_module_name(finding):
                return 0.9
            if self._is_startup_argument_mismatch(self._finding_text(finding).lower()):
                return 0.7
            return 0.8
        if finding.root_cause_class == "test":
            return 0.8
        return 0.75

    def _affected_files_from_evidence(self, finding: Finding) -> list[str]:
        candidates: list[str] = []
        for key in ("path", "module_path", "matched_entrypoint", "matched_config_anchor"):
            value = finding.evidence.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
        return sorted(dict.fromkeys(candidates))

    def _extract_missing_module_name(self, finding: Finding) -> str | None:
        return self._extract_missing_module_from_text(self._finding_text(finding))

    def _extract_missing_env_var(self, finding: Finding) -> str | None:
        text = self._finding_text(finding)
        for pattern in (
            r"KeyError:\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
            r"Missing environment variable[:\s]+([A-Z][A-Z0-9_]+)",
            r"environment variable\s+([A-Z][A-Z0-9_]+)\s+is required",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _finding_text(self, finding: Finding) -> str:
        return "\n".join(
            [
                finding.message,
                str(finding.evidence.get("stderr_excerpt", "")),
                str(finding.evidence.get("combined_excerpt", "")),
            ]
        )

    def _startup_context(self, finding: Finding) -> str:
        value = finding.evidence.get("startup_context")
        return value if isinstance(value, str) and value else "generic"

    def _extract_missing_module_from_text(self, text: str) -> str | None:
        match = re.search(
            r"(?:No module named|ModuleNotFoundError:\s*No module named)\s+['\"]([^'\"]+)['\"]",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _extract_asgi_module_from_text(self, text: str) -> str | None:
        match = re.search(
            r"could not import module\s+['\"]([^'\"]+)['\"]",
            text,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _is_missing_env_text(self, text: str) -> bool:
        lowered_text = text.lower()
        return (
            "missing environment variable" in lowered_text
            or "environment variable" in lowered_text and "is required" in lowered_text
            or bool(re.search(r"KeyError:\s*['\"][A-Z][A-Z0-9_]+['\"]", text))
            or bool(re.search(r"(?m)^([A-Z][A-Z0-9_]+)\s*$\n\s+Field required\b", text))
        )

    def _is_startup_argument_mismatch(self, lowered_text: str) -> bool:
        argument_markers = (
            "unrecognized arguments:",
            "the following arguments are required:",
            "error: argument",
        )
        return any(marker in lowered_text for marker in argument_markers)

    def _detect_startup_context(self, text: str) -> str:
        lowered = text.lower()
        if "manage.py" in lowered or "django.core.management" in lowered:
            return "django_manage"
        if "asgi.py" in lowered:
            return "asgi"
        if "wsgi.py" in lowered:
            return "wsgi"
        if "uvicorn" in lowered or "asgi app" in lowered:
            return "uvicorn"
        if "gunicorn" in lowered:
            return "gunicorn"
        if "alembic" in lowered or "env.py" in lowered:
            return "alembic"
        if "celery" in lowered:
            return "celery"
        if "pytest" in lowered or "importerror while importing test module" in lowered:
            return "pytest"
        return "generic"

    def _local_module_path(self, module_name: str | None, repo_root: Path) -> str | None:
        if not module_name:
            return None
        top_level = module_name.split(".", 1)[0]
        candidates = [
            repo_root / "src" / top_level / "__init__.py",
            repo_root / "src" / f"{top_level}.py",
            repo_root / top_level / "__init__.py",
            repo_root / f"{top_level}.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.relative_to(repo_root).as_posix()
        return None

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
