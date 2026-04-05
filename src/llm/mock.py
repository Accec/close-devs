from __future__ import annotations

import ast
from collections import Counter
from typing import Any

from core.models import (
    AgentActionType,
    AgentKind,
    AgentSession,
    AgentStep,
    EvidenceArtifact,
    FixRequest,
    Finding,
    Severity,
    ToolSpec,
)
from llm.base import BaseLLMClient


class MockLLMClient(BaseLLMClient):
    provider_name = "mock"

    async def complete_agent_step(
        self,
        *,
        session: AgentSession,
        available_tools: list[ToolSpec],
        skill_profile=None,
        candidate_skill=None,
        agent_policy=None,
        skill_examples=None,
        skill_version=None,
    ) -> AgentStep:
        if session.agent_kind == AgentKind.STATIC_REVIEW:
            return self._complete_static_step(session, skill_profile=skill_profile)
        if session.agent_kind == AgentKind.DYNAMIC_DEBUG:
            return self._complete_dynamic_step(session, skill_profile=skill_profile)
        return self._complete_maintenance_step(session, skill_profile=skill_profile)

    def _complete_static_step(self, session: AgentSession, *, skill_profile=None) -> AgentStep:
        step_index = session.step_count + 1
        tool_names = [call.tool_name for call in session.tool_calls]
        static_context_value = session.payload.get("static_context", {})
        if hasattr(static_context_value, "top_targets"):
            top_targets = list(getattr(static_context_value, "top_targets", []))
        elif isinstance(static_context_value, dict):
            top_targets = [
                str(item)
                for item in static_context_value.get("top_targets", [])
                if isinstance(item, str)
            ]
        else:
            top_targets = []
        session_targets = [
            str(item) for item in (top_targets or session.targets) if isinstance(item, str)
        ] or list(session.targets)
        python_targets = [target for target in session_targets if target.endswith(".py")]
        source_targets = [
            target
            for target in session_targets
            if target.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".kts"))
        ]
        preferred_tools = list(getattr(getattr(skill_profile, "policy", None), "tool_preferences", []))

        if "search_repo" in preferred_tools and "search_repo" not in tool_names:
            return AgentStep(
                step_index=step_index,
                decision_summary="Start with a repository-wide search for high-signal markers before the deterministic scan.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="search_repo",
                tool_input={"query": "TODO|FIXME|except:|except Exception", "paths": session_targets},
            )

        if "run_static_review" not in tool_names:
            return AgentStep(
                step_index=step_index,
                decision_summary="Run baseline static analysis before semantic review.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="run_static_review",
                tool_input={"paths": session_targets},
            )

        read_paths = {call.tool_input.get("path") for call in session.tool_calls if call.tool_name == "read_file"}
        for path in session_targets[:3]:
            if path not in read_paths:
                return AgentStep(
                    step_index=step_index,
                    decision_summary=f"Inspect target file {path} directly for semantic issues.",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="read_file",
                    tool_input={"path": path},
                )

        summarized_paths = {
            call.tool_input.get("path")
            for call in session.tool_calls
            if call.tool_name == "ast_summary"
        }
        for path in (python_targets[:2] or source_targets[:2]):
            if path not in summarized_paths:
                return AgentStep(
                    step_index=step_index,
                    decision_summary=f"Summarize AST structure for {path}.",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="ast_summary",
                    tool_input={"path": path},
                )

        if "search_repo" not in tool_names:
            return AgentStep(
                step_index=step_index,
                decision_summary="Search for TODO/FIXME markers across current targets.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="search_repo",
                tool_input={"query": "TODO|FIXME", "paths": session_targets},
            )

        deterministic_findings = self._deterministic_static_findings_from_session(session)
        semantic_findings = self._static_semantic_findings_from_session(session)
        semantic_high_value = [
            finding for finding in semantic_findings if self._is_high_value_finding(finding)
        ]
        deterministic_high_value = [
            finding for finding in deterministic_findings if self._is_high_value_finding(finding)
        ]
        fix_requests = self._build_fix_requests(
            AgentKind.STATIC_REVIEW,
            semantic_high_value or deterministic_high_value,
        )
        if semantic_high_value or deterministic_high_value:
            summary = self._static_high_value_summary(
                session,
                semantic_high_value=semantic_high_value,
                deterministic_high_value=deterministic_high_value,
            )
        else:
            summary = (
                f"Autonomous static review inspected {len(session_targets)} targets and produced "
                f"{len(semantic_findings)} semantic findings."
            )
        return AgentStep(
            step_index=step_index,
            decision_summary="Finalize static review after collecting deterministic and semantic evidence.",
            action_type=AgentActionType.FINALIZE,
            final_response={
                "summary": summary,
                "findings": [self._finding_to_dict(item) for item in semantic_findings],
                "fix_requests": [self._fix_request_to_dict(item) for item in fix_requests],
            },
        )

    def _complete_dynamic_step(self, session: AgentSession, *, skill_profile=None) -> AgentStep:
        step_index = session.step_count + 1
        command_calls = [call for call in session.tool_calls if call.tool_name == "run_test_command"]
        commands = list(session.payload.get("commands", []))
        executed_commands = [str(call.tool_input.get("command", "")) for call in command_calls]
        preferred_tools = list(getattr(getattr(skill_profile, "policy", None), "tool_preferences", []))

        for command in commands:
            if command not in executed_commands:
                return AgentStep(
                    step_index=step_index,
                    decision_summary=f"Execute runtime reproduction command: {command}",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="run_test_command",
                    tool_input={"command": command},
                )

        parsed_inputs = {
            call.tool_input.get("text")
            for call in session.tool_calls
            if call.tool_name == "parse_traceback"
        }
        for call in command_calls:
            output = call.output
            if int(output.get("returncode", 0) or 0) == 0 or output.get("timed_out"):
                continue
            combined = "\n".join(
                [str(output.get("stdout", "")), str(output.get("stderr", ""))]
            ).strip()
            if combined and combined not in parsed_inputs and (
                "parse_traceback" in preferred_tools or preferred_tools == []
            ):
                return AgentStep(
                    step_index=step_index,
                    decision_summary="Parse traceback details from failing runtime output.",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="parse_traceback",
                    tool_input={"text": combined},
                )

        diagnosis_findings = self._dynamic_findings_from_session(session)
        fix_requests = self._build_fix_requests(AgentKind.DYNAMIC_DEBUG, diagnosis_findings)
        summary = (
            f"Autonomous dynamic debugging executed {len(command_calls)} commands and produced "
            f"{len(diagnosis_findings)} additional fix requests."
        )
        suggestions = self._dynamic_suggestions_from_session(session)
        return AgentStep(
            step_index=step_index,
            decision_summary="Finalize runtime diagnosis after reproducing failures and parsing evidence.",
            action_type=AgentActionType.FINALIZE,
            final_response={
                "summary": summary,
                "findings": [self._finding_to_dict(item) for item in diagnosis_findings],
                "fix_requests": [self._fix_request_to_dict(item) for item in fix_requests],
                "suggestions": suggestions,
            },
        )

    def _complete_maintenance_step(self, session: AgentSession, *, skill_profile=None) -> AgentStep:
        step_index = session.step_count + 1
        targets = session.targets or list(session.payload.get("touched_files", []))
        read_paths = {call.tool_input.get("path") for call in session.tool_calls if call.tool_name == "read_file"}
        for path in targets[:3]:
            if path and path not in read_paths:
                return AgentStep(
                    step_index=step_index,
                    decision_summary=f"Inspect file {path} before preparing a maintenance patch.",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="read_file",
                    tool_input={"path": path},
                )

        if "prepare_safe_patch" not in [call.tool_name for call in session.tool_calls]:
            return AgentStep(
                step_index=step_index,
                decision_summary="Generate the current best safe patch candidate from collected handoffs.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="prepare_safe_patch",
                tool_input={"paths": targets},
            )

        latest_patch = next(
            (call.output for call in reversed(session.tool_calls) if call.tool_name == "prepare_safe_patch"),
            {},
        )
        counts = Counter(
            handoff.get("severity", Severity.LOW.value)
            for handoff in session.payload.get("handoffs", [])
            if isinstance(handoff, dict)
        )
        summary = latest_patch.get(
            "summary",
            "Maintenance agent completed autonomous patch planning.",
        )
        rationale = "Agent-derived maintenance rationale: " + ", ".join(
            f"{severity}={count}" for severity, count in sorted(counts.items())
        ) if counts else "Agent-derived maintenance rationale: no handoff severity distribution available."
        suggestions = list(latest_patch.get("suggestions", []))
        if not suggestions:
            suggestions.append("No additional manual follow-up required.")
        return AgentStep(
            step_index=step_index,
            decision_summary="Finalize maintenance with the latest safe patch candidate.",
            action_type=AgentActionType.FINALIZE,
            final_response={
                "summary": summary,
                "rationale": rationale,
                "suggestions": suggestions,
                "use_latest_patch": True,
            },
        )

    def _static_semantic_findings_from_session(self, session: AgentSession) -> list[Finding]:
        findings: list[Finding] = []
        seen: set[str] = set()
        for call in session.tool_calls:
            if call.tool_name != "read_file":
                continue
            path = str(call.output.get("path") or call.tool_input.get("path") or "")
            content = str(call.output.get("content") or "")
            for finding in self._comment_findings(path, content):
                if finding.fingerprint in seen:
                    continue
                seen.add(finding.fingerprint)
                findings.append(finding)
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content or "\n")
            except SyntaxError:
                continue
            for finding in self._python_semantic_findings(path, tree):
                if finding.fingerprint in seen:
                    continue
                seen.add(finding.fingerprint)
                findings.append(finding)
        return findings

    def _dynamic_findings_from_session(self, session: AgentSession) -> list[Finding]:
        findings: list[Finding] = []
        seen: set[str] = set()
        for call in session.tool_calls:
            if call.tool_name == "parse_traceback":
                for item in call.output.get("tracebacks", []):
                    if not isinstance(item, dict):
                        continue
                    finding = Finding(
                        source_agent=AgentKind.DYNAMIC_DEBUG,
                        severity=Severity.HIGH,
                        rule_id="runtime-root-cause",
                        message=(
                            f"Likely runtime root cause is {item.get('exception_type', 'Exception')}: "
                            f"{item.get('message', '')}"
                        ).strip(),
                        category="runtime",
                        root_cause_class=self._runtime_root_cause_class(
                            f"{item.get('exception_type', '')}: {item.get('message', '')}"
                        ),
                        evidence={"traceback": item.get("text", "")},
                    )
                    if finding.fingerprint in seen:
                        continue
                    seen.add(finding.fingerprint)
                    findings.append(finding)
        return findings

    def _dynamic_suggestions_from_session(self, session: AgentSession) -> list[str]:
        suggestions: list[str] = []
        for call in session.tool_calls:
            if call.tool_name != "run_test_command":
                continue
            if call.output.get("timed_out"):
                suggestions.append("Narrow the repro command or increase timeout for long-running dynamic analysis.")
            elif int(call.output.get("returncode", 0) or 0) != 0:
                suggestions.append(
                    f"Inspect command `{call.output.get('command', call.tool_input.get('command', 'unknown'))}` first because it reproduced the failure."
                )
        if not suggestions:
            suggestions.append("Executed runtime checks completed without actionable failures.")
        return suggestions

    def _deterministic_static_findings_from_session(self, session: AgentSession) -> list[Finding]:
        findings: list[Finding] = []
        for call in session.tool_calls:
            if call.tool_name != "run_static_review":
                continue
            for item in call.output.get("findings", []):
                if not isinstance(item, dict):
                    continue
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity(str(item.get("severity", Severity.LOW.value)).lower())
                        if str(item.get("severity", Severity.LOW.value)).lower() in {level.value for level in Severity}
                        else Severity.LOW,
                        rule_id=str(item.get("rule_id", "static-tooling-finding")),
                        message=str(item.get("message", "")),
                        category=str(item.get("category", "analysis")),
                        path=str(item["path"]) if item.get("path") else None,
                        line=int(item["line"]) if isinstance(item.get("line"), int) else None,
                        symbol=str(item["symbol"]) if item.get("symbol") else None,
                        evidence=dict(item.get("evidence", {})) if isinstance(item.get("evidence"), dict) else {},
                    )
                )
        return findings

    def _is_high_value_finding(self, finding: Finding) -> bool:
        if finding.rule_id in {
            "missing-module-docstring",
            "missing-final-newline",
            "trailing-whitespace",
            "excessive-eof-blank-lines",
        }:
            return False
        if finding.category in {"correctness", "architecture", "security", "dependency", "runtime", "typing"}:
            return True
        return finding.severity in {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}

    def _static_high_value_summary(
        self,
        session: AgentSession,
        *,
        semantic_high_value: list[Finding],
        deterministic_high_value: list[Finding],
    ) -> str:
        primary = (semantic_high_value or deterministic_high_value)[0]
        if primary.path and "bootstrap" in primary.path:
            return (
                "Static review found mostly low-value documentation lint, but one higher-value "
                "correctness/operability issue stands out in bootstrap initialization."
            )
        return (
            f"Autonomous static review identified a higher-value {primary.category} issue in "
            f"{primary.path or 'the inspected code'} beyond low-priority documentation noise."
        )

    def _build_fix_requests(
        self,
        source_agent: AgentKind,
        findings: list[Finding],
    ) -> list[FixRequest]:
        requests: list[FixRequest] = []
        for finding in findings[:8]:
            evidence = [
                EvidenceArtifact(
                    kind="finding",
                    title=finding.rule_id,
                    summary=finding.message,
                    path=finding.path,
                    data={"line": finding.line, "rule_id": finding.rule_id},
                )
            ]
            requests.append(
                FixRequest(
                    source_agent=source_agent,
                    title=f"Address {finding.rule_id}",
                    description=finding.message,
                    recommended_change=f"Update {finding.path or 'the affected area'} to resolve {finding.rule_id}.",
                    severity=finding.severity,
                    kind=self._fix_request_kind(finding),
                    confidence=self._fix_request_confidence(finding),
                    affected_files=[finding.path] if finding.path else [],
                    evidence=evidence,
                    metadata={
                        "rule_id": finding.rule_id,
                        "root_cause_class": finding.root_cause_class,
                    },
                )
            )
        return requests

    def _fix_request_kind(self, finding: Finding) -> str:
        if finding.root_cause_class in {"dependency", "config", "runtime"}:
            return finding.root_cause_class
        if finding.category in {"dependency", "config"}:
            return finding.category
        return "code"

    def _fix_request_confidence(self, finding: Finding) -> float:
        if finding.root_cause_class == "dependency":
            return 0.95
        if finding.root_cause_class == "config":
            return 0.9
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}:
            return 0.85
        return 0.7

    def _runtime_root_cause_class(self, text: str) -> str:
        lowered = text.lower()
        if "no module named" in lowered or "modulenotfounderror" in lowered:
            return "dependency"
        if "keyerror" in lowered or "environment variable" in lowered:
            return "config"
        return "runtime"

    def _comment_findings(self, path: str, content: str) -> list[Finding]:
        findings: list[Finding] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            normalized = line.strip().lower()
            if "todo" in normalized or "fixme" in normalized:
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="todo-comment",
                        message="Code contains a TODO/FIXME marker that should be tracked or resolved.",
                        category="maintainability",
                        path=path,
                        line=line_number,
                    )
                )
                break
        return findings

    def _python_semantic_findings(self, path: str, tree: ast.AST) -> list[Finding]:
        findings: list[Finding] = []
        is_test_file = "/tests/" in f"/{path}" or path.startswith("tests/") or path.endswith("_test.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None:
                        findings.append(
                            Finding(
                                source_agent=AgentKind.STATIC_REVIEW,
                                severity=Severity.MEDIUM,
                                rule_id="bare-except",
                                message="Bare except hides the actual failure mode and makes maintenance harder.",
                                category="correctness",
                                path=path,
                                line=handler.lineno,
                            )
                        )
                        break
            if (
                isinstance(node, ast.Call)
                and not is_test_file
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                findings.append(
                    Finding(
                        source_agent=AgentKind.STATIC_REVIEW,
                        severity=Severity.LOW,
                        rule_id="debug-print-statement",
                        message="Non-test module contains print() debugging output.",
                        category="maintainability",
                        path=path,
                        line=node.lineno,
                    )
                )
                break
        return findings

    def _finding_to_dict(self, finding: Finding) -> dict[str, Any]:
        return {
            "source_agent": finding.source_agent.value,
            "severity": finding.severity.value,
            "rule_id": finding.rule_id,
            "message": finding.message,
            "category": finding.category,
            "path": finding.path,
            "line": finding.line,
            "symbol": finding.symbol,
            "evidence": finding.evidence,
            "fingerprint": finding.fingerprint,
            "state": finding.state,
        }

    def _fix_request_to_dict(self, request: FixRequest) -> dict[str, Any]:
        return {
            "source_agent": request.source_agent.value,
            "title": request.title,
            "description": request.description,
            "recommended_change": request.recommended_change,
            "severity": request.severity.value,
            "affected_files": request.affected_files,
            "evidence": [
                {
                    "kind": item.kind,
                    "title": item.title,
                    "summary": item.summary,
                    "path": item.path,
                    "data": item.data,
                }
                for item in request.evidence
            ],
            "metadata": request.metadata,
        }
