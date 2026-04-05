from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from core.agent_kernel import AgentKernel, AgentKernelResult
from core.config import AgentRuntimeConfig
from core.models import (
    AgentKind,
    AgentMessage,
    AgentResult,
    AgentSession,
    CompletionReason,
    EvidenceArtifact,
    Finding,
    FixRequest,
    RunContext,
    Severity,
    Task,
    TaskType,
    ToolPermissionSet,
)
from llm.base import BaseLLMClient


class BaseAgent(ABC):
    kind: AgentKind
    allowed_task_types: frozenset[TaskType]

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        runtime_config: AgentRuntimeConfig,
        permissions: ToolPermissionSet,
    ) -> None:
        self.llm_client = llm_client
        self.runtime_config = runtime_config
        self.permissions = permissions
        self.kernel = AgentKernel(
            llm_client=llm_client,
            runtime_config=runtime_config,
        )

    def ensure_task_type(self, task: Task) -> None:
        if task.task_type not in self.allowed_task_types:
            raise ValueError(
                f"{self.kind.value} cannot handle task type {task.task_type.value}"
            )

    def build_session(
        self,
        *,
        task: Task,
        context: RunContext,
        objective: str,
    ) -> AgentSession:
        return AgentSession(
            session_id=uuid4().hex,
            run_id=task.run_id,
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            working_repo_root=str(context.working_repo_root),
            objective=objective,
            targets=list(task.targets),
            payload=dict(task.payload),
            active_skill_version=context.active_skill.version if context.active_skill is not None else None,
            candidate_skill_version=context.candidate_skill.version if context.candidate_skill is not None else None,
            active_skill=context.active_skill,
            candidate_skill=context.candidate_skill,
            skill_profile_hash=context.active_skill.profile_hash if context.active_skill is not None else None,
            messages=[
                AgentMessage(
                    role="system",
                    content=objective,
                )
            ],
        )

    def base_artifacts(
        self,
        kernel_result: AgentKernelResult,
        *,
        handoffs: list[FixRequest] | None = None,
    ) -> dict[str, Any]:
        return {
            "session_id": kernel_result.session.session_id,
            "completion_reason": (
                kernel_result.session.completion_reason.value
                if kernel_result.session.completion_reason
                else CompletionReason.FAILED.value
            ),
            "tool_call_count": kernel_result.session.tool_call_count,
            "decision_summaries": [
                step.decision_summary for step in kernel_result.session.steps
            ],
            "active_skill_version": kernel_result.session.active_skill_version,
            "candidate_skill_version": kernel_result.session.candidate_skill_version,
            "handoffs": [self.fix_request_to_dict(item) for item in handoffs or []],
            "session_summary": {
                "step_count": kernel_result.session.step_count,
                "tool_call_count": kernel_result.session.tool_call_count,
                "completion_reason": (
                    kernel_result.session.completion_reason.value
                    if kernel_result.session.completion_reason
                    else CompletionReason.FAILED.value
                ),
                "budget_exhausted": (
                    kernel_result.session.completion_reason
                    in {
                        CompletionReason.MAX_STEPS,
                        CompletionReason.MAX_TOOL_CALLS,
                        CompletionReason.MAX_WALL_TIME,
                        CompletionReason.MAX_CONSECUTIVE_FAILURES,
                    }
                ),
            },
        }

    def finding_from_dict(
        self,
        data: dict[str, Any],
        *,
        default_source_agent: AgentKind,
    ) -> Finding:
        severity_raw = str(data.get("severity", Severity.LOW.value)).lower()
        severity = Severity(severity_raw) if severity_raw in {item.value for item in Severity} else Severity.LOW
        source_raw = str(data.get("source_agent", default_source_agent.value))
        source_agent = AgentKind(source_raw) if source_raw in {item.value for item in AgentKind} else default_source_agent
        return Finding(
            source_agent=source_agent,
            severity=severity,
            rule_id=str(data.get("rule_id", "agent-observation")),
            message=str(data.get("message", "")),
            category=str(data.get("category", "analysis")),
            root_cause_class=(
                str(data["root_cause_class"])
                if data.get("root_cause_class") not in (None, "")
                else None
            ),
            path=str(data["path"]) if data.get("path") else None,
            line=self._coerce_line(data.get("line")),
            symbol=str(data["symbol"]) if data.get("symbol") else None,
            evidence=self._coerce_mapping(data.get("evidence")),
            fingerprint=str(data.get("fingerprint", "")),
            state=str(data.get("state", "open")),
        )

    def fix_request_from_dict(
        self,
        data: dict[str, Any],
        *,
        default_source_agent: AgentKind,
    ) -> FixRequest:
        severity_raw = str(data.get("severity", Severity.LOW.value)).lower()
        severity = Severity(severity_raw) if severity_raw in {item.value for item in Severity} else Severity.LOW
        source_raw = str(data.get("source_agent", default_source_agent.value))
        source_agent = AgentKind(source_raw) if source_raw in {item.value for item in AgentKind} else default_source_agent
        evidence = [
            EvidenceArtifact(
                kind=str(item.get("kind", "evidence")),
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                path=str(item["path"]) if item.get("path") else None,
                data=self._coerce_mapping(item.get("data")),
            )
            for item in data.get("evidence", [])
            if isinstance(item, dict)
        ]
        return FixRequest(
            source_agent=source_agent,
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            recommended_change=str(data.get("recommended_change", "")),
            severity=severity,
            kind=str(data.get("kind", "code") or "code"),
            confidence=self._coerce_confidence(data.get("confidence")),
            affected_files=[str(item) for item in data.get("affected_files", [])],
            evidence=evidence,
            metadata=self._coerce_mapping(data.get("metadata")),
        )

    def fix_request_to_dict(self, request: FixRequest) -> dict[str, Any]:
        return {
            "source_agent": request.source_agent.value,
            "title": request.title,
            "description": request.description,
            "recommended_change": request.recommended_change,
            "severity": request.severity.value,
            "kind": request.kind,
            "confidence": request.confidence,
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

    @abstractmethod
    async def run(self, task: Task, context: RunContext) -> AgentResult:
        raise NotImplementedError

    def _coerce_mapping(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        if isinstance(value, list):
            return {"items": [self._coerce_json_value(item) for item in value]}
        return {"value": self._coerce_json_value(value)}

    def _coerce_json_value(self, value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, dict):
            return {str(key): self._coerce_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._coerce_json_value(item) for item in value]
        return str(value)

    def _coerce_line(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _coerce_confidence(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            try:
                return max(0.0, min(1.0, float(value)))
            except ValueError:
                return 0.5
        return 0.5
