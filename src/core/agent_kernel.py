from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from core.config import AgentRuntimeConfig
from core.models import (
    AgentActionType,
    AgentSession,
    CompletionReason,
    ToolCall,
    ToolPermissionSet,
)
from llm.base import BaseLLMClient
from reports.serializer import to_jsonable
from tools.agent_toolkit import AgentTool


@dataclass(slots=True)
class AgentKernelResult:
    session: AgentSession
    final_response: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class AgentKernel:
    def __init__(
        self,
        *,
        llm_client: BaseLLMClient,
        runtime_config: AgentRuntimeConfig,
    ) -> None:
        self.llm_client = llm_client
        self.runtime_config = runtime_config

    async def run_session(
        self,
        *,
        session: AgentSession,
        tools: dict[str, AgentTool],
        permissions: ToolPermissionSet,
        context: "RunContext",
    ) -> AgentKernelResult:
        started = time.monotonic()
        final_response: dict[str, object] = {}
        errors: list[str] = []
        consecutive_failures = 0
        effective_permissions = self._effective_permissions(session, permissions)
        effective_limits = self._effective_limits(session)
        logger = self._session_logger(context, session)
        activity_logging = bool(getattr(context.config, "log_agent_activity", True))

        if activity_logging:
            self._log_session_start(logger, session, effective_permissions, effective_limits)
        await context.state_store.start_agent_session(session)
        try:
            while True:
                if session.step_count >= effective_limits["max_steps"]:
                    session.completion_reason = CompletionReason.MAX_STEPS
                    break
                if session.tool_call_count >= effective_limits["max_tool_calls"]:
                    session.completion_reason = CompletionReason.MAX_TOOL_CALLS
                    break
                if time.monotonic() - started >= effective_limits["max_wall_time_seconds"]:
                    session.completion_reason = CompletionReason.MAX_WALL_TIME
                    break
                if consecutive_failures >= self.runtime_config.max_consecutive_failures:
                    session.completion_reason = CompletionReason.MAX_CONSECUTIVE_FAILURES
                    break

                available_tools = [
                    tool.spec
                    for name, tool in tools.items()
                    if effective_permissions.allows(name)
                ]
                try:
                    step = await self.llm_client.complete_agent_step(
                        session=session,
                        available_tools=available_tools,
                        skill_profile=session.active_skill,
                        candidate_skill=session.candidate_skill,
                        agent_policy=self._agent_policy_payload(
                            session,
                            effective_permissions,
                            effective_limits,
                        ),
                        skill_examples=(
                            session.active_skill.examples if session.active_skill is not None else []
                        ),
                        skill_version=session.active_skill_version,
                    )
                except TypeError:
                    step = await self.llm_client.complete_agent_step(
                        session=session,
                        available_tools=available_tools,
                    )
                session.steps.append(step)
                await context.state_store.save_agent_step(session.session_id, step)
                if activity_logging:
                    self._log_step(logger, session, step)

                if step.action_type == AgentActionType.FINALIZE:
                    final_response = dict(step.final_response)
                    session.completion_reason = CompletionReason.COMPLETED
                    if activity_logging:
                        self._log_finalize(logger, session, final_response)
                    break

                tool_name = step.tool_name or ""
                now = datetime.now(timezone.utc)
                tool_call = ToolCall(
                    step_index=step.step_index,
                    tool_name=tool_name,
                    tool_input=dict(step.tool_input),
                    status="running",
                    active_skill_version=session.active_skill_version,
                    candidate_skill_version=session.candidate_skill_version,
                    skill_profile_hash=session.skill_profile_hash,
                    started_at=now,
                    finished_at=now,
                )

                tool = tools.get(tool_name)
                if tool is None or not effective_permissions.allows(tool_name):
                    tool_call.status = "rejected"
                    tool_call.error = f"Tool is not permitted: {tool_name}"
                    tool_call.summary = tool_call.error
                    session.tool_calls.append(tool_call)
                    await context.state_store.save_tool_call(session.session_id, tool_call)
                    if activity_logging:
                        self._log_tool_result(logger, session, tool_call)
                    errors.append(tool_call.error)
                    consecutive_failures += 1
                    continue
                if tool.spec.requires_write and not effective_permissions.allow_write:
                    tool_call.status = "rejected"
                    tool_call.error = f"Write access denied for tool: {tool_name}"
                    tool_call.summary = tool_call.error
                    session.tool_calls.append(tool_call)
                    await context.state_store.save_tool_call(session.session_id, tool_call)
                    if activity_logging:
                        self._log_tool_result(logger, session, tool_call)
                    errors.append(tool_call.error)
                    consecutive_failures += 1
                    continue

                if activity_logging:
                    self._log_tool_start(logger, session, step)
                try:
                    result = await tool.invoke(session, dict(step.tool_input))
                    tool_call.output = dict(result.output)
                    tool_call.summary = result.summary
                    tool_call.error = result.error
                    tool_call.status = "succeeded" if result.ok else "failed"
                except Exception as exc:
                    tool_call.status = "failed"
                    tool_call.error = str(exc)
                    tool_call.summary = f"Tool {tool_name} raised an exception."
                    errors.append(str(exc))
                finally:
                    tool_call.finished_at = datetime.now(timezone.utc)

                session.tool_calls.append(tool_call)
                await context.state_store.save_tool_call(session.session_id, tool_call)
                if activity_logging:
                    self._log_tool_result(logger, session, tool_call)
                if tool_call.status == "succeeded":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if tool_call.error:
                        errors.append(tool_call.error)

            if session.completion_reason is None:
                session.completion_reason = CompletionReason.FAILED
        finally:
            session.finished_at = datetime.now(timezone.utc)
            await context.state_store.finish_agent_session(
                session.session_id,
                session.completion_reason or CompletionReason.FAILED,
                summary=str(final_response.get("summary", "")) if final_response else "",
                step_count=session.step_count,
                tool_call_count=session.tool_call_count,
            )
            if activity_logging:
                self._log_session_finish(logger, session, final_response, errors)

        return AgentKernelResult(
            session=session,
            final_response=final_response,
            errors=errors,
        )

    def _session_logger(self, context: "RunContext", session: AgentSession) -> logging.Logger:
        return context.logger.getChild(f"agent.{session.agent_kind.value}")

    def _log_session_start(
        self,
        logger: logging.Logger,
        session: AgentSession,
        permissions: ToolPermissionSet,
        limits: dict[str, int],
    ) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Session started: session=%s task=%s type=%s targets=%s max_steps=%s max_tool_calls=%s wall_time=%ss allow_write=%s tools=%s skill=%s candidate=%s objective=%s",
                session.session_id,
                session.task_id,
                session.task_type.value,
                len(session.targets),
                limits["max_steps"],
                limits["max_tool_calls"],
                limits["max_wall_time_seconds"],
                permissions.allow_write,
                ",".join(sorted(permissions.allowed_tools)),
                session.active_skill_version or "-",
                session.candidate_skill_version or "-",
                self._truncate_text(session.objective, 180),
            )

    def _log_step(
        self,
        logger: logging.Logger,
        session: AgentSession,
        step: "AgentStep",
    ) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Step decided: session=%s step=%s action=%s tool=%s summary=%s",
                session.session_id,
                step.step_index,
                step.action_type.value,
                step.tool_name or "-",
                self._truncate_text(step.decision_summary, 220),
            )

    def _log_tool_start(
        self,
        logger: logging.Logger,
        session: AgentSession,
        step: "AgentStep",
    ) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Tool started: session=%s step=%s tool=%s input=%s",
                session.session_id,
                step.step_index,
                step.tool_name or "-",
                self._compact_value(step.tool_input),
            )

    def _log_tool_result(
        self,
        logger: logging.Logger,
        session: AgentSession,
        tool_call: ToolCall,
    ) -> None:
        level = logging.INFO if tool_call.status == "succeeded" else logging.WARNING
        logger.log(
            level,
            "Tool finished: session=%s step=%s tool=%s status=%s summary=%s error=%s",
            session.session_id,
            tool_call.step_index,
            tool_call.tool_name,
            tool_call.status,
            self._truncate_text(tool_call.summary, 220),
            self._truncate_text(tool_call.error or "-", 220),
        )
        if logger.isEnabledFor(logging.DEBUG) and tool_call.output:
            logger.debug(
                "Tool output: session=%s step=%s tool=%s output=%s",
                session.session_id,
                tool_call.step_index,
                tool_call.tool_name,
                self._compact_value(tool_call.output, max_length=800),
            )

    def _log_finalize(
        self,
        logger: logging.Logger,
        session: AgentSession,
        final_response: dict[str, object],
    ) -> None:
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Session finalizing: session=%s summary=%s",
                session.session_id,
                self._truncate_text(str(final_response.get("summary", "")), 220),
            )

    def _log_session_finish(
        self,
        logger: logging.Logger,
        session: AgentSession,
        final_response: dict[str, object],
        errors: list[str],
    ) -> None:
        level = (
            logging.INFO
            if session.completion_reason == CompletionReason.COMPLETED
            else logging.WARNING
        )
        logger.log(
            level,
            "Session finished: session=%s reason=%s steps=%s tool_calls=%s errors=%s summary=%s",
            session.session_id,
            session.completion_reason.value if session.completion_reason else CompletionReason.FAILED.value,
            session.step_count,
            session.tool_call_count,
            len(errors),
            self._truncate_text(str(final_response.get("summary", "")), 220),
        )

    def _compact_value(self, value: object, *, max_length: int = 240) -> str:
        try:
            text = json.dumps(to_jsonable(value), ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(value)
        return self._truncate_text(text, max_length)

    def _effective_permissions(
        self,
        session: AgentSession,
        permissions: ToolPermissionSet,
    ) -> ToolPermissionSet:
        allowed_tools = set(permissions.allowed_tools)
        if session.active_skill is not None and session.active_skill.policy.allowed_tools:
            allowed_tools &= set(session.active_skill.policy.allowed_tools)
        return ToolPermissionSet(
            allowed_tools=frozenset(allowed_tools),
            allow_write=permissions.allow_write,
        )

    def _effective_limits(self, session: AgentSession) -> dict[str, int]:
        policy = session.active_skill.policy if session.active_skill is not None else None
        max_steps = self.runtime_config.max_steps
        if policy is not None and policy.recommended_max_steps is not None:
            ceiling = self.runtime_config.max_budget_ceiling or self.runtime_config.max_steps
            max_steps = max(1, min(policy.recommended_max_steps, ceiling))
        max_tool_calls = self.runtime_config.max_tool_calls
        if policy is not None and policy.recommended_max_tool_calls is not None:
            max_tool_calls = max(1, min(policy.recommended_max_tool_calls, self.runtime_config.max_tool_calls))
        max_wall_time_seconds = self.runtime_config.max_wall_time_seconds
        if policy is not None and policy.recommended_max_wall_time_seconds is not None:
            max_wall_time_seconds = max(
                30,
                min(policy.recommended_max_wall_time_seconds, self.runtime_config.max_wall_time_seconds),
            )
        return {
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
            "max_wall_time_seconds": max_wall_time_seconds,
        }

    def _agent_policy_payload(
        self,
        session: AgentSession,
        permissions: ToolPermissionSet,
        limits: dict[str, int],
    ) -> dict[str, object]:
        if session.active_skill is None:
            return {
                "allowed_tools": sorted(permissions.allowed_tools),
                "limits": limits,
            }
        return {
            "allowed_tools": sorted(permissions.allowed_tools),
            "limits": limits,
            "tool_preferences": list(session.active_skill.policy.tool_preferences),
            "planning_heuristics": list(session.active_skill.policy.planning_heuristics),
            "completion_checklist": list(session.active_skill.policy.completion_checklist),
            "noise_suppression": list(session.active_skill.policy.noise_suppression),
            "command_preferences": list(session.active_skill.policy.command_preferences),
            "severity_bias": dict(session.active_skill.policy.severity_bias),
            "rule_weights": dict(session.active_skill.policy.rule_weights),
            "environment_preferences": list(session.active_skill.policy.environment_preferences),
        }

    def _truncate_text(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return f"{value[: max_length - 3]}..."


from core.models import RunContext  # noqa: E402  # circular import guard
from core.models import AgentStep  # noqa: E402  # circular import guard
