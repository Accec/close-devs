from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.agent_kernel import AgentKernel
from core.config import AgentRuntimeConfig, AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import (
    AgentActionType,
    AgentKind,
    AgentSession,
    AgentStep,
    CompletionReason,
    RunContext,
    TaskType,
    ToolPermissionSet,
    ToolResult,
    ToolSpec,
)
from llm.base import BaseLLMClient
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentTool


class InMemoryStateStore:
    def __init__(self) -> None:
        self.session_ids: list[str] = []
        self.step_indices: list[int] = []
        self.tool_names: list[str] = []
        self.finished: list[tuple[str, CompletionReason, int, int]] = []

    async def start_agent_session(self, session: AgentSession) -> None:
        self.session_ids.append(session.session_id)

    async def save_agent_step(self, session_id: str, step: AgentStep) -> None:
        self.step_indices.append(step.step_index)

    async def save_tool_call(self, session_id: str, tool_call: object) -> None:
        self.tool_names.append(getattr(tool_call, "tool_name", ""))

    async def finish_agent_session(
        self,
        session_id: str,
        completion_reason: CompletionReason,
        summary: str,
        step_count: int,
        tool_call_count: int,
    ) -> None:
        self.finished.append((session_id, completion_reason, step_count, tool_call_count))


class RejectWriteThenFinalizeLLM(BaseLLMClient):
    async def complete_agent_step(
        self,
        *,
        session: AgentSession,
        available_tools: list[ToolSpec],
    ) -> AgentStep:
        if not session.steps:
            return AgentStep(
                step_index=0,
                decision_summary="Try to write directly.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="write_file",
                tool_input={"path": "note.txt", "content": "should not write"},
            )
        return AgentStep(
            step_index=len(session.steps),
            decision_summary="Stop after permission rejection.",
            action_type=AgentActionType.FINALIZE,
            final_response={"summary": "Rejected write was handled correctly."},
        )


class AlwaysFailingLLM(BaseLLMClient):
    async def complete_agent_step(
        self,
        *,
        session: AgentSession,
        available_tools: list[ToolSpec],
    ) -> AgentStep:
        return AgentStep(
            step_index=len(session.steps),
            decision_summary="Keep retrying the failing tool.",
            action_type=AgentActionType.TOOL_CALL,
            tool_name="failing_tool",
            tool_input={},
        )


def _build_context(tmp_path: Path) -> RunContext:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")
    return RunContext(
        run_id="run-kernel",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=InMemoryStateStore(),
        logger=logging.getLogger("test"),
        rules={},
    )


def _build_session(agent_kind: AgentKind, task_type: TaskType, repo_root: Path) -> AgentSession:
    return AgentSession(
        session_id=f"session-{agent_kind.value}",
        run_id="run-kernel",
        task_id=f"task-{agent_kind.value}",
        agent_kind=agent_kind,
        task_type=task_type,
        working_repo_root=str(repo_root),
        objective="exercise agent kernel",
        targets=[],
        payload={},
    )


@pytest.mark.asyncio
async def test_agent_kernel_rejects_write_tool_without_permission(tmp_path: Path) -> None:
    context = _build_context(tmp_path)
    session = _build_session(
        AgentKind.STATIC_REVIEW,
        TaskType.STATIC_REVIEW,
        context.working_repo_root,
    )
    tools = {
        "write_file": AgentTool(
            spec=ToolSpec(
                name="write_file",
                description="Write file content.",
                input_schema={"path": "file path", "content": "text"},
                requires_write=True,
            ),
            handler=lambda session, tool_input: (_ for _ in ()).throw(
                AssertionError("write handler should not be called")
            ),
        )
    }

    kernel = AgentKernel(
        llm_client=RejectWriteThenFinalizeLLM(),
        runtime_config=AgentRuntimeConfig(max_steps=4, max_tool_calls=4, max_consecutive_failures=2),
    )
    result = await kernel.run_session(
        session=session,
        tools=tools,
        permissions=ToolPermissionSet(allow_write=False, allowed_tools=["write_file"]),
        context=context,
    )

    assert result.session.completion_reason == CompletionReason.COMPLETED
    assert "Write access denied for tool: write_file" in result.errors
    assert result.session.tool_calls[0].status == "rejected"
    assert result.session.tool_calls[0].tool_name == "write_file"
    assert context.state_store.finished[0][1] == CompletionReason.COMPLETED


@pytest.mark.asyncio
async def test_agent_kernel_stops_after_consecutive_failures(tmp_path: Path) -> None:
    context = _build_context(tmp_path)
    session = _build_session(
        AgentKind.DYNAMIC_DEBUG,
        TaskType.DYNAMIC_DEBUG,
        context.working_repo_root,
    )

    async def failing_handler(session: AgentSession, tool_input: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool_name="failing_tool",
            ok=False,
            error="simulated failure",
            summary="tool failed",
        )

    tools = {
        "failing_tool": AgentTool(
            spec=ToolSpec(
                name="failing_tool",
                description="Always fails.",
                input_schema={},
            ),
            handler=failing_handler,
        )
    }

    kernel = AgentKernel(
        llm_client=AlwaysFailingLLM(),
        runtime_config=AgentRuntimeConfig(
            max_steps=8,
            max_tool_calls=8,
            max_consecutive_failures=2,
        ),
    )
    result = await kernel.run_session(
        session=session,
        tools=tools,
        permissions=ToolPermissionSet(allow_write=False, allowed_tools=["failing_tool"]),
        context=context,
    )

    assert result.session.completion_reason == CompletionReason.MAX_CONSECUTIVE_FAILURES
    assert len(result.session.tool_calls) == 2
    assert all(tool_call.status == "failed" for tool_call in result.session.tool_calls)
    assert context.state_store.finished[0][1] == CompletionReason.MAX_CONSECUTIVE_FAILURES


@pytest.mark.asyncio
async def test_agent_kernel_logs_session_activity(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="test.agent.static_review")
    context = _build_context(tmp_path)
    session = _build_session(
        AgentKind.STATIC_REVIEW,
        TaskType.STATIC_REVIEW,
        context.working_repo_root,
    )

    async def successful_handler(session: AgentSession, tool_input: dict[str, object]) -> ToolResult:
        return ToolResult(
            tool_name="read_file",
            ok=True,
            output={"path": "module.py"},
            summary="Read file module.py",
        )

    class SingleToolThenFinalizeLLM(BaseLLMClient):
        async def complete_agent_step(
            self,
            *,
            session: AgentSession,
            available_tools: list[ToolSpec],
        ) -> AgentStep:
            if not session.steps:
                return AgentStep(
                    step_index=0,
                    decision_summary="Inspect the target file.",
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="read_file",
                    tool_input={"path": "module.py"},
                )
            return AgentStep(
                step_index=len(session.steps),
                decision_summary="Finalize after reading the file.",
                action_type=AgentActionType.FINALIZE,
                final_response={"summary": "Static review finished."},
            )

    tools = {
        "read_file": AgentTool(
            spec=ToolSpec(
                name="read_file",
                description="Read a file.",
                input_schema={"path": "file path"},
            ),
            handler=successful_handler,
        )
    }

    kernel = AgentKernel(
        llm_client=SingleToolThenFinalizeLLM(),
        runtime_config=AgentRuntimeConfig(max_steps=4, max_tool_calls=4, max_consecutive_failures=2),
    )
    await kernel.run_session(
        session=session,
        tools=tools,
        permissions=ToolPermissionSet(allow_write=False, allowed_tools=["read_file"]),
        context=context,
    )

    messages = [record.message for record in caplog.records if record.name == "test.agent.static_review"]
    assert any("Session started:" in message for message in messages)
    assert any("Step decided:" in message for message in messages)
    assert any("Tool started:" in message for message in messages)
    assert any("Tool finished:" in message for message in messages)
    assert any("Session finished:" in message for message in messages)
