from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.config import LLMConfig
from core.models import (
    AgentActionType,
    AgentKind,
    AgentSession,
    ChangeSet,
    FeedbackBundle,
    RepoSnapshot,
    TaskType,
    ToolSpec,
)
from llm.mock import MockLLMClient
from llm.openai_compatible import OpenAICompatibleLLMClient


class StubChatClient:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        prompt = getattr(messages[-1], "content", "")
        self.last_prompt = str(prompt)
        return SimpleNamespace(
            content=(
                '{"decision_summary":"Finalize maintenance step.",'
                '"action_type":"finalize",'
                '"tool_name":null,'
                '"tool_input":{},'
                '"final_response":{"summary":"maintenance-ready"}}'
            )
        )


@pytest.mark.asyncio
async def test_openai_compatible_client_serializes_feedback_bundle_payload() -> None:
    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.config = LLMConfig(provider="openai_compatible", model="gpt-5.4")
    client.client = StubChatClient()
    client.fallback = MockLLMClient()

    snapshot = RepoSnapshot(
        repo_root="/tmp/repo",
        scanned_at=datetime.now(timezone.utc),
        revision="abc123",
        file_hashes={"module.py": "deadbeef"},
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["module.py"],
            added_files=[],
            removed_files=[],
            baseline_revision="base123",
            current_revision="abc123",
        ),
    )
    session = AgentSession(
        session_id="session-maintenance",
        run_id="run-1",
        task_id="task-1",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        working_repo_root="/tmp/repo",
        objective="prepare maintenance patch",
        targets=["module.py"],
        payload={"feedback": feedback, "handoffs": []},
    )

    step = await client.complete_agent_step(
        session=session,
        available_tools=[
            ToolSpec(
                name="prepare_safe_patch",
                description="Build a safe patch candidate.",
                input_schema={"feedback": "serialized feedback bundle"},
            )
        ],
    )

    assert step.action_type == AgentActionType.FINALIZE
    assert step.final_response["summary"] == "maintenance-ready"
    assert client.client.last_prompt is not None
    assert '"payload": {' in client.client.last_prompt
    assert '"feedback": {' in client.client.last_prompt
    assert "FeedbackBundle(" not in client.client.last_prompt
