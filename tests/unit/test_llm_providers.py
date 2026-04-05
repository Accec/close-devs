from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from core.config import AppConfig, LLMConfig
from core.models import AgentActionType, AgentKind, AgentSession, ChangeSet, RepoSnapshot, TaskStatus, TaskType, ToolSpec, WorkflowReport, utc_now
from core.orchestrator import Orchestrator
from llm.anthropic_native import AnthropicNativeLLMClient
from llm.google_genai import GoogleGenAILLMClient
from llm.ollama_local import OllamaLocalLLMClient
from llm.openai_compatible import OpenAICompatibleLLMClient
from llm.openai_native import OpenAINativeLLMClient


class StubChatClient:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        prompt = getattr(messages[-1], "content", "")
        self.last_prompt = str(prompt)
        return SimpleNamespace(
            content=(
                '{"decision_summary":"Run complete.",'
                '"action_type":"finalize",'
                '"tool_name":null,'
                '"tool_input":{},'
                '"final_response":{"summary":"provider-ready"}}'
            )
        )


class ExplodingChatClient:
    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        raise RuntimeError("provider invocation failed")


class NoisyJSONChatClient:
    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        return SimpleNamespace(
            content=(
                'Here is the next step.\n'
                '{"decision_summary":"Run complete.",'
                '"action_type":"finalize",'
                '"tool_name":null,'
                '"tool_input":{},'
                '"final_response":{"summary":"provider-ready"}}\n'
                "Additional note that should be ignored."
            )
        )


def _maintenance_session() -> AgentSession:
    return AgentSession(
        session_id="session-maintenance",
        run_id="run-1",
        task_id="task-1",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        working_repo_root="/tmp/repo",
        objective="prepare maintenance patch",
        targets=["module.py"],
        payload={"feedback": {"summary": "example"}, "handoffs": []},
    )


def _static_session_with_large_context() -> AgentSession:
    return AgentSession(
        session_id="session-static",
        run_id="run-1",
        task_id="task-static",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        working_repo_root="/tmp/repo",
        objective="review statically",
        targets=[f"module_{index:02d}.py" for index in range(60)],
        payload={
            "static_context": {
                "top_targets": [f"target_{index:02d}.py" for index in range(60)],
                "high_signal_targets": [f"signal_{index:02d}.py" for index in range(30)],
                "related_files": [f"related_{index:02d}.py" for index in range(30)],
                "baseline_static_digest": {
                    "top_findings": [
                        {"rule_id": f"rule-{index:02d}", "message": f"finding-{index:02d}"}
                        for index in range(60)
                    ]
                },
            }
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_cls",
    [
        OpenAINativeLLMClient,
        OpenAICompatibleLLMClient,
        AnthropicNativeLLMClient,
        GoogleGenAILLMClient,
        OllamaLocalLLMClient,
    ],
)
async def test_provider_clients_share_structured_prompt_behavior(client_cls: type[object]) -> None:
    client = client_cls.__new__(client_cls)
    client.config = LLMConfig(provider=getattr(client_cls, "provider_name", "mock"), model="demo-model")
    client.client = StubChatClient()

    step = await client.complete_agent_step(
        session=_maintenance_session(),
        available_tools=[
            ToolSpec(
                name="prepare_safe_patch",
                description="Build a safe patch candidate.",
                input_schema={"feedback": "serialized feedback bundle"},
            )
        ],
    )

    assert step.action_type == AgentActionType.FINALIZE
    assert step.final_response["summary"] == "provider-ready"
    assert client.client.last_prompt is not None
    assert '"payload": {' in client.client.last_prompt


@pytest.mark.asyncio
async def test_provider_client_raises_on_invoke_failure() -> None:
    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.config = LLMConfig(provider="openai_compatible", model="demo-model")
    client.client = ExplodingChatClient()

    with pytest.raises(RuntimeError, match="provider invocation failed"):
        await client.complete_agent_step(
            session=_maintenance_session(),
            available_tools=[ToolSpec(name="prepare_safe_patch", description="x", input_schema={})],
        )


@pytest.mark.asyncio
async def test_provider_client_tolerates_extra_text_around_json_response() -> None:
    client = AnthropicNativeLLMClient.__new__(AnthropicNativeLLMClient)
    client.config = LLMConfig(provider="anthropic", model="demo-model")
    client.client = NoisyJSONChatClient()

    step = await client.complete_agent_step(
        session=_maintenance_session(),
        available_tools=[ToolSpec(name="prepare_safe_patch", description="x", input_schema={})],
    )

    assert step.action_type == AgentActionType.FINALIZE
    assert step.final_response["summary"] == "provider-ready"


@pytest.mark.asyncio
async def test_provider_prompt_preserves_static_context_lists_beyond_generic_limit() -> None:
    client = OpenAINativeLLMClient.__new__(OpenAINativeLLMClient)
    client.config = LLMConfig(provider="openai", model="demo-model")
    client.client = StubChatClient()

    await client.complete_agent_step(
        session=_static_session_with_large_context(),
        available_tools=[ToolSpec(name="run_static_review", description="x", input_schema={})],
    )

    assert client.client.last_prompt is not None
    assert "target_39.py" in client.client.last_prompt
    assert "target_40.py" not in client.client.last_prompt
    assert "signal_19.py" in client.client.last_prompt
    assert "signal_20.py" not in client.client.last_prompt
    assert "related_19.py" in client.client.last_prompt
    assert "related_20.py" not in client.client.last_prompt
    assert "rule-49" in client.client.last_prompt
    assert "rule-50" not in client.client.last_prompt


def test_orchestrator_agent_llm_config_applies_per_agent_provider_overrides(tmp_path) -> None:
    config = AppConfig(
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        llm=LLMConfig(
            provider="openai_compatible",
            model="global-model",
            base_url="https://gateway.example/v1",
            api_key_env="OPENAI_API_KEY",
            timeout_seconds=45,
            temperature=0.1,
            max_retries=3,
            system_prompt="global-prompt",
        ),
    )
    config.agents.static.provider = "google_genai"
    config.agents.static.model = "gemini-2.5-pro"
    config.agents.dynamic.provider = "openai"
    config.agents.dynamic.model = "gpt-5.4"
    config.agents.maintenance.provider = "anthropic"
    config.agents.maintenance.model = "claude-sonnet-4"

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = config

    static_config = Orchestrator._agent_llm_config(orchestrator, config.agents.static)
    dynamic_config = Orchestrator._agent_llm_config(orchestrator, config.agents.dynamic)
    maintenance_config = Orchestrator._agent_llm_config(orchestrator, config.agents.maintenance)

    assert static_config.provider == "google_genai"
    assert static_config.api_key_env == "GOOGLE_API_KEY"
    assert static_config.base_url is None
    assert dynamic_config.provider == "openai"
    assert dynamic_config.api_key_env == "OPENAI_API_KEY"
    assert maintenance_config.provider == "anthropic"
    assert maintenance_config.api_key_env == "ANTHROPIC_API_KEY"


@pytest.mark.asyncio
async def test_shadow_replay_candidates_fail_soft_when_candidate_run_errors(tmp_path) -> None:
    config = AppConfig(
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        llm=LLMConfig(provider="mock"),
    )
    config.skills.shadow_evaluation_enabled = True
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = config
    orchestrator.logger = logging.getLogger("test")

    async def _explode(*args, **kwargs):
        raise RuntimeError("shadow replay decode failure")

    orchestrator._shadow_replay_candidate = _explode  # type: ignore[method-assign]
    now = utc_now()
    report = WorkflowReport(
        run_id="run-shadow",
        workflow_name="maintenance_loop",
        repo_root=str(tmp_path),
        started_at=now,
        finished_at=now,
        status=TaskStatus.SUCCEEDED,
        snapshot=RepoSnapshot(
            repo_root=str(tmp_path),
            scanned_at=now,
            revision=None,
            file_hashes={"module.py": "hash"},
        ),
        change_set=ChangeSet(changed_files=["module.py"], added_files=[], removed_files=[]),
    )

    replays = await Orchestrator._shadow_replay_candidates(
        orchestrator,
        state={},
        report=report,
        candidate_skills={"static_review": SimpleNamespace(version="cand-v1", cooldown_until=None)},
    )

    assert replays["static_review"]["mode"] == "heuristic"
    assert "shadow-replay-failed" in replays["static_review"]["reasons"]
