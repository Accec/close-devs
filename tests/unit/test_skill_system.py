from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.agent_kernel import AgentKernel
from core.config import AgentRuntimeConfig, load_config
from core.models import (
    AgentActionType,
    AgentKind,
    AgentResult,
    AgentSession,
    AgentStep,
    Finding,
    RunContext,
    Severity,
    TaskStatus,
    TaskType,
    WorkflowReport,
    ChangeSet,
    RepoSnapshot,
    utc_now,
)
from llm.base import BaseLLMClient
from memory.state_store import StateStore
from skills.evolution import SkillEvolutionService
from skills.manager import SkillManager
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentTool
from core.models import ToolPermissionSet, ToolResult, ToolSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CaptureSkillLLM(BaseLLMClient):
    def __init__(self) -> None:
        self.last_skill_version: str | None = None
        self.last_policy: dict[str, object] | None = None

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
        self.last_skill_version = skill_version
        self.last_policy = dict(agent_policy or {})
        return AgentStep(
            step_index=session.step_count + 1,
            decision_summary="Finalize immediately for skill-context capture.",
            action_type=AgentActionType.FINALIZE,
            final_response={"summary": "captured"},
        )


def _load_test_config(tmp_path: Path, repo_root: Path):
    config = load_config(PROJECT_ROOT / "config" / "default.toml", repo_override=repo_root)
    config.database = sqlite_database_config(tmp_path)
    config.skills.min_shadow_runs = 1
    config.skills.promotion_margin = 0.0
    return config


@pytest.mark.asyncio
async def test_skill_manager_loads_repo_baselines(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = _load_test_config(tmp_path, repo_root)
    store = await StateStore.create(config.database, ensure_schema=True)
    try:
        manager = SkillManager(config, store)
        active, candidates = await manager.resolve_run_skills(
            repo_root,
            {
                AgentKind.STATIC_REVIEW: config.agents.static,
                AgentKind.DYNAMIC_DEBUG: config.agents.dynamic,
                AgentKind.MAINTENANCE: config.agents.maintenance,
            },
        )
    finally:
        await store.close()

    assert active["static_review"].version == "static-v1"
    assert active["dynamic_debug"].version == "dynamic-v1"
    assert active["maintenance"].version == "maintenance-v1"
    assert active["static_review"].policy.allowed_tools
    assert candidates == {}


@pytest.mark.asyncio
async def test_agent_kernel_passes_skill_context_to_llm(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = _load_test_config(tmp_path, repo_root)
    store = await StateStore.create(config.database, ensure_schema=True)
    llm = CaptureSkillLLM()
    try:
        manager = SkillManager(config, store)
        active, candidates = await manager.resolve_run_skills(
            repo_root,
            {AgentKind.STATIC_REVIEW: config.agents.static},
        )
        context = RunContext(
            run_id="run-skill-kernel",
            repo_root=repo_root,
            working_repo_root=repo_root,
            config=config,
            state_store=store,
            logger=logging.getLogger("test"),
            rules={},
            active_skill=active["static_review"],
            candidate_skill=candidates.get("static_review"),
        )
        session = AgentSession(
            session_id="session-skill-kernel",
            run_id="run-skill-kernel",
            task_id="task-skill-kernel",
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.STATIC_REVIEW,
            working_repo_root=str(repo_root),
            objective="capture skill context",
            targets=[],
            payload={},
            active_skill=active["static_review"],
            active_skill_version=active["static_review"].version,
            skill_profile_hash=active["static_review"].profile_hash,
        )
        kernel = AgentKernel(
            llm_client=llm,
            runtime_config=AgentRuntimeConfig(
                max_steps=4,
                max_tool_calls=4,
                max_consecutive_failures=2,
                allowed_tools=["read_file"],
                allowed_tool_superset=["read_file"],
                max_budget_ceiling=4,
            ),
        )
        result = await kernel.run_session(
            session=session,
            tools={
                "read_file": AgentTool(
                    spec=ToolSpec(
                        name="read_file",
                        description="Read file",
                        input_schema={"path": "relative path"},
                    ),
                    handler=lambda session, tool_input: ToolResult(
                        tool_name="read_file",
                        ok=True,
                        output={},
                        summary="unused",
                    ),
                )
            },
            permissions=ToolPermissionSet(allowed_tools=frozenset({"read_file"}), allow_write=False),
            context=context,
        )
    finally:
        await store.close()

    assert result.session.completion_reason.value == "completed"
    assert llm.last_skill_version == "static-v1"
    assert llm.last_policy is not None
    assert llm.last_policy["tool_preferences"]


@pytest.mark.asyncio
async def test_skill_evolution_promotes_candidate_after_shadow_eval(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = _load_test_config(tmp_path, repo_root)
    store = await StateStore.create(config.database, ensure_schema=True)
    try:
        manager = SkillManager(config, store)
        evolution = SkillEvolutionService(config, store)
        active, _ = await manager.resolve_run_skills(
            repo_root,
            {AgentKind.STATIC_REVIEW: config.agents.static},
        )
        active_skill = active["static_review"]
        result = AgentResult(
            task_id="task-static-skill",
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.STATIC_REVIEW,
            status=TaskStatus.SUCCEEDED,
            summary="Static review found a correctness concern, but findings are mostly low-value documentation noise.",
            findings=[
                Finding(
                    source_agent=AgentKind.STATIC_REVIEW,
                    severity=Severity.LOW,
                    rule_id="missing-module-docstring",
                    message="Module docstring missing.",
                    category="documentation",
                    path=f"module_{index}.py",
                )
                for index in range(12)
            ],
            artifacts={
                "session_id": "session-static-skill",
                "session_summary": {"tool_call_count": 1, "step_count": 2, "budget_exhausted": False},
                "handoffs": [],
                "high_value_findings": 0,
            },
        )
        reflection, candidate = await evolution.reflect_and_seed_candidate(
            repo_root=str(repo_root),
            run_id="run-static-skill",
            task_id=result.task_id,
            session_id="session-static-skill",
            result=result,
            active_skill=active_skill,
        )
        snapshot = RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"module.py": "deadbeef"},
        )
        report = WorkflowReport(
            run_id="run-static-skill",
            workflow_name="static_review",
            repo_root=str(repo_root),
            started_at=utc_now(),
            finished_at=utc_now(),
            status=TaskStatus.SUCCEEDED,
            snapshot=snapshot,
            change_set=ChangeSet(changed_files=["module.py"], added_files=[], removed_files=[]),
            static_result=result,
        )
        metadata = await evolution.evaluate_report(
            repo_root=str(repo_root),
            report=report,
            active_skills={"static_review": active_skill},
            candidate_skills={"static_review": candidate} if candidate is not None else {},
        )
        binding = await store.get_skill_binding(str(repo_root), AgentKind.STATIC_REVIEW)
    finally:
        await store.close()

    assert reflection is not None
    assert candidate is not None
    assert metadata["skill_upgrade_events"]
    assert binding is not None
    assert binding.active_version == candidate.version
