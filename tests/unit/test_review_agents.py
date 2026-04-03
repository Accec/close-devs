from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agents.dynamic_debug import DynamicDebugAgent
from agents.static_review import StaticReviewAgent
from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import AgentActionType, AgentKind, AgentStep, RunContext, Task, TaskType, ToolSpec
from llm.base import BaseLLMClient
from memory.state_store import StateStore
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentToolkitFactory
from tools.command_runner import CommandResult


def _build_config(tmp_path: Path, repo_root: Path) -> AppConfig:
    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=["pytest -q"], test_commands=["pytest -q"]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")
    return config


class StubFailingTestRunner:
    async def run(self, command: str, repo_root: Path, timeout_seconds: int) -> CommandResult:
        return CommandResult(
            command=command,
            returncode=1,
            stdout=(
                "Traceback (most recent call last):\n"
                '  File "/tmp/example.py", line 3, in <module>\n'
                '    raise ValueError("boom")\n'
                "ValueError: boom\n"
            ),
            stderr="",
            duration_seconds=0.01,
            timed_out=False,
        )


class SummaryOnlyStaticLLM(BaseLLMClient):
    async def complete_agent_step(
        self,
        *,
        session,
        available_tools: list[ToolSpec],
    ) -> AgentStep:
        if not session.steps:
            return AgentStep(
                step_index=0,
                decision_summary="Inspect the bootstrap module before finalizing.",
                action_type=AgentActionType.TOOL_CALL,
                tool_name="read_file",
                tool_input={"path": "src/app/bootstrap/application.py"},
            )
        return AgentStep(
            step_index=len(session.steps),
            decision_summary="Finalize with a high-value bootstrap concern.",
            action_type=AgentActionType.FINALIZE,
            final_response={
                "summary": (
                    "Static review found mostly low-value documentation lint, but one "
                    "higher-value correctness/operability issue stands out in bootstrap initialization."
                ),
                "findings": [],
                "fix_requests": [],
            },
        )


@pytest.mark.asyncio
async def test_static_review_agent_performs_semantic_code_review(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "# TODO: remove debug path\n"
        "def run(value):\n"
        "    print(value)\n"
        "    try:\n"
        "        return 10 / value\n"
        "    except:\n"
        "        return 0\n",
        encoding="utf-8",
    )
    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    context = RunContext(
        run_id="run-static",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
    )
    task = Task(
        task_id="task-static",
        run_id="run-static",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        targets=["module.py"],
        payload={},
    )

    try:
        result = await StaticReviewAgent().run(task, context)
    finally:
        await state_store.close()

    rule_ids = {finding.rule_id for finding in result.findings}
    assert "todo-comment" in rule_ids
    assert "bare-except" in rule_ids
    assert "debug-print-statement" in rule_ids
    assert result.artifacts["reviewed_files"] == ["module.py"]
    assert result.artifacts["semantic_findings"] >= 3
    assert result.artifacts["handoffs"]
    assert result.artifacts["session_summary"]["step_count"] >= 3


@pytest.mark.asyncio
async def test_dynamic_debug_agent_produces_agent_diagnosis(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    context = RunContext(
        run_id="run-dynamic",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
    )
    task = Task(
        task_id="task-dynamic",
        run_id="run-dynamic",
        agent_kind=AgentKind.DYNAMIC_DEBUG,
        task_type=TaskType.DYNAMIC_DEBUG,
        targets=[],
        payload={"commands": ["pytest -q"]},
    )

    try:
        result = await DynamicDebugAgent(
            toolkit_factory=AgentToolkitFactory(test_runner=StubFailingTestRunner()),
        ).run(task, context)
    finally:
        await state_store.close()

    assert any(finding.rule_id == "command-failed" for finding in result.findings)
    assert any(finding.rule_id == "runtime-root-cause" for finding in result.findings)
    assert "agent_diagnosis" in result.artifacts
    assert result.artifacts["agent_diagnosis"]["suggestions"]
    assert result.artifacts["handoffs"]
    assert result.artifacts["session_summary"]["tool_call_count"] >= 2


def test_finding_from_dict_normalizes_non_mapping_evidence() -> None:
    agent = StaticReviewAgent()

    finding = agent.finding_from_dict(
        {
            "rule_id": "semantic-observation",
            "message": "LLM returned evidence as a list.",
            "category": "analysis",
            "line": "12",
            "evidence": ["first", {"detail": "second"}],
        },
        default_source_agent=AgentKind.STATIC_REVIEW,
    )

    assert finding.line == 12
    assert finding.evidence == {"items": ["first", {"detail": "second"}]}


@pytest.mark.asyncio
async def test_static_review_agent_derives_handoffs_from_deterministic_logic_findings(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "module.py").write_text(
        "def append_value(values=[]):\n"
        "    values.append(1)\n"
        "    return values\n",
        encoding="utf-8",
    )
    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    context = RunContext(
        run_id="run-static-deterministic",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
    )
    task = Task(
        task_id="task-static-deterministic",
        run_id="run-static-deterministic",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        targets=["module.py"],
        payload={},
    )

    try:
        result = await StaticReviewAgent().run(task, context)
    finally:
        await state_store.close()

    assert any(finding.rule_id == "mutable-default-argument" for finding in result.findings)
    assert result.artifacts["high_value_findings"] >= 1
    assert result.artifacts["handoffs"]


@pytest.mark.asyncio
async def test_static_review_agent_promotes_summary_only_logic_issue_to_finding_and_handoff(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app" / "bootstrap").mkdir(parents=True)
    (repo_root / "src" / "app" / "bootstrap" / "application.py").write_text(
        "def bootstrap():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    context = RunContext(
        run_id="run-static-summary",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
    )
    task = Task(
        task_id="task-static-summary",
        run_id="run-static-summary",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        targets=["src/app/bootstrap/application.py"],
        payload={},
    )

    try:
        result = await StaticReviewAgent(llm_client=SummaryOnlyStaticLLM()).run(task, context)
    finally:
        await state_store.close()

    assert any(finding.rule_id == "semantic-review-observation" for finding in result.findings)
    assert result.artifacts["handoffs"]
    assert result.artifacts["high_value_findings"] >= 1
