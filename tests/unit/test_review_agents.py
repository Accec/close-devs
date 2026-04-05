from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from agents.dynamic_debug import DynamicDebugAgent
from agents.static_review import StaticReviewAgent
from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import AgentActionType, AgentKind, AgentStep, RunContext, StaticContextBundle, Task, TaskType, ToolSpec
from llm.base import BaseLLMClient
from memory.state_store import StateStore
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentToolkitFactory
from tools.command_runner import CommandResult
from tools.static_context_builder import StaticContextBuilder


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


class PayloadEchoStaticLLM(BaseLLMClient):
    def __init__(self) -> None:
        self.last_payload: dict[str, object] | None = None

    async def complete_agent_step(
        self,
        *,
        session,
        available_tools: list[ToolSpec],
    ) -> AgentStep:
        self.last_payload = dict(session.payload)
        return AgentStep(
            step_index=len(session.steps),
            decision_summary="Finalize after checking injected static context.",
            action_type=AgentActionType.FINALIZE,
            final_response={
                "summary": "Static context was available before any tool call.",
                "findings": [],
                "fix_requests": [],
            },
        )


def test_dynamic_debug_agent_classifies_missing_env_as_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    classification = DynamicDebugAgent()._classify_runtime_output(
        {
            "stdout": "",
            "stderr": "KeyError: 'APP_SECRET'\n",
            "timed_out": False,
        },
        repo_root=repo_root,
    )

    assert classification == "config"


def test_dynamic_debug_agent_classifies_src_layout_import_failure_as_startup(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    classification = DynamicDebugAgent()._classify_runtime_output(
        {
            "stdout": "",
            "stderr": (
                "ImportError while importing test module '/tmp/test_app.py'\n"
                "ModuleNotFoundError: No module named 'app'\n"
            ),
            "timed_out": False,
        },
        repo_root=repo_root,
    )

    assert classification == "startup"


def test_dynamic_debug_agent_classifies_argument_mismatch_as_startup(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    classification = DynamicDebugAgent()._classify_runtime_output(
        {
            "stdout": "",
            "stderr": "usage: app.py [-h] --mode MODE\napp.py: error: the following arguments are required: --mode\n",
            "timed_out": False,
        },
        repo_root=repo_root,
    )

    assert classification == "startup"


def test_dynamic_debug_agent_classifies_asgi_import_failure_as_startup(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")

    classification = DynamicDebugAgent()._classify_runtime_output(
        {
            "stdout": "",
            "stderr": 'ERROR:    Error loading ASGI app. Could not import module "app.main".\n',
            "timed_out": False,
        },
        repo_root=repo_root,
    )

    assert classification == "startup"


def test_dynamic_debug_agent_tags_uvicorn_startup_context(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    evidence = DynamicDebugAgent()._runtime_evidence(
        {
            "command": "uvicorn app.main:app --reload",
            "returncode": 1,
            "stdout": "",
            "stderr": 'ERROR:    Error loading ASGI app. Could not import module "app.main".\n',
            "timed_out": False,
        }
    )

    assert evidence["startup_context"] == "uvicorn"


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
    assert any(finding.root_cause_class == "application" for finding in result.findings)
    assert "agent_diagnosis" in result.artifacts
    assert result.artifacts["agent_diagnosis"]["suggestions"]
    assert result.artifacts["handoffs"]
    assert any(handoff["kind"] == "runtime" for handoff in result.artifacts["handoffs"])
    assert "investigation_depth" in result.artifacts
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


@pytest.mark.asyncio
async def test_static_review_agent_discovers_startup_topology_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
        encoding="utf-8",
    )
    (repo_root / "asgi.py").write_text("from app.main import application\n", encoding="utf-8")
    (repo_root / "wsgi.py").write_text("from app.main import application\n", encoding="utf-8")
    (repo_root / "serve.py").write_text(
        "import uvicorn\n\n"
        'if __name__ == "__main__":\n'
        '    uvicorn.run("app.main:app", reload=True)\n',
        encoding="utf-8",
    )
    (repo_root / "celeryconfig.py").write_text('broker_url = "redis://localhost:6379/0"\n', encoding="utf-8")
    (repo_root / "gunicorn.conf.py").write_text('workers = 2\nbind = "127.0.0.1:8000"\n', encoding="utf-8")
    (repo_root / "alembic.ini").write_text("[alembic]\nscript_location = alembic\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )

    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    context = RunContext(
        run_id="run-topology",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
    )
    task = Task(
        task_id="task-topology",
        run_id="run-topology",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        targets=["manage.py"],
        payload={},
    )

    try:
        result = await StaticReviewAgent().run(task, context)
    finally:
        await state_store.close()

    topology = result.artifacts["startup_topology"]
    contexts = {item.context for item in topology.entrypoints}

    assert result.artifacts["entrypoint_count"] >= 6
    assert result.artifacts["config_anchor_count"] >= 7
    assert {"django_manage", "asgi", "wsgi", "uvicorn", "celery", "gunicorn", "alembic", "pytest"} <= contexts
    assert any(finding.rule_id == "startup-topology-anchor-missing" for finding in result.findings)
    assert any(
        handoff["metadata"].get("entrypoint_path") == "manage.py"
        for handoff in result.artifacts["startup_handoffs"]
    )


@pytest.mark.asyncio
async def test_static_review_agent_receives_prebuilt_static_context_and_reports_summary(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "src" / "app" / "bootstrap.py").write_text(
        "from app import settings\n\n\ndef bootstrap():\n    return settings\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "settings.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )

    config = _build_config(tmp_path, repo_root)
    state_store = await StateStore.create(config.database, ensure_schema=True)
    static_context = await StaticContextBuilder(
        toolkit_factory=AgentToolkitFactory(),
        static_tooling=AgentToolkitFactory().static_tooling,
    ).build(
        repo_root=repo_root,
        targets=["src/app/bootstrap.py", "src/app/settings.py", "pyproject.toml"],
        config=config,
        rules={},
    )
    llm = PayloadEchoStaticLLM()
    context = RunContext(
        run_id="run-static-context",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=state_store,
        logger=logging.getLogger("test"),
        rules={},
        startup_topology=static_context.startup_topology,
        static_context=static_context,
    )
    task = Task(
        task_id="task-static-context",
        run_id="run-static-context",
        agent_kind=AgentKind.STATIC_REVIEW,
        task_type=TaskType.STATIC_REVIEW,
        targets=["src/app/bootstrap.py", "src/app/settings.py", "pyproject.toml"],
        payload={"static_context": static_context},
    )

    try:
        result = await StaticReviewAgent(llm_client=llm).run(task, context)
    finally:
        await state_store.close()

    assert isinstance(llm.last_payload, dict)
    payload_static_context = llm.last_payload["static_context"]
    assert isinstance(payload_static_context, StaticContextBundle)
    assert payload_static_context.startup_topology.entrypoints
    assert payload_static_context.project_topology.entrypoints
    assert payload_static_context.language_profile.primary_language == "python"
    assert isinstance(payload_static_context.tool_coverage_summary.tool_statuses, dict)
    assert payload_static_context.baseline_static_digest.total_findings >= 1
    assert result.artifacts["static_context_summary"]["enabled"] is True
    assert result.artifacts["project_topology_summary"]["entrypoint_count"] >= 1
    assert result.artifacts["language_profile"].primary_language == "python"
    assert result.artifacts["top_target_count"] >= 3
    assert "baseline_digest_counts" in result.artifacts


def test_dynamic_debug_agent_matches_runtime_failure_to_static_topology(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
        encoding="utf-8",
    )

    topology = asyncio.run(AgentToolkitFactory().discover_startup_topology(repo_root))

    evidence = DynamicDebugAgent()._runtime_evidence(
        {
            "command": "python manage.py runserver",
            "returncode": 1,
            "stdout": "",
            "stderr": "ModuleNotFoundError: No module named 'app'\n  File \"manage.py\", line 1, in <module>",
            "timed_out": False,
        },
        root_cause_class="startup",
        startup_topology=topology,
    )

    assert evidence["startup_context"] == "django_manage"
    assert evidence["matched_entrypoint"] == "manage.py"
    assert evidence["matched_config_anchor"] == "manage.py"
