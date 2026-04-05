from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agents.maintenance import MaintenanceAgent
from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import (
    AgentKind,
    ChangeSet,
    FeedbackBundle,
    Finding,
    RepoSnapshot,
    RunContext,
    Severity,
    Task,
    TaskType,
    utc_now,
)
from memory.state_store import StateStore
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentToolkitFactory


@pytest.mark.asyncio
async def test_maintenance_agent_generates_docstring_and_newline_patch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    module_path = repo_root / "module.py"
    module_path.write_text("def add(a, b):    \n    return a + b\n\n\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={"module.py": "hash"},
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["module.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        static_findings=[
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="missing-module-docstring",
                message="Python module is missing a top-level docstring.",
                category="documentation",
                path="module.py",
            ),
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="W291",
                message="trailing whitespace",
                category="style",
                path="module.py",
            ),
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.LOW,
                rule_id="W391",
                message="too many blank lines at end of file",
                category="style",
                path="module.py",
            ),
        ],
    )
    task = Task(
        task_id="task-1",
        run_id="run-1",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["module.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-1",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches
    new_content = result.patch.file_patches[0].new_content
    assert '"""Maintained by Close-Devs."""' in new_content
    assert new_content.endswith("\n")
    assert "def add(a, b):\n" in new_content
    assert not new_content.endswith("\n\n")


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_missing_dependency_declaration(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "infra" / "security").mkdir(parents=True)
    (repo_root / "src" / "requirements.txt").write_text("fastapi==0.115.0\n", encoding="utf-8")
    (repo_root / "src" / "infra" / "security" / "passwords.py").write_text(
        "from argon2 import PasswordHasher\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "src/requirements.txt"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={
            "src/infra/security/passwords.py": "hash-passwords",
            "src/requirements.txt": "hash-reqs",
        },
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["src/infra/security/passwords.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Pytest fails during collection with ModuleNotFoundError: No module named 'argon2'.",
                category="dependency",
                root_cause_class="dependency",
                evidence={"stderr_excerpt": "ModuleNotFoundError: No module named 'argon2'"},
            )
        ],
    )
    task = Task(
        task_id="task-dependency",
        run_id="run-dependency",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/infra/security/passwords.py"],
        payload={
            "feedback": feedback,
            "handoffs": [
                {
                    "source_agent": AgentKind.DYNAMIC_DEBUG.value,
                    "title": "Address dependency blocker",
                    "description": "Missing argon2 dependency blocks test collection.",
                    "recommended_change": "Declare and install argon2 before rerunning tests.",
                    "severity": Severity.HIGH.value,
                    "kind": "dependency",
                    "confidence": 0.95,
                    "affected_files": ["src/infra/security/passwords.py"],
                    "metadata": {
                        "rule_id": "command-failed",
                        "root_cause_class": "dependency",
                    },
                    "evidence": [
                        {
                            "kind": "runtime-output",
                            "title": "missing dependency",
                            "summary": "ModuleNotFoundError: No module named 'argon2'",
                            "data": {
                                "stderr_excerpt": "ModuleNotFoundError: No module named 'argon2'"
                            },
                        }
                    ],
                }
            ],
        },
    )
    context = RunContext(
        run_id="run-dependency",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "src/requirements.txt" in patched_files
    assert "argon2-cffi" in patched_files["src/requirements.txt"]
    assert result.patch.metadata["repair_scope"] == ["dependency"]
    assert result.artifacts["repair_scope"] == ["dependency"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_poetry_dependency_declaration(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "infra" / "security").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        """
[tool.poetry]
name = "demo"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.115.0"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "infra" / "security" / "passwords.py").write_text(
        "from argon2 import PasswordHasher\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "pyproject.toml"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={
            "src/infra/security/passwords.py": "hash-passwords",
            "pyproject.toml": "hash-pyproject",
        },
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["src/infra/security/passwords.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Pytest fails during collection with ModuleNotFoundError: No module named 'argon2'.",
                category="dependency",
                root_cause_class="dependency",
                evidence={"stderr_excerpt": "ModuleNotFoundError: No module named 'argon2'"},
            )
        ],
    )
    task = Task(
        task_id="task-poetry-dependency",
        run_id="run-poetry-dependency",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/infra/security/passwords.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-poetry-dependency",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "pyproject.toml" in patched_files
    assert 'argon2-cffi = "*"' in patched_files["pyproject.toml"]
    assert result.patch.metadata["repair_scope"] == ["dependency"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_missing_test_dependency_declaration(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "tests").mkdir(parents=True)
    (repo_root / "requirements-test.txt").write_text("pytest==8.3.0\n", encoding="utf-8")
    (repo_root / "tests" / "test_feature.py").write_text(
        "import pytest_mock\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "requirements-test.txt"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={
            "tests/test_feature.py": "hash-test",
            "requirements-test.txt": "hash-reqs",
        },
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["tests/test_feature.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Pytest fails during collection with ModuleNotFoundError: No module named 'pytest_mock'.",
                category="dependency",
                root_cause_class="dependency",
                path="tests/test_feature.py",
                evidence={"stderr_excerpt": "ModuleNotFoundError: No module named 'pytest_mock'"},
            )
        ],
    )
    task = Task(
        task_id="task-test-dependency",
        run_id="run-test-dependency",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["tests/test_feature.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-test-dependency",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "requirements-test.txt" in patched_files
    assert "pytest-mock" in patched_files["requirements-test.txt"]
    assert result.patch.metadata["repair_scope"] == ["dependency"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_project_optional_test_dependency_declaration(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "tests").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"
dependencies = ["httpx>=0.27.0"]

[project.optional-dependencies]
test = ["pytest>=8.0.0"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_feature.py").write_text(
        "import hypothesis\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "pyproject.toml"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    snapshot = RepoSnapshot(
        repo_root=str(repo_root),
        scanned_at=utc_now(),
        revision=None,
        file_hashes={
            "tests/test_feature.py": "hash-test",
            "pyproject.toml": "hash-pyproject",
        },
    )
    feedback = FeedbackBundle(
        snapshot=snapshot,
        change_set=ChangeSet(
            changed_files=["tests/test_feature.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Pytest fails during collection with ModuleNotFoundError: No module named 'hypothesis'.",
                category="dependency",
                root_cause_class="dependency",
                path="tests/test_feature.py",
                evidence={"stderr_excerpt": "ModuleNotFoundError: No module named 'hypothesis'"},
            )
        ],
    )
    task = Task(
        task_id="task-project-test-extra",
        run_id="run-project-test-extra",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["tests/test_feature.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-project-test-extra",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "pyproject.toml" in patched_files
    assert '"hypothesis",' in patched_files["pyproject.toml"]
    assert result.patch.metadata["repair_scope"] == ["dependency"]


@pytest.mark.asyncio
async def test_maintenance_agent_creates_env_example_for_missing_environment_variable(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"app.py": "hash-app"},
        ),
        change_set=ChangeSet(
            changed_files=["app.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Missing environment variable APP_SECRET.",
                category="config",
                root_cause_class="config",
                evidence={"stderr_excerpt": "KeyError: 'APP_SECRET'"},
            )
        ],
    )
    task = Task(
        task_id="task-env-template",
        run_id="run-env-template",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["app.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-env-template",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert patched_files[".env.example"] == "APP_SECRET=\n"
    assert result.patch.metadata["repair_scope"] == ["config"]
    assert result.artifacts["auto_fixed_blockers"] == ["config"]


@pytest.mark.asyncio
async def test_maintenance_agent_extracts_env_placeholder_from_settings_validation_text(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "settings.py").write_text("class Settings:\n    pass\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"settings.py": "hash-settings"},
        ),
        change_set=ChangeSet(
            changed_files=["settings.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Settings validation failed.",
                category="config",
                root_cause_class="config",
                evidence={
                    "stderr_excerpt": (
                        "pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings\n"
                        "DATABASE_URL\n"
                        "  Field required [type=missing, input_value={}, input_type=dict]\n"
                    )
                },
            )
        ],
    )
    task = Task(
        task_id="task-settings-env",
        run_id="run-settings-env",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["settings.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-settings-env",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert patched_files[".env.example"] == "DATABASE_URL=\n"


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_pyproject_pytest_pythonpath_for_src_layout(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "pyproject.toml"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={
                "src/app/__init__.py": "hash-app",
                "pyproject.toml": "hash-pyproject",
            },
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Pytest fails during collection with ModuleNotFoundError: No module named 'app'.",
                category="runtime",
                root_cause_class="startup",
                path="tests/test_app.py",
                evidence={"stderr_excerpt": "ImportError while importing test module\nModuleNotFoundError: No module named 'app'"},
            )
        ],
    )
    task = Task(
        task_id="task-pytest-bootstrap",
        run_id="run-pytest-bootstrap",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-pytest-bootstrap",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "pyproject.toml" in patched_files
    assert "[tool.pytest.ini_options]" in patched_files["pyproject.toml"]
    assert 'pythonpath = ["src"]' in patched_files["pyproject.toml"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_creates_pytest_ini_when_pyproject_is_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="ModuleNotFoundError: No module named 'app'.",
                category="runtime",
                root_cause_class="startup",
                evidence={"stderr_excerpt": "ImportError while importing test module"},
            )
        ],
    )
    task = Task(
        task_id="task-pytest-ini",
        run_id="run-pytest-ini",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-pytest-ini",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert patched_files["pytest.ini"] == "[pytest]\npythonpath = src\n"
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_alembic_prepend_sys_path_for_src_layout(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "alembic.ini").write_text("[alembic]\nscript_location = alembic\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "alembic.ini"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "alembic.ini": "hash-alembic"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "alembic.ini"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Alembic startup failed with ModuleNotFoundError: No module named 'app'.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "alembic upgrade head",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\n  File \"alembic/env.py\", line 1, in <module>",
                },
            )
        ],
    )
    task = Task(
        task_id="task-alembic-startup",
        run_id="run-alembic-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "alembic.ini"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-alembic-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "alembic.ini" in patched_files
    assert "prepend_sys_path = src" in patched_files["alembic.ini"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_gunicorn_pythonpath_for_src_layout(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "gunicorn.conf.py").write_text('workers = 2\nbind = "127.0.0.1:8000"\n', encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py", "gunicorn.conf.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "gunicorn.conf.py": "hash-gunicorn"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "gunicorn.conf.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Gunicorn startup failed with ModuleNotFoundError: No module named 'app'.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "gunicorn app.main:app -c gunicorn.conf.py",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\n[INFO] Starting gunicorn",
                },
            )
        ],
    )
    task = Task(
        task_id="task-gunicorn-startup",
        run_id="run-gunicorn-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "gunicorn.conf.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-gunicorn-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "gunicorn.conf.py" in patched_files
    assert 'pythonpath = "src"' in patched_files["gunicorn.conf.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_keeps_startup_argument_mismatch_as_unresolved_handoff(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"app.py": "hash-app"},
        ),
        change_set=ChangeSet(
            changed_files=["app.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="app.py: error: the following arguments are required: --mode",
                category="runtime",
                root_cause_class="startup",
                evidence={"stderr_excerpt": "usage: app.py [-h] --mode MODE"},
            )
        ],
    )
    task = Task(
        task_id="task-startup-args",
        run_id="run-startup-args",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["app.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-startup-args",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches == []
    assert result.artifacts["auto_fixed_blockers"] == []
    assert any(item["kind"] == "startup" for item in result.artifacts["unresolved_handoffs"])


@pytest.mark.asyncio
async def test_maintenance_agent_marks_uvicorn_src_layout_blocker_as_unresolved(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Uvicorn failed to load the ASGI app because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "uvicorn app.main:app --reload",
                    "stderr_excerpt": 'ERROR:    Error loading ASGI app. Could not import module "app.main".',
                },
            )
        ],
    )
    task = Task(
        task_id="task-uvicorn-startup",
        run_id="run-uvicorn-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-uvicorn-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches == []
    assert any(item["reason"] == "uvicorn-app-dir-unsupported" for item in result.artifacts["unresolved_handoffs"])
    assert any("uvicorn" in item.get("guidance", "").lower() for item in result.artifacts["unresolved_handoffs"])


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_uvicorn_runner_with_src_bootstrap(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "serve.py").write_text(
        "import uvicorn\n\n"
        'if __name__ == "__main__":\n'
        '    uvicorn.run("app.main:app", reload=True)\n',
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "serve.py": "hash-serve"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "serve.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Uvicorn failed to load the ASGI app because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "python serve.py",
                    "stderr_excerpt": 'ERROR:    Error loading ASGI app. Could not import module "app.main".',
                },
            )
        ],
    )
    task = Task(
        task_id="task-uvicorn-runner-startup",
        run_id="run-uvicorn-runner-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "serve.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-uvicorn-runner-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "serve.py" in patched_files
    assert "_CLOSE_DEVS_SRC_ROOT" in patched_files["serve.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_marks_celery_src_layout_blocker_as_unresolved(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Celery worker startup failed because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "celery -A app.worker worker",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\ncelery.exceptions.InvalidValueError",
                },
            )
        ],
    )
    task = Task(
        task_id="task-celery-startup",
        run_id="run-celery-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-celery-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches == []
    assert any(item["reason"] == "celery-pythonpath-unsupported" for item in result.artifacts["unresolved_handoffs"])
    assert any("celery" in item.get("guidance", "").lower() for item in result.artifacts["unresolved_handoffs"])


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_celeryconfig_with_src_bootstrap(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "celeryconfig.py").write_text('broker_url = "redis://localhost:6379/0"\n', encoding="utf-8")

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "celeryconfig.py": "hash-celery"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "celeryconfig.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Celery worker startup failed because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "celery -A app.worker worker",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\ncelery.exceptions.InvalidValueError",
                },
            )
        ],
    )
    task = Task(
        task_id="task-celeryconfig-startup",
        run_id="run-celeryconfig-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "celeryconfig.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-celeryconfig-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "celeryconfig.py" in patched_files
    assert "_CLOSE_DEVS_SRC_ROOT" in patched_files["celeryconfig.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_manage_py_with_src_bootstrap(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        "import os\n\n"
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "manage.py": "hash-manage"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "manage.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="manage.py startup failed with ModuleNotFoundError: No module named 'app'.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "python manage.py runserver",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\n  File \"manage.py\", line 1, in <module>",
                },
            )
        ],
    )
    task = Task(
        task_id="task-managepy-startup",
        run_id="run-managepy-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "manage.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-managepy-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "manage.py" in patched_files
    assert "_CLOSE_DEVS_SRC_ROOT" in patched_files["manage.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_root_asgi_entrypoint_with_src_bootstrap(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "asgi.py").write_text(
        "from app.main import application\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "asgi.py": "hash-asgi"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "asgi.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Uvicorn failed to load the ASGI app because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "uvicorn asgi:application --reload",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\n  File \"asgi.py\", line 1, in <module>",
                },
            )
        ],
    )
    task = Task(
        task_id="task-asgi-entrypoint-startup",
        run_id="run-asgi-entrypoint-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "asgi.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-asgi-entrypoint-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "asgi.py" in patched_files
    assert "_CLOSE_DEVS_SRC_ROOT" in patched_files["asgi.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_repairs_root_wsgi_entrypoint_with_src_bootstrap(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "wsgi.py").write_text(
        "from app.main import application\n",
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "wsgi.py": "hash-wsgi"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "wsgi.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        dynamic_findings=[
            Finding(
                source_agent=AgentKind.DYNAMIC_DEBUG,
                severity=Severity.HIGH,
                rule_id="command-failed",
                message="Gunicorn failed to load the WSGI app because the app module could not be imported.",
                category="runtime",
                root_cause_class="startup",
                evidence={
                    "command": "gunicorn wsgi:application",
                    "stderr_excerpt": "ModuleNotFoundError: No module named 'app'\n  File \"wsgi.py\", line 1, in <module>",
                },
            )
        ],
    )
    task = Task(
        task_id="task-wsgi-entrypoint-startup",
        run_id="run-wsgi-entrypoint-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "wsgi.py"],
        payload={"feedback": feedback},
    )
    context = RunContext(
        run_id="run-wsgi-entrypoint-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    patched_files = {file_patch.path: file_patch.new_content for file_patch in result.patch.file_patches}
    assert "wsgi.py" in patched_files
    assert "_CLOSE_DEVS_SRC_ROOT" in patched_files["wsgi.py"]
    assert result.patch.metadata["repair_scope"] == ["startup"]
    assert result.artifacts["auto_fixed_blockers"] == ["startup"]


@pytest.mark.asyncio
async def test_maintenance_agent_does_not_autofix_static_startup_advisory_without_dynamic_confirmation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        'os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
        encoding="utf-8",
    )

    config = AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=tmp_path / "rules.toml",
        include=["*.py", "**/*.py"],
        exclude=[],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(ruff_command=None, mypy_command=None, bandit_command=None),
        dynamic_debug=DynamicDebugConfig(smoke_commands=[], test_commands=[]),
        database=sqlite_database_config(tmp_path),
    )
    config.rules_path.write_text("", encoding="utf-8")

    topology = await AgentToolkitFactory().discover_startup_topology(repo_root)
    feedback = FeedbackBundle(
        snapshot=RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=None,
            file_hashes={"src/app/__init__.py": "hash-app", "manage.py": "hash-manage"},
        ),
        change_set=ChangeSet(
            changed_files=["src/app/__init__.py", "manage.py"],
            added_files=[],
            removed_files=[],
            reason="hash-diff",
        ),
        static_findings=[
            Finding(
                source_agent=AgentKind.STATIC_REVIEW,
                severity=Severity.MEDIUM,
                rule_id="startup-topology-anchor-missing",
                message="Static review found a missing startup anchor for manage.py.",
                category="config",
                root_cause_class="startup",
                path="manage.py",
                evidence={
                    "advisory": True,
                    "startup_context": "django_manage",
                    "entrypoint_path": "manage.py",
                    "config_anchor_path": "manage.py",
                    "repair_hint": "managepy-sys-path-src",
                },
            )
        ],
    )
    task = Task(
        task_id="task-static-advisory-startup",
        run_id="run-static-advisory-startup",
        agent_kind=AgentKind.MAINTENANCE,
        task_type=TaskType.MAINTENANCE,
        targets=["src/app/__init__.py", "manage.py"],
        payload={
            "feedback": feedback,
            "startup_topology": topology,
            "handoffs": [
                {
                    "source_agent": AgentKind.STATIC_REVIEW.value,
                    "title": "Review django_manage startup topology",
                    "description": "Static review found a missing startup anchor for manage.py.",
                    "recommended_change": "Add deterministic src bootstrap to manage.py.",
                    "severity": Severity.MEDIUM.value,
                    "kind": "startup",
                    "confidence": 0.82,
                    "affected_files": ["manage.py"],
                    "metadata": {
                        "root_cause_class": "startup",
                        "startup_context": "django_manage",
                        "entrypoint_path": "manage.py",
                        "config_anchor_path": "manage.py",
                        "repair_hint": "managepy-sys-path-src",
                        "advisory": True,
                    },
                    "evidence": [],
                }
            ],
        },
    )
    context = RunContext(
        run_id="run-static-advisory-startup",
        repo_root=repo_root,
        working_repo_root=repo_root,
        config=config,
        state_store=await StateStore.create(config.database, ensure_schema=True),
        logger=logging.getLogger("test"),
        rules={},
    )

    try:
        result = await MaintenanceAgent().run(task, context)
    finally:
        await context.state_store.close()

    assert result.patch is not None
    assert result.patch.file_patches == []
    assert result.artifacts["auto_fixed_blockers"] == []
    assert result.artifacts["repair_scope"] == ["advisory"]
