from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppConfig, DynamicDebugConfig, LLMConfig, StaticReviewConfig
from core.models import PullRequestContext
from core.orchestrator import Orchestrator
from tests.support import sqlite_database_config
from tools.agent_toolkit import AgentToolkitFactory
from tools.static_context_builder import StaticContextBuilder


def _build_config(tmp_path: Path, repo_root: Path) -> AppConfig:
    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    return AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py", "*.toml", "**/*.toml", "*.ini", "**/*.ini"],
        exclude=["state/**", "reports/**"],
        llm=LLMConfig(provider="mock"),
        static_review=StaticReviewConfig(
            ruff_command=None,
            mypy_command=None,
            bandit_command=None,
        ),
        dynamic_debug=DynamicDebugConfig(smoke_commands=["pytest -q"], test_commands=["pytest -q"]),
        database=sqlite_database_config(tmp_path),
    )


def _create_large_repo(repo_root: Path, file_count: int = 24) -> None:
    (repo_root / "src" / "app" / "bootstrap").mkdir(parents=True)
    (repo_root / "src" / "app" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "src" / "app" / "bootstrap" / "application.py").write_text(
        "from app.settings import SETTINGS\n\n\ndef bootstrap():\n    return SETTINGS\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "settings.py").write_text("SETTINGS = {}\n", encoding="utf-8")
    (repo_root / "manage.py").write_text(
        'import os\nos.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")\n',
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    for index in range(file_count):
        (repo_root / "src" / "app" / f"module_{index:02d}.py").write_text(
            f"def value_{index}():\n    return {index}\n",
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_static_context_builder_preserves_large_ranked_target_digest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_large_repo(repo_root, file_count=36)
    config = _build_config(tmp_path, repo_root)
    toolkit_factory = AgentToolkitFactory()
    builder = StaticContextBuilder(
        toolkit_factory=toolkit_factory,
        static_tooling=toolkit_factory.static_tooling,
    )
    targets = sorted(
        path.relative_to(repo_root).as_posix()
        for path in repo_root.rglob("*.py")
    )

    bundle = await builder.build(
        repo_root=repo_root,
        targets=targets,
        config=config,
        rules={},
    )

    assert len(bundle.target_digest.prioritized_targets) >= 20
    assert len(bundle.top_targets) > 12
    assert "manage.py" in bundle.top_targets
    assert bundle.startup_topology.entrypoints
    assert bundle.repo_map_summary.src_layout is True
    assert bundle.import_adjacency_digest.related_files
    assert bundle.baseline_static_digest.total_findings >= len(targets)


@pytest.mark.asyncio
async def test_scan_pr_repo_populates_static_context_and_static_task_payload(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _create_large_repo(repo_root, file_count=6)
    config = _build_config(tmp_path, repo_root)
    config.environment.enabled = False
    orchestrator = await Orchestrator.create(config, ensure_schema=True)
    pr_context = PullRequestContext(
        repo_full_name="example/demo",
        base_repo_full_name="example/demo",
        head_repo_full_name="example/demo",
        pr_number=1,
        title="Test",
        html_url="https://example.com/pr/1",
        base_branch="main",
        head_branch="feature",
        head_sha="abc123",
    )
    try:
        state = await orchestrator._node_scan_pr_repo(
            {
                "run_id": "run-pr-context",
                "pr_repo_root": str(repo_root),
                "pr_context": pr_context,
            }
        )
    finally:
        await orchestrator.close()

    static_context = state["static_context"]
    assert static_context.enabled is True
    assert state["startup_topology"].entrypoints
    assert state["static_task"].payload["static_context"] is static_context
    assert len(static_context.top_targets) >= 6


@pytest.mark.asyncio
async def test_static_context_builder_detects_node_project_profile(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "package.json").write_text(
        '{"name":"demo","scripts":{"lint":"eslint .","test":"vitest"},"dependencies":{"react":"18.0.0"}}',
        encoding="utf-8",
    )
    (repo_root / "tsconfig.json").write_text('{"compilerOptions":{"strict":true}}', encoding="utf-8")
    (repo_root / "src" / "index.ts").write_text(
        'import { helper } from "./helper"\nexport const run = () => helper()\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "helper.ts").write_text("export function helper() { return 1 }\n", encoding="utf-8")
    config = _build_config(tmp_path, repo_root)
    config.include = ["*", "**/*"]
    builder = StaticContextBuilder(
        toolkit_factory=AgentToolkitFactory(),
        static_tooling=AgentToolkitFactory().static_tooling,
    )

    bundle = await builder.build(
        repo_root=repo_root,
        targets=["package.json", "tsconfig.json", "src/index.ts", "src/helper.ts"],
        config=config,
        rules={},
    )

    assert bundle.language_profile.primary_ecosystem == "node"
    assert "typescript" in bundle.language_profile.languages
    assert bundle.project_topology.dependency_manifests == ["package.json"]
    assert any(item.context == "node_script:lint" for item in bundle.project_topology.entrypoints)
    assert any(item.path == "tsconfig.json" for item in bundle.project_topology.config_anchors)
    assert "eslint" in bundle.tool_coverage_summary.enabled_tools


@pytest.mark.asyncio
async def test_static_context_builder_detects_go_rust_and_java_manifests(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "cmd" / "api").mkdir(parents=True)
    (repo_root / "src" / "main" / "java" / "app").mkdir(parents=True)
    (repo_root / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (repo_root / "cmd" / "api" / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    (repo_root / "Cargo.toml").write_text("[package]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8")
    (repo_root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    (repo_root / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    (repo_root / "src" / "main" / "java" / "app" / "Main.java").write_text(
        "package app; public class Main { public static void main(String[] args) {} }\n",
        encoding="utf-8",
    )
    config = _build_config(tmp_path, repo_root)
    config.include = ["*", "**/*"]
    builder = StaticContextBuilder(
        toolkit_factory=AgentToolkitFactory(),
        static_tooling=AgentToolkitFactory().static_tooling,
    )

    bundle = await builder.build(
        repo_root=repo_root,
        targets=["go.mod", "cmd/api/main.go", "Cargo.toml", "src/main.rs", "pom.xml"],
        config=config,
        rules={},
    )

    assert "go" in bundle.project_topology.ecosystems
    assert "rust" in bundle.project_topology.ecosystems
    assert "java" in bundle.project_topology.ecosystems
    assert any(item.context == "go_cmd" for item in bundle.project_topology.entrypoints)
    assert any(item.path == "Cargo.toml" for item in bundle.project_topology.config_anchors)
    assert any(item.path == "pom.xml" for item in bundle.project_topology.config_anchors)
