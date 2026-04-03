from __future__ import annotations

from pathlib import Path
import sys

import pytest

from core.config import (
    AppConfig,
    DatabaseConfig,
    DynamicDebugConfig,
    GitHubRuntimeConfig,
    LLMConfig,
    PRWorkflowConfig,
    StaticReviewConfig,
)
from core.models import ArtifactReference, GitHubCapabilities, PullRequestContext, WorkflowTrigger
from core.orchestrator import Orchestrator


class FakeGitHubAdapter:
    def __init__(self) -> None:
        self.summary_comments: list[object] = []
        self.inline_comments: list[dict[str, object]] = []
        self.fix_branches: list[tuple[str, list[str]]] = []
        self.companion_prs: list[object] = []
        self.step_summaries: list[str] = []
        self.token = "token"

    async def resolve_capabilities(
        self,
        pr_context: PullRequestContext,
        allow_companion_pr: bool,
    ) -> GitHubCapabilities:
        return GitHubCapabilities(
            can_comment=True,
            can_inline_review=True,
            can_push_fix_branch=allow_companion_pr and not pr_context.is_from_fork,
            can_open_companion_pr=allow_companion_pr and not pr_context.is_from_fork,
        )

    async def load_pr_context(self, event_path: Path, pr_number_override: int | None = None) -> PullRequestContext | None:
        raise AssertionError("load_pr_context should not be called")

    async def prepare_workspace(self, repo_root: Path, pr_context: PullRequestContext) -> Path:
        return repo_root

    async def resolve_run_artifacts(
        self,
        pr_context: PullRequestContext,
        artifact_name: str,
        references: list[ArtifactReference],
    ) -> list[ArtifactReference]:
        return [
            ArtifactReference(
                name=reference.name,
                path=reference.path,
                url=f"https://example/artifacts/{artifact_name}",
                fallback_url="https://example/actions/24",
            )
            for reference in references
        ]

    async def create_or_update_summary_comment(
        self,
        pr_context: PullRequestContext,
        payload: object,
    ) -> dict[str, object]:
        self.summary_comments.append(payload)
        return {"status": "published", "id": 240, "url": f"https://example/reviews/{pr_context.pr_number}"}

    async def find_existing_inline_comment_markers(self, pr_context: PullRequestContext) -> set[str]:
        return set()

    async def publish_inline_comments(
        self,
        pr_context: PullRequestContext,
        inline_comments: list[dict[str, object]],
    ) -> dict[str, object]:
        self.inline_comments.extend(inline_comments)
        return {"status": "published", "published": len(inline_comments)}

    async def publish_fix_branch(
        self,
        pr_context: PullRequestContext,
        repo_root: Path,
        branch_name: str,
        touched_files: list[str],
    ) -> dict[str, object]:
        self.fix_branches.append((branch_name, list(touched_files)))
        return {"status": "published", "branch": branch_name}

    async def create_or_update_companion_pr(
        self,
        pr_context: PullRequestContext,
        payload: object,
    ) -> dict[str, object]:
        self.companion_prs.append(payload)
        return {"status": "published", "number": 24, "url": "https://example/pulls/24"}

    def workflow_run_url(self) -> str | None:
        return "https://example/actions/24"

    async def write_step_summary(self, markdown: str) -> None:
        self.step_summaries.append(markdown)


def _postgres_database_config(url: str) -> DatabaseConfig:
    return DatabaseConfig(
        backend="postgres",
        url=url,
        url_env="DATABASE_URL",
        echo=False,
    )


def _build_app_config(tmp_path: Path, repo_root: Path, database_url: str) -> AppConfig:
    rules_path = tmp_path / "rules.toml"
    rules_path.write_text("", encoding="utf-8")
    return AppConfig(
        repo_root=repo_root,
        state_dir=tmp_path / "state",
        reports_dir=tmp_path / "reports",
        rules_path=rules_path,
        include=["*.py", "**/*.py"],
        exclude=["state/**", "reports/**"],
        llm=LLMConfig(),
        static_review=StaticReviewConfig(
            max_complexity=10,
            ruff_command=None,
            mypy_command=None,
            bandit_command=None,
        ),
        dynamic_debug=DynamicDebugConfig(
            smoke_commands=[],
            test_commands=[f"{sys.executable} -m pytest -q"],
            timeout_seconds=60,
        ),
        github=GitHubRuntimeConfig(
            repo_full_name="acme/repo",
            token_env="GITHUB_TOKEN",
            review_mode="review",
        ),
        pr_workflow=PRWorkflowConfig(),
        database=_postgres_database_config(database_url),
    )


def _build_pr_context() -> PullRequestContext:
    return PullRequestContext(
        repo_full_name="acme/repo",
        base_repo_full_name="acme/repo",
        head_repo_full_name="acme/repo",
        pr_number=11,
        title="Fix whitespace",
        html_url="https://github.com/acme/repo/pull/11",
        base_branch="main",
        head_branch="feature/fix",
        head_sha="abc123",
        changed_files=["module.py"],
        trigger=WorkflowTrigger.PULL_REQUEST,
        actor="alice",
    )


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_postgres_maintenance_loop_runs_end_to_end(
    tmp_path: Path,
    postgres_database_url: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "from app import add\n\n\ndef test_add():\n    assert add(1, 2) == 4\n",
        encoding="utf-8",
    )

    config = _build_app_config(tmp_path, repo_root, postgres_database_url)
    orchestrator = await Orchestrator.create(config)
    try:
        report = await orchestrator.run_workflow("maintenance_loop")
    finally:
        await orchestrator.close()

    assert report.workflow_name == "maintenance_loop"
    assert report.report_dir
    assert (Path(report.report_dir) / "summary.md").exists()


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_postgres_pull_request_workflow_runs_end_to_end(
    tmp_path: Path,
    postgres_database_url: str,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_module.py").write_text(
        "from module import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )

    config = _build_app_config(tmp_path, repo_root, postgres_database_url)
    adapter = FakeGitHubAdapter()
    orchestrator = await Orchestrator.create(config, github_adapter=adapter)
    try:
        report = await orchestrator.run_workflow(
            "pull_request_maintenance",
            pr_context=_build_pr_context(),
        )
        published = await orchestrator.publish_pull_request_results(
            report_path=Path(report.report_dir) / "report.json"
        )
    finally:
        await orchestrator.close()

    assert published.workflow_name == "pull_request_maintenance"
    assert published.metadata["publish_mode"] == "companion_pr"
    assert published.metadata["companion_pr_url"] == "https://example/pulls/24"
    assert published.metadata["review_comment_url"] == "https://example/reviews/11"
