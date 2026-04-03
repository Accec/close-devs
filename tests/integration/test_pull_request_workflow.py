from __future__ import annotations

from pathlib import Path
import sys

import pytest

from core.config import (
    AppConfig,
    DynamicDebugConfig,
    GitHubRuntimeConfig,
    LLMConfig,
    PRWorkflowConfig,
    StaticReviewConfig,
)
from core.models import ArtifactReference, GitHubCapabilities, PullRequestContext, WorkflowTrigger
from core.orchestrator import Orchestrator
from tests.support import sqlite_database_config


class FakeGitHubAdapter:
    def __init__(self, *, token: str | None = "token", workflow_url: str | None = "https://example/actions/1") -> None:
        self.token = token
        self.workflow_url_value = workflow_url
        self.summary_comments: list[object] = []
        self.inline_comments: list[dict[str, object]] = []
        self.fix_branches: list[tuple[str, list[str]]] = []
        self.companion_prs: list[object] = []
        self.step_summaries: list[str] = []
        self.existing_inline_markers: set[str] = set()
        self.summary_comment_id = 800

    async def resolve_capabilities(
        self,
        pr_context: PullRequestContext,
        allow_companion_pr: bool,
    ) -> GitHubCapabilities:
        if not self.token:
            return GitHubCapabilities(reasons=["missing-token"])
        reasons: list[str] = []
        can_push = allow_companion_pr and not pr_context.is_from_fork
        if pr_context.is_from_fork:
            reasons.append("fork-pr")
        if not allow_companion_pr:
            reasons.append("companion-pr-disabled")
        return GitHubCapabilities(
            can_comment=True,
            can_inline_review=True,
            can_push_fix_branch=can_push,
            can_open_companion_pr=can_push,
            reasons=reasons,
        )

    async def load_pr_context(self, event_path: Path, pr_number_override: int | None = None) -> PullRequestContext | None:
        raise AssertionError("load_pr_context should not be called in these tests")

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
                fallback_url=self.workflow_url(),
            )
            for reference in references
        ]

    async def create_or_update_summary_comment(
        self,
        pr_context: PullRequestContext,
        payload: object,
    ) -> dict[str, object]:
        self.summary_comments.append(payload)
        if not self.token:
            return {"status": "skipped", "reason": "missing-token"}
        return {
            "status": "published",
            "id": self.summary_comment_id,
            "url": f"https://example/comments/{self.summary_comment_id}",
        }

    async def find_existing_inline_comment_markers(self, pr_context: PullRequestContext) -> set[str]:
        return set(self.existing_inline_markers)

    async def publish_inline_comments(
        self,
        pr_context: PullRequestContext,
        inline_comments: list[dict[str, object]],
    ) -> dict[str, object]:
        self.inline_comments.extend(inline_comments)
        self.existing_inline_markers.update(
            str(item["marker"]) for item in inline_comments if "marker" in item
        )
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
        return {"status": "published", "number": 42, "url": "https://example/pulls/42"}

    def workflow_url(self) -> str | None:
        return self.workflow_url_value

    def workflow_run_url(self) -> str | None:
        return self.workflow_url_value

    async def write_step_summary(self, markdown: str) -> None:
        self.step_summaries.append(markdown)


def _build_config(tmp_path: Path, repo_root: Path, *, test_commands: list[str] | None = None) -> AppConfig:
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
            test_commands=test_commands or [],
            timeout_seconds=30,
        ),
        github=GitHubRuntimeConfig(
            repo_full_name="acme/repo",
            token_env="GITHUB_TOKEN",
            review_mode="review",
        ),
        pr_workflow=PRWorkflowConfig(
            inline_comment_limit=5,
            allow_companion_pr=True,
            safe_fix_only=True,
            issue_comment_trigger="/close-devs rerun",
        ),
        database=sqlite_database_config(tmp_path),
    )


def _build_pr_context(*, fork: bool = False) -> PullRequestContext:
    head_repo = "fork/repo" if fork else "acme/repo"
    return PullRequestContext(
        repo_full_name="acme/repo",
        base_repo_full_name="acme/repo",
        head_repo_full_name=head_repo,
        pr_number=7,
        title="Fix whitespace",
        html_url="https://github.com/acme/repo/pull/7",
        base_branch="main",
        head_branch="feature/fix",
        head_sha="abc123",
        changed_files=["module.py"],
        trigger=WorkflowTrigger.PULL_REQUEST,
        actor="alice",
    )


async def _run_pr_workflow(config: AppConfig, adapter: FakeGitHubAdapter, pr_context: PullRequestContext):
    orchestrator = await Orchestrator.create(config, github_adapter=adapter, ensure_schema=True)
    try:
        report = await orchestrator.run_workflow("pull_request_maintenance", pr_context=pr_context)
        published = await orchestrator.publish_pull_request_results(
            report_path=Path(report.report_dir) / "report.json"
        )
        return report, published
    finally:
        await orchestrator.close()


@pytest.mark.asyncio
async def test_pull_request_workflow_publishes_companion_pr_for_safe_patch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    config = _build_config(tmp_path, repo_root)
    adapter = FakeGitHubAdapter(token="token")

    report, published = await _run_pr_workflow(config, adapter, _build_pr_context())

    assert report.workflow_name == "pull_request_maintenance"
    assert report.metadata["publish_context_path"]
    assert published.metadata["publish_mode"] == "companion_pr"
    assert published.metadata["review_comment_url"] == "https://example/comments/800"
    assert published.metadata["companion_pr_url"] == "https://example/pulls/42"
    assert published.metadata["artifact_urls"]["summary_md"] == "https://example/artifacts/close-devs-report-" + report.run_id
    assert adapter.fix_branches
    assert adapter.companion_prs
    assert adapter.summary_comments
    assert adapter.inline_comments
    assert adapter.step_summaries
    assert (Path(report.report_dir) / "summary.md").exists()


@pytest.mark.asyncio
async def test_pull_request_workflow_degrades_to_comment_only_for_fork(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    config = _build_config(tmp_path, repo_root)
    adapter = FakeGitHubAdapter(token="token")

    _report, published = await _run_pr_workflow(config, adapter, _build_pr_context(fork=True))

    assert published.metadata["publish_mode"] == "comment_only"
    assert published.metadata["review_comment_url"] == "https://example/comments/800"
    assert not published.metadata.get("companion_pr_url")
    assert adapter.fix_branches == []
    assert adapter.companion_prs == []


@pytest.mark.asyncio
async def test_pull_request_workflow_degrades_to_artifact_only_without_token(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    config = _build_config(tmp_path, repo_root)
    adapter = FakeGitHubAdapter(token=None)

    _report, published = await _run_pr_workflow(config, adapter, _build_pr_context())

    assert published.metadata["publish_mode"] == "artifact_only"
    assert not published.metadata.get("review_comment_url")
    assert adapter.fix_branches == []
    assert adapter.companion_prs == []
    assert adapter.summary_comments == []


@pytest.mark.asyncio
async def test_pull_request_workflow_skips_companion_pr_when_validation_regresses(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    config = _build_config(
        tmp_path,
        repo_root,
        test_commands=[
            (
                f"{sys.executable} -c "
                "\"from pathlib import Path; import sys; "
                "text = Path('module.py').read_text(encoding='utf-8'); "
                "sys.exit(1 if 'Maintained by Close-Devs.' in text else 0)\""
            )
        ],
    )
    adapter = FakeGitHubAdapter(token="token")

    _report, published = await _run_pr_workflow(config, adapter, _build_pr_context())

    assert published.metadata["publish_mode"] == "comment_only"
    assert adapter.fix_branches == []
    assert adapter.companion_prs == []
    assert "validation_regressed_findings" in published.metadata["publish_reasons"]


@pytest.mark.asyncio
async def test_pull_request_publish_updates_same_summary_comment_on_repeat(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text("def add(a, b):    \n    return a + b\n\n", encoding="utf-8")
    config = _build_config(tmp_path, repo_root)
    adapter = FakeGitHubAdapter(token="token")

    orchestrator = await Orchestrator.create(config, github_adapter=adapter, ensure_schema=True)
    try:
        report = await orchestrator.run_workflow("pull_request_maintenance", pr_context=_build_pr_context())
        first = await orchestrator.publish_pull_request_results(report_path=Path(report.report_dir) / "report.json")
        second = await orchestrator.publish_pull_request_results(report_path=Path(report.report_dir) / "report.json")
    finally:
        await orchestrator.close()

    assert first.metadata["review_comment_id"] == 800
    assert second.metadata["review_comment_id"] == 800
    assert len(adapter.summary_comments) == 2
