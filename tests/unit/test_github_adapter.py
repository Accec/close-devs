from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from core.config import GitHubRuntimeConfig
from core.models import ArtifactReference, CompanionPRPayload, ReviewPayload, WorkflowTrigger
from github.adapter import GitHubAdapter
from github.rendering import summary_comment_marker


class StubGitHubAdapter(GitHubAdapter):
    def __init__(
        self,
        config: GitHubRuntimeConfig,
        *,
        changed_files: list[str] | None = None,
        repo_permissions: dict[str, bool] | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(config=config, logger=logging.getLogger("test"))
        self.changed_files = changed_files or []
        self.repo_permissions = repo_permissions or {"push": True}
        self.issue_comments = issue_comments or []
        self.artifacts = artifacts or []
        self.request_log: list[tuple[str, str]] = []

    async def _fetch_pr_changed_files(self, repo_full_name: str, pr_number: int) -> list[str]:
        return list(self.changed_files)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        self.request_log.append((method, path))
        if method == "GET" and path == "/repos/acme/repo":
            return {"permissions": self.repo_permissions}
        if method == "GET" and path.endswith("/pulls/12"):
            return {
                "number": 12,
                "title": "Fix tests",
                "html_url": "https://github.com/acme/repo/pull/12",
                "base": {"ref": "main", "repo": {"full_name": "acme/repo"}},
                "head": {
                    "ref": "feature-branch",
                    "sha": "abc123",
                    "repo": {"full_name": "acme/repo"},
                },
            }
        if method == "GET" and path.endswith("/pulls"):
            return [{"number": 99, "html_url": "https://github.com/acme/repo/pull/99"}]
        if method == "GET" and path.endswith("/issues/7/comments"):
            return list(self.issue_comments)
        if method == "PATCH" and path.endswith("/issues/comments/101"):
            return {"id": 101, "html_url": "https://github.com/acme/repo/issues/7#issuecomment-101"}
        if method == "POST" and path.endswith("/issues/7/comments"):
            return {"id": 201, "html_url": "https://github.com/acme/repo/issues/7#issuecomment-201"}
        if method == "GET" and path.endswith("/artifacts"):
            return {"artifacts": list(self.artifacts)}
        if method == "PATCH" and path.endswith("/pulls/99"):
            return {"number": 99, "html_url": "https://github.com/acme/repo/pull/99"}
        if method == "POST" and path.endswith("/labels"):
            return {}
        raise AssertionError(f"Unexpected request: {method} {path}")


@pytest.mark.asyncio
async def test_load_pull_request_context_from_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event_path = tmp_path / "pull_request.json"
    event_path.write_text(
        json.dumps(
            {
                "repository": {"full_name": "acme/repo"},
                "sender": {"login": "alice"},
                "pull_request": {
                    "number": 7,
                    "title": "Refactor module",
                    "html_url": "https://github.com/acme/repo/pull/7",
                    "base": {"ref": "main", "repo": {"full_name": "acme/repo"}},
                    "head": {
                        "ref": "feature/refactor",
                        "sha": "deadbeef",
                        "repo": {"full_name": "acme/repo"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")

    adapter = StubGitHubAdapter(
        GitHubRuntimeConfig(repo_full_name="acme/repo"),
        changed_files=["src/module.py"],
    )
    context = await adapter.load_pr_context(event_path)

    assert context is not None
    assert context.repo_full_name == "acme/repo"
    assert context.pr_number == 7
    assert context.changed_files == ["src/module.py"]
    assert context.trigger is WorkflowTrigger.PULL_REQUEST
    assert context.actor == "alice"


@pytest.mark.asyncio
async def test_load_issue_comment_context_from_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event_path = tmp_path / "issue_comment.json"
    event_path.write_text(
        json.dumps(
            {
                "repository": {"full_name": "acme/repo"},
                "sender": {"login": "bob"},
                "issue": {"number": 12, "pull_request": {"url": "https://api.github.com/repos/acme/repo/pulls/12"}},
                "comment": {"body": "/close-devs rerun"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")

    adapter = StubGitHubAdapter(
        GitHubRuntimeConfig(repo_full_name="acme/repo"),
        changed_files=["pkg/app.py"],
    )
    context = await adapter.load_pr_context(event_path)

    assert context is not None
    assert context.pr_number == 12
    assert context.changed_files == ["pkg/app.py"]
    assert context.trigger is WorkflowTrigger.ISSUE_COMMENT
    assert context.comment_body == "/close-devs rerun"
    assert context.actor == "bob"


@pytest.mark.asyncio
async def test_resolve_capabilities_degrades_without_push_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    adapter = StubGitHubAdapter(
        GitHubRuntimeConfig(repo_full_name="acme/repo", token_env="GITHUB_TOKEN"),
        repo_permissions={"push": False},
    )
    pr_context = adapter._context_from_pull_request(
        {
            "number": 7,
            "title": "Main PR",
            "html_url": "https://github.com/acme/repo/pull/7",
            "base": {"ref": "main", "repo": {"full_name": "acme/repo"}},
            "head": {"ref": "feature", "sha": "abc123", "repo": {"full_name": "acme/repo"}},
        },
        repo_full_name="acme/repo",
        changed_files=["module.py"],
        trigger=WorkflowTrigger.PULL_REQUEST,
        event_name="pull_request",
        event_path=Path("event.json"),
        comment_body=None,
        actor="alice",
    )

    capabilities = await adapter.resolve_capabilities(pr_context, allow_companion_pr=True)

    assert capabilities.can_comment is True
    assert capabilities.can_push_fix_branch is False
    assert capabilities.can_open_companion_pr is False
    assert "insufficient-push-permission" in capabilities.reasons


@pytest.mark.asyncio
async def test_create_or_update_summary_comment_updates_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    adapter = StubGitHubAdapter(
        GitHubRuntimeConfig(repo_full_name="acme/repo", token_env="GITHUB_TOKEN"),
        issue_comments=[
            {
                "id": 101,
                "body": f"{summary_comment_marker(_sample_pr_context())}\nold body",
                "html_url": "https://github.com/acme/repo/issues/7#issuecomment-101",
            }
        ],
    )

    result = await adapter.create_or_update_summary_comment(
        _sample_pr_context(),
        ReviewPayload(
            title="summary",
            body=f"{summary_comment_marker(_sample_pr_context())}\nnew body",
            summary="summary",
            publish_mode=_sample_publish_mode(),
        ),
    )

    assert result["status"] == "published"
    assert result["id"] == 101
    assert ("PATCH", "/repos/acme/repo/issues/comments/101") in adapter.request_log
    assert ("POST", "/repos/acme/repo/issues/7/comments") not in adapter.request_log


@pytest.mark.asyncio
async def test_resolve_run_artifacts_builds_ui_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_RUN_ID", "321")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    adapter = StubGitHubAdapter(
        GitHubRuntimeConfig(repo_full_name="acme/repo", token_env="GITHUB_TOKEN"),
        artifacts=[{"id": 55, "name": "close-devs-report-run-1"}],
    )

    references = await adapter.resolve_run_artifacts(
        _sample_pr_context(),
        "close-devs-report-run-1",
        [
            ArtifactReference(name="summary_md", path="summary.md"),
            ArtifactReference(name="workflow_run", path="Actions run", url="https://github.com/acme/repo/actions/runs/321"),
        ],
    )

    assert references[0].url == "https://github.com/acme/repo/actions/runs/321/artifacts/55"
    assert references[1].url == "https://github.com/acme/repo/actions/runs/321"


@pytest.mark.asyncio
async def test_create_or_update_companion_pr_updates_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    adapter = StubGitHubAdapter(GitHubRuntimeConfig(repo_full_name="acme/repo", token_env="GITHUB_TOKEN"))

    result = await adapter.create_or_update_companion_pr(
        pr_context=_sample_pr_context(),
        payload=CompanionPRPayload(
            head_branch="close-devs/fix/7/abcd1234",
            base_branch="main",
            title="[Close-Devs] Safe autofixes for PR #7",
            body="Safe fixes",
            labels=["close-devs"],
        ),
    )

    assert result["status"] == "published"
    assert result["number"] == 99
    assert ("PATCH", "/repos/acme/repo/pulls/99") in adapter.request_log
    assert ("POST", "/repos/acme/repo/pulls") not in adapter.request_log


def _sample_pr_context():
    adapter = StubGitHubAdapter(GitHubRuntimeConfig(repo_full_name="acme/repo"))
    return adapter._context_from_pull_request(
        {
            "number": 7,
            "title": "Main PR",
            "html_url": "https://github.com/acme/repo/pull/7",
            "base": {"ref": "main", "repo": {"full_name": "acme/repo"}},
            "head": {"ref": "feature", "sha": "abc123", "repo": {"full_name": "acme/repo"}},
        },
        repo_full_name="acme/repo",
        changed_files=["module.py"],
        trigger=WorkflowTrigger.PULL_REQUEST,
        event_name="pull_request",
        event_path=Path("event.json"),
        comment_body=None,
        actor="alice",
    )


def _sample_publish_mode():
    from core.models import PublishMode

    return PublishMode.COMMENT_ONLY
