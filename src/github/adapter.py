from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shlex
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from core.config import GitHubRuntimeConfig
from core.models import ArtifactReference, CompanionPRPayload, GitHubCapabilities, PullRequestContext, ReviewPayload, WorkflowTrigger
from github.rendering import summary_comment_marker
from tools.command_runner import CommandRunner


class GitHubAdapter:
    def __init__(
        self,
        config: GitHubRuntimeConfig,
        logger: logging.Logger,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.command_runner = command_runner or CommandRunner()

    @property
    def token(self) -> str | None:
        return os.environ.get(self.config.token_env)

    async def resolve_capabilities(
        self,
        pr_context: PullRequestContext,
        allow_companion_pr: bool,
    ) -> GitHubCapabilities:
        reasons: list[str] = []
        if not self.token:
            return GitHubCapabilities(reasons=["missing-token"])

        try:
            repo = await self._request_json("GET", f"/repos/{pr_context.repo_full_name}")
        except Exception as exc:
            self.logger.warning("Unable to probe repository capabilities: %s", exc)
            return GitHubCapabilities(reasons=["repo-api-unavailable"])

        permissions = dict(repo.get("permissions") or {})
        can_comment = True
        can_inline_review = True
        push_allowed = permissions.get("push", True)
        if not allow_companion_pr:
            reasons.append("companion-pr-disabled")
        if pr_context.is_from_fork:
            reasons.append("fork-pr")
        if not push_allowed:
            reasons.append("insufficient-push-permission")

        can_push_fix_branch = allow_companion_pr and not pr_context.is_from_fork and push_allowed
        can_open_companion_pr = can_push_fix_branch
        return GitHubCapabilities(
            can_comment=can_comment,
            can_inline_review=can_inline_review,
            can_push_fix_branch=can_push_fix_branch,
            can_open_companion_pr=can_open_companion_pr,
            reasons=sorted(set(reasons)),
        )

    async def load_pr_context(
        self,
        event_path: Path,
        pr_number_override: int | None = None,
    ) -> PullRequestContext | None:
        payload = await asyncio.to_thread(self._read_json_file, event_path)
        repository = payload.get("repository", {})
        repo_full_name = str(
            repository.get("full_name") or self.config.repo_full_name or os.environ.get("GITHUB_REPOSITORY", "")
        )

        if "pull_request" in payload:
            pr_payload = payload["pull_request"]
            changed_files = await self._fetch_pr_changed_files(repo_full_name, int(pr_payload["number"]))
            return self._context_from_pull_request(
                pr_payload,
                repo_full_name=repo_full_name,
                changed_files=changed_files,
                trigger=WorkflowTrigger.PULL_REQUEST,
                event_name=str(os.environ.get("GITHUB_EVENT_NAME", "pull_request")),
                event_path=event_path,
                comment_body=None,
                actor=payload.get("sender", {}).get("login"),
            )

        issue = payload.get("issue")
        comment = payload.get("comment")
        if issue and issue.get("pull_request"):
            pr_number = int(pr_number_override or issue["number"])
            pr_payload = await self._request_json("GET", f"/repos/{repo_full_name}/pulls/{pr_number}")
            changed_files = await self._fetch_pr_changed_files(repo_full_name, pr_number)
            return self._context_from_pull_request(
                pr_payload,
                repo_full_name=repo_full_name,
                changed_files=changed_files,
                trigger=WorkflowTrigger.ISSUE_COMMENT,
                event_name=str(os.environ.get("GITHUB_EVENT_NAME", "issue_comment")),
                event_path=event_path,
                comment_body=comment.get("body") if comment else None,
                actor=payload.get("sender", {}).get("login"),
            )

        if pr_number_override is not None and repo_full_name:
            pr_payload = await self._request_json("GET", f"/repos/{repo_full_name}/pulls/{pr_number_override}")
            changed_files = await self._fetch_pr_changed_files(repo_full_name, pr_number_override)
            return self._context_from_pull_request(
                pr_payload,
                repo_full_name=repo_full_name,
                changed_files=changed_files,
                trigger=WorkflowTrigger.PULL_REQUEST,
                event_name="manual",
                event_path=event_path,
                comment_body=None,
                actor=os.environ.get("GITHUB_ACTOR"),
            )

        return None

    async def prepare_workspace(self, repo_root: Path, pr_context: PullRequestContext) -> Path:
        git_dir = repo_root / ".git"
        if not git_dir.exists():
            return repo_root

        current_sha = await self._git_stdout(repo_root, "rev-parse HEAD")
        if current_sha == pr_context.head_sha:
            return repo_root

        fetch_ref = f"pull/{pr_context.pr_number}/head"
        temp_branch = f"close-devs-pr-{pr_context.pr_number}"
        await self._git(repo_root, f"fetch origin {fetch_ref}")
        await self._git(repo_root, f"checkout -B {shlex.quote(temp_branch)} FETCH_HEAD")
        return repo_root

    async def find_existing_summary_comment(
        self,
        pr_context: PullRequestContext,
        marker: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.token:
            return None

        marker = marker or summary_comment_marker(pr_context)
        comments = await self._request_json(
            "GET",
            f"/repos/{pr_context.repo_full_name}/issues/{pr_context.pr_number}/comments",
            query={"per_page": "100"},
        )
        matched = [
            item for item in comments
            if marker in str(item.get("body", ""))
        ]
        if not matched:
            return None
        return max(matched, key=lambda item: int(item.get("id", 0)))

    async def create_or_update_summary_comment(
        self,
        pr_context: PullRequestContext,
        payload: ReviewPayload,
    ) -> dict[str, Any]:
        if not self.token:
            return {"status": "skipped", "reason": "missing-token"}

        marker = summary_comment_marker(pr_context)
        try:
            existing = await self.find_existing_summary_comment(pr_context, marker)
            if existing is None:
                response = await self._request_json(
                    "POST",
                    f"/repos/{pr_context.repo_full_name}/issues/{pr_context.pr_number}/comments",
                    data={"body": payload.body},
                )
            else:
                response = await self._request_json(
                    "PATCH",
                    f"/repos/{pr_context.repo_full_name}/issues/comments/{existing['id']}",
                    data={"body": payload.body},
                )
        except Exception as exc:
            self.logger.warning("Failed to create/update summary comment: %s", exc)
            return {"status": "failed", "error": str(exc)}

        return {
            "status": "published",
            "url": response.get("html_url"),
            "id": response.get("id"),
        }

    async def resolve_run_artifacts(
        self,
        pr_context: PullRequestContext,
        artifact_name: str,
        references: list[ArtifactReference],
    ) -> list[ArtifactReference]:
        workflow_url = self.workflow_run_url()
        run_id = os.environ.get("GITHUB_RUN_ID")
        if not self.token or not run_id:
            return [
                ArtifactReference(
                    name=reference.name,
                    path=reference.path,
                    url=reference.url,
                    fallback_url=reference.fallback_url or workflow_url,
                )
                for reference in references
            ]

        artifact_url: str | None = None
        try:
            response = await self._request_json(
                "GET",
                f"/repos/{pr_context.repo_full_name}/actions/runs/{run_id}/artifacts",
                query={"per_page": "100"},
            )
            artifacts = response.get("artifacts", []) if isinstance(response, dict) else []
            for artifact in artifacts:
                if str(artifact.get("name")) != artifact_name:
                    continue
                artifact_url = self._artifact_ui_url(
                    pr_context.repo_full_name,
                    str(run_id),
                    int(artifact["id"]),
                )
                break
        except Exception as exc:
            self.logger.warning("Failed to resolve workflow artifacts: %s", exc)

        return [
            ArtifactReference(
                name=reference.name,
                path=reference.path,
                url=reference.url or artifact_url,
                fallback_url=reference.fallback_url or workflow_url,
            )
            for reference in references
        ]

    async def find_existing_inline_comment_markers(self, pr_context: PullRequestContext) -> set[str]:
        if not self.token:
            return set()

        try:
            comments = await self._request_json(
                "GET",
                f"/repos/{pr_context.repo_full_name}/pulls/{pr_context.pr_number}/comments",
                query={"per_page": "100"},
            )
        except Exception as exc:
            self.logger.warning("Failed to load existing inline comments: %s", exc)
            return set()

        markers: set[str] = set()
        for item in comments:
            body = str(item.get("body", ""))
            markers.update(self._extract_markers(body, "<!-- close-devs:inline"))
        return markers

    async def publish_inline_comments(
        self,
        pr_context: PullRequestContext,
        inline_comments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.token:
            return {"status": "skipped", "reason": "missing-token", "published": 0}
        if not inline_comments:
            return {"status": "skipped", "reason": "no-inline-comments", "published": 0}

        published = 0
        errors: list[str] = []
        for item in inline_comments:
            try:
                await self._request_json(
                    "POST",
                    f"/repos/{pr_context.repo_full_name}/pulls/{pr_context.pr_number}/comments",
                    data={
                        "body": item["body"],
                        "commit_id": pr_context.head_sha,
                        "path": item["path"],
                        "line": item["line"],
                        "side": "RIGHT",
                    },
                )
                published += 1
            except Exception as exc:
                self.logger.warning("Failed to publish inline comment: %s", exc)
                errors.append(str(exc))

        return {
            "status": "published" if published else "failed",
            "published": published,
            "errors": errors,
        }

    async def publish_review(
        self,
        pr_context: PullRequestContext,
        payload: ReviewPayload,
    ) -> dict[str, Any]:
        return await self.create_or_update_summary_comment(pr_context, payload)

    async def publish_fix_branch(
        self,
        pr_context: PullRequestContext,
        repo_root: Path,
        branch_name: str,
        touched_files: list[str],
    ) -> dict[str, Any]:
        if not self.token:
            return {"status": "skipped", "reason": "missing-token"}
        if pr_context.is_from_fork:
            return {"status": "skipped", "reason": "fork-pr"}
        if not (repo_root / ".git").exists():
            return {"status": "skipped", "reason": "missing-git-dir"}

        try:
            await self._git(repo_root, f"checkout -B {shlex.quote(branch_name)}")
            await self._git(repo_root, 'config user.name "close-devs[bot]"')
            await self._git(repo_root, 'config user.email "close-devs[bot]@users.noreply.github.com"')
            quoted_files = " ".join(shlex.quote(path) for path in touched_files)
            await self._git(repo_root, f"add {quoted_files}")
            await self._git(
                repo_root,
                (
                    f'commit --allow-empty -m '
                    f'"[Close-Devs] Apply safe autofixes for PR #{pr_context.pr_number}"'
                ),
            )
            await self._git(
                repo_root,
                f"push --force-with-lease origin HEAD:{shlex.quote(branch_name)}",
            )
        except Exception as exc:
            self.logger.warning("Failed to publish fix branch: %s", exc)
            return {"status": "failed", "error": str(exc), "branch": branch_name}

        return {"status": "published", "branch": branch_name}

    async def create_or_update_companion_pr(
        self,
        pr_context: PullRequestContext,
        payload: CompanionPRPayload,
    ) -> dict[str, Any]:
        if not self.token:
            return {"status": "skipped", "reason": "missing-token"}

        owner = pr_context.repo_full_name.split("/", 1)[0]
        try:
            existing = await self._request_json(
                "GET",
                f"/repos/{pr_context.repo_full_name}/pulls",
                query={"state": "open", "head": f"{owner}:{payload.head_branch}"},
            )
            if existing:
                pr = existing[0]
                response = await self._request_json(
                    "PATCH",
                    f"/repos/{pr_context.repo_full_name}/pulls/{pr['number']}",
                    data={"title": payload.title, "body": payload.body, "base": payload.base_branch},
                )
                number = int(response["number"])
            else:
                response = await self._request_json(
                    "POST",
                    f"/repos/{pr_context.repo_full_name}/pulls",
                    data={
                        "title": payload.title,
                        "head": payload.head_branch,
                        "base": payload.base_branch,
                        "body": payload.body,
                    },
                )
                number = int(response["number"])
            if payload.labels:
                await self._request_json(
                    "POST",
                    f"/repos/{pr_context.repo_full_name}/issues/{number}/labels",
                    data={"labels": payload.labels},
                )
        except Exception as exc:
            self.logger.warning("Failed to create/update companion PR: %s", exc)
            return {"status": "failed", "error": str(exc)}

        return {
            "status": "published",
            "number": number,
            "url": response.get("html_url"),
        }

    def workflow_run_url(self) -> str | None:
        server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo = os.environ.get("GITHUB_REPOSITORY") or self.config.repo_full_name
        run_id = os.environ.get("GITHUB_RUN_ID")
        if not repo or not run_id:
            return None
        return f"{server_url}/{repo}/actions/runs/{run_id}"

    async def write_step_summary(self, markdown: str) -> None:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return
        await asyncio.to_thread(self._append_text, Path(summary_path), markdown)

    def _append_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)

    def _artifact_ui_url(self, repo_full_name: str, run_id: str, artifact_id: int) -> str:
        server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        return f"{server_url}/{repo_full_name}/actions/runs/{run_id}/artifacts/{artifact_id}"

    def _extract_markers(self, body: str, prefix: str) -> set[str]:
        markers: set[str] = set()
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix) and stripped.endswith("-->"):
                markers.add(stripped)
        return markers

    def _context_from_pull_request(
        self,
        pr_payload: dict[str, Any],
        *,
        repo_full_name: str,
        changed_files: list[str],
        trigger: WorkflowTrigger,
        event_name: str,
        event_path: Path,
        comment_body: str | None,
        actor: str | None,
    ) -> PullRequestContext:
        base = pr_payload["base"]
        head = pr_payload["head"]
        return PullRequestContext(
            repo_full_name=repo_full_name,
            base_repo_full_name=base["repo"]["full_name"],
            head_repo_full_name=head["repo"]["full_name"],
            pr_number=int(pr_payload["number"]),
            title=str(pr_payload.get("title", "")),
            html_url=str(pr_payload.get("html_url", "")),
            base_branch=str(base["ref"]),
            head_branch=str(head["ref"]),
            head_sha=str(head["sha"]),
            changed_files=changed_files,
            trigger=trigger,
            event_name=event_name,
            actor=actor,
            comment_body=comment_body,
            event_path=str(event_path),
        )

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    async def _fetch_pr_changed_files(self, repo_full_name: str, pr_number: int) -> list[str]:
        if not self.token:
            return []
        try:
            files = await self._request_json(
                "GET",
                f"/repos/{repo_full_name}/pulls/{pr_number}/files",
                query={"per_page": "100"},
            )
        except Exception:
            return []
        return [str(item["filename"]) for item in files]

    async def _git(self, repo_root: Path, args: str) -> None:
        command = f"git -C {shlex.quote(str(repo_root))} {args}"
        result = await self.command_runner.run(command, cwd=repo_root)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or f"git command failed: {args}")

    async def _git_stdout(self, repo_root: Path, args: str) -> str:
        command = f"git -C {shlex.quote(str(repo_root))} {args}"
        result = await self.command_runner.run(command, cwd=repo_root)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._request_json_sync, method, path, data, query)

    def _request_json_sync(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None,
        query: dict[str, str] | None,
    ) -> Any:
        url = f"https://api.github.com{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        body = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "close-devs",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {details}") from exc

        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))
