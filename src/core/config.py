from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_POSTGRES_URL = "postgres://close_devs:close_devs@127.0.0.1:5432/close_devs"


@dataclass(slots=True)
class LLMConfig:
    provider: str = "mock"
    model: str = "close-devs-mock"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
    temperature: float = 0.0
    system_prompt: str = (
        "You are the Close-Devs maintenance planning model. "
        "Produce concise repo maintenance rationale and suggestions."
    )


@dataclass(slots=True)
class StaticReviewConfig:
    max_complexity: int = 10
    ruff_command: str | None = "ruff check"
    mypy_command: str | None = "mypy"
    bandit_command: str | None = "bandit -q -r"


@dataclass(slots=True)
class DynamicDebugConfig:
    smoke_commands: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=lambda: ["python3 -m pytest -q"])
    timeout_seconds: int = 120


@dataclass(slots=True)
class GitHubRuntimeConfig:
    provider: str = "github_actions"
    repo_full_name: str = ""
    base_branch: str = "main"
    token_env: str = "GITHUB_TOKEN"
    bot_branch_prefix: str = "close-devs/fix"
    companion_pr_label: str = "close-devs"
    review_mode: str = "review"
    artifact_retention_days: int = 7


@dataclass(slots=True)
class PRWorkflowConfig:
    inline_comment_limit: int = 5
    allow_companion_pr: bool = True
    safe_fix_only: bool = True
    issue_comment_trigger: str = "/close-devs rerun"


@dataclass(slots=True)
class DatabaseConfig:
    backend: str = "postgres"
    url: str = DEFAULT_POSTGRES_URL
    url_env: str = "DATABASE_URL"
    echo: bool = False


@dataclass(slots=True)
class AppConfig:
    repo_root: Path
    state_dir: Path
    reports_dir: Path
    rules_path: Path
    include: list[str] = field(default_factory=lambda: ["*.py", "**/*.py"])
    exclude: list[str] = field(default_factory=list)
    scan_interval_minutes: int = 60
    log_level: str = "INFO"
    auto_apply_patch: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    static_review: StaticReviewConfig = field(default_factory=StaticReviewConfig)
    dynamic_debug: DynamicDebugConfig = field(default_factory=DynamicDebugConfig)
    github: GitHubRuntimeConfig = field(default_factory=GitHubRuntimeConfig)
    pr_workflow: PRWorkflowConfig = field(default_factory=PRWorkflowConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


def _resolve_path(base_dir: Path, value: str | None, fallback: str) -> Path:
    raw = Path(value or fallback)
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def _default_database_url(backend: str, state_dir: Path) -> str:
    if backend == "sqlite":
        return f"sqlite://{(state_dir / 'agent_memory.db').resolve()}"
    return DEFAULT_POSTGRES_URL


def _resolve_database_url(
    *,
    backend: str,
    configured_url: str | None,
    url_env: str,
    state_dir: Path,
) -> str:
    env_url = os.getenv(url_env)
    if env_url:
        return env_url
    if configured_url:
        return configured_url
    return _default_database_url(backend, state_dir)


def load_rules(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(path: Path, repo_override: Path | None = None) -> AppConfig:
    config_path = path.resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    app_section = data.get("app", {})
    llm_section = data.get("llm", {})
    static_section = data.get("static_review", {})
    dynamic_section = data.get("dynamic_debug", {})
    github_section = data.get("github", {})
    pr_section = data.get("pr_workflow", {})
    database_section = data.get("database", {})

    base_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    repo_root = (
        repo_override.resolve()
        if repo_override is not None
        else _resolve_path(base_dir, app_section.get("repo_root"), ".")
    )
    state_dir = _resolve_path(base_dir, app_section.get("state_dir"), "state")
    reports_dir = _resolve_path(base_dir, app_section.get("reports_dir"), "reports")
    rules_path = _resolve_path(base_dir, app_section.get("rules_path"), "config/rules.toml")
    database_backend = str(database_section.get("backend", "postgres"))
    database_url_env = str(database_section.get("url_env", "DATABASE_URL"))
    database_url = _resolve_database_url(
        backend=database_backend,
        configured_url=database_section.get("url"),
        url_env=database_url_env,
        state_dir=state_dir,
    )

    return AppConfig(
        repo_root=repo_root,
        state_dir=state_dir,
        reports_dir=reports_dir,
        rules_path=rules_path,
        include=list(app_section.get("include", ["*.py", "**/*.py"])),
        exclude=list(app_section.get("exclude", [])),
        scan_interval_minutes=int(app_section.get("scan_interval_minutes", 60)),
        log_level=str(app_section.get("log_level", "INFO")),
        auto_apply_patch=bool(app_section.get("auto_apply_patch", False)),
        llm=LLMConfig(
            provider=str(llm_section.get("provider", "mock")),
            model=str(llm_section.get("model", "close-devs-mock")),
            base_url=(
                str(llm_section["base_url"])
                if llm_section.get("base_url") not in (None, "")
                else None
            ),
            api_key_env=str(llm_section.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=int(llm_section.get("timeout_seconds", 30)),
            temperature=float(llm_section.get("temperature", 0.0)),
            system_prompt=str(
                llm_section.get(
                    "system_prompt",
                    (
                        "You are the Close-Devs maintenance planning model. "
                        "Produce concise repo maintenance rationale and suggestions."
                    ),
                )
            ),
        ),
        static_review=StaticReviewConfig(
            max_complexity=int(static_section.get("max_complexity", 10)),
            ruff_command=static_section.get("ruff_command"),
            mypy_command=static_section.get("mypy_command"),
            bandit_command=static_section.get("bandit_command"),
        ),
        dynamic_debug=DynamicDebugConfig(
            smoke_commands=list(dynamic_section.get("smoke_commands", [])),
            test_commands=list(dynamic_section.get("test_commands", ["python3 -m pytest -q"])),
            timeout_seconds=int(dynamic_section.get("timeout_seconds", 120)),
        ),
        github=GitHubRuntimeConfig(
            provider=str(github_section.get("provider", "github_actions")),
            repo_full_name=str(github_section.get("repo_full_name", "")),
            base_branch=str(github_section.get("base_branch", "main")),
            token_env=str(github_section.get("token_env", "GITHUB_TOKEN")),
            bot_branch_prefix=str(
                github_section.get("bot_branch_prefix", "close-devs/fix")
            ),
            companion_pr_label=str(
                github_section.get("companion_pr_label", "close-devs")
            ),
            review_mode=str(github_section.get("review_mode", "review")),
            artifact_retention_days=int(
                github_section.get("artifact_retention_days", 7)
            ),
        ),
        pr_workflow=PRWorkflowConfig(
            inline_comment_limit=int(pr_section.get("inline_comment_limit", 5)),
            allow_companion_pr=bool(pr_section.get("allow_companion_pr", True)),
            safe_fix_only=bool(pr_section.get("safe_fix_only", True)),
            issue_comment_trigger=str(
                pr_section.get("issue_comment_trigger", "/close-devs rerun")
            ),
        ),
        database=DatabaseConfig(
            backend=database_backend,
            url=str(database_url),
            url_env=database_url_env,
            echo=bool(database_section.get("echo", False)),
        ),
    )
