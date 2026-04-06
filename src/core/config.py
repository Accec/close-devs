from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
from typing import Any
import tomllib


DEFAULT_POSTGRES_URL = "postgres://close_devs:close_devs@127.0.0.1:5432/close_devs"
APP_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO_SCHEMES = ("http://", "https://", "ssh://", "git://", "file://")
REMOTE_SCP_PATTERN = re.compile(r"^[^/\s]+@[^:\s]+:.+")


@dataclass(slots=True)
class LLMConfig:
    provider: str = "mock"
    model: str = "close-devs-mock"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
    temperature: float = 0.0
    max_retries: int = 2
    system_prompt: str = (
        "You are the Close-Devs maintenance planning model. "
        "Produce concise repo maintenance rationale and suggestions."
    )


@dataclass(slots=True)
class StaticReviewConfig:
    max_complexity: int = 10
    ruff_command: str | None = "ruff check"
    mypy_command: str | None = "mypy"
    bandit_command: str | None = "bandit -q -r -f json"
    dependency_audit_command: str | None = "pip-audit --format json"
    language_adapters_enabled: list[str] = field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "kotlin",
        ]
    )
    tool_policy: str = "augment_only"
    dependency_audit_mode: str = "auto"
    unsupported_language_mode: str = "generic_review"


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
    publish_retry_count: int = 2


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
    sqlite_busy_timeout_ms: int = 5000


@dataclass(slots=True)
class EnvironmentConfig:
    enabled: bool = True
    scope: str = "all_analysis"
    install_mode: str = "auto_detect"
    install_fail_policy: str = "mark_degraded"
    python_executable: str = "python3"
    bootstrap_tools: bool = True
    git_auth_mode: str = "auto"
    git_https_token_env: str = "GIT_AUTH_TOKEN"
    git_https_username: str = "git"
    git_ssh_key_path: str | None = None
    git_ssh_key_path_env: str = "GIT_SSH_KEY_PATH"
    git_known_hosts_path: str | None = None
    git_known_hosts_path_env: str = "GIT_KNOWN_HOSTS_PATH"
    git_ssh_strict_host_key_checking: str = "accept-new"
    git_clone_timeout_seconds: int = 900
    dependency_sources_priority: list[str] = field(
        default_factory=lambda: [
            "src/requirements.txt",
            "requirements.txt",
            "requirements/base.txt",
            "src/requirements-dev.txt",
            "requirements-dev.txt",
            "requirements/dev.txt",
            "src/requirements-test.txt",
            "requirements-test.txt",
            "requirements/test.txt",
            "pyproject.toml:project.dependencies",
            "pyproject.toml:project.optional-dependencies.dev",
            "pyproject.toml:project.optional-dependencies.test",
            "pyproject.toml:poetry.dependencies",
            "poetry.lock",
            "pyproject.toml:pdm.dependencies",
            "pdm.lock",
            "pyproject.toml:uv.dependencies",
            "uv.lock",
        ]
    )


@dataclass(slots=True)
class AgentRuntimeConfig:
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    timeout_seconds: int | None = None
    system_prompt: str | None = None
    max_steps: int = 16
    max_tool_calls: int = 24
    max_wall_time_seconds: int = 600
    max_consecutive_failures: int = 3
    allowed_tools: list[str] = field(default_factory=list)
    max_budget_ceiling: int = 0
    allowed_tool_superset: list[str] = field(default_factory=list)
    safety_lock: bool = True


@dataclass(slots=True)
class SkillAgentConfig:
    baseline: str = "baseline"
    auto_upgrade: bool = True


@dataclass(slots=True)
class SkillsConfig:
    enabled: bool = True
    repo_root: Path = Path("config/skills")
    shadow_evaluation_enabled: bool = True
    min_shadow_runs: int = 5
    promotion_margin: float = 0.10
    static: SkillAgentConfig = field(default_factory=SkillAgentConfig)
    dynamic: SkillAgentConfig = field(default_factory=SkillAgentConfig)
    maintenance: SkillAgentConfig = field(default_factory=SkillAgentConfig)


@dataclass(slots=True)
class AgentsConfig:
    static: AgentRuntimeConfig = field(
        default_factory=lambda: AgentRuntimeConfig(
            max_steps=24,
            max_tool_calls=32,
            max_wall_time_seconds=900,
            max_consecutive_failures=3,
            allowed_tools=[],
        )
    )
    dynamic: AgentRuntimeConfig = field(
        default_factory=lambda: AgentRuntimeConfig(
            max_steps=32,
            max_tool_calls=40,
            max_wall_time_seconds=1200,
            max_consecutive_failures=4,
            allowed_tools=[],
        )
    )
    maintenance: AgentRuntimeConfig = field(
        default_factory=lambda: AgentRuntimeConfig(
            max_steps=40,
            max_tool_calls=48,
            max_wall_time_seconds=1500,
            max_consecutive_failures=4,
            allowed_tools=[],
        )
    )


@dataclass(slots=True)
class AppConfig:
    repo_root: Path
    state_dir: Path
    reports_dir: Path
    rules_path: Path
    repo_source: str = ""
    repo_ref: str | None = None
    include: list[str] = field(default_factory=lambda: ["*", "**/*"])
    exclude: list[str] = field(
        default_factory=lambda: [
            ".git/**",
            ".venv/**",
            "reports/**",
            "state/**",
            "__pycache__/**",
            ".pytest_cache/**",
            ".mypy_cache/**",
            "node_modules/**",
            "dist/**",
            "build/**",
            "target/**",
            ".gradle/**",
            "coverage/**",
        ]
    )
    scan_interval_minutes: int = 60
    log_level: str = "INFO"
    log_agent_activity: bool = True
    auto_apply_patch: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    static_review: StaticReviewConfig = field(default_factory=StaticReviewConfig)
    dynamic_debug: DynamicDebugConfig = field(default_factory=DynamicDebugConfig)
    github: GitHubRuntimeConfig = field(default_factory=GitHubRuntimeConfig)
    pr_workflow: PRWorkflowConfig = field(default_factory=PRWorkflowConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)

    def __post_init__(self) -> None:
        if not self.repo_source:
            self.repo_source = str(self.repo_root)

    @property
    def repo_is_remote(self) -> bool:
        return is_remote_repo_source(self.repo_source)


def _resolve_path(base_dir: Path, value: str | None, fallback: str) -> Path:
    raw = Path(value or fallback)
    if raw.is_absolute():
        return raw
    return (base_dir / raw).resolve()


def _resolve_optional_path_string(base_dir: Path, value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return str(_resolve_path(base_dir, value, value))


def _resolve_support_path(base_dir: Path, value: str | None, fallback: str) -> Path:
    raw = Path(value or fallback)
    if raw.is_absolute():
        return raw
    base_candidate = (base_dir / raw).resolve()
    if base_candidate.exists():
        return base_candidate
    app_candidate = (APP_ROOT / raw).resolve()
    if app_candidate.exists():
        return app_candidate
    return base_candidate


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def is_remote_repo_source(value: str | Path) -> bool:
    raw = str(value).strip()
    if not raw:
        return False
    if raw.startswith(REMOTE_REPO_SCHEMES):
        return True
    return bool(REMOTE_SCP_PATTERN.match(raw))


def _remote_placeholder_root(state_dir: Path, repo_source: str) -> Path:
    digest = hashlib.sha1(repo_source.encode("utf-8")).hexdigest()[:12]
    stem = repo_source.rstrip("/").rsplit("/", 1)[-1]
    stem = stem.removesuffix(".git") or "remote-repo"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-") or "remote-repo"
    return (state_dir / "remote_sources" / f"{safe_stem}-{digest}").resolve()


def _resolve_repo_source(
    *,
    base_dir: Path,
    state_dir: Path,
    configured_value: object,
    repo_override: str | Path | None,
) -> tuple[Path, str]:
    raw = str(repo_override).strip() if repo_override is not None else str(configured_value or ".").strip()
    if is_remote_repo_source(raw):
        return _remote_placeholder_root(state_dir, raw), raw
    raw_path = Path(raw)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (base_dir / raw_path).resolve()
    return resolved, str(resolved)


def default_api_key_env_for_provider(provider: str) -> str:
    mapping = {
        "openai": "OPENAI_API_KEY",
        "openai_compatible": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "ollama": "",
        "mock": "",
    }
    return mapping.get(provider, "OPENAI_API_KEY")


def default_base_url_for_provider(provider: str) -> str | None:
    if provider == "ollama":
        return "http://127.0.0.1:11434"
    if provider == "openai_compatible":
        return "https://api.openai.com/v1"
    return None


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


def _infer_database_backend(url: str, configured_backend: str) -> str:
    lowered = url.lower()
    if lowered.startswith("sqlite://"):
        return "sqlite"
    if lowered.startswith(("postgres://", "postgresql://")):
        return "postgres"
    return configured_backend


def load_rules(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(
    path: Path,
    repo_override: str | Path | None = None,
    repo_ref_override: str | None = None,
) -> AppConfig:
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
    environment_section = data.get("environment", {})
    agents_section = data.get("agents", {})
    skills_section = data.get("skills", {})

    base_dir = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    state_dir = _resolve_path(base_dir, app_section.get("state_dir"), "state")
    repo_root, repo_source = _resolve_repo_source(
        base_dir=base_dir,
        state_dir=state_dir,
        configured_value=app_section.get("repo_root"),
        repo_override=repo_override,
    )
    reports_dir = _resolve_path(base_dir, app_section.get("reports_dir"), "reports")
    rules_path = _resolve_support_path(base_dir, app_section.get("rules_path"), "config/rules.toml")
    repo_ref = repo_ref_override if repo_ref_override is not None else _optional_str(app_section.get("repo_ref"))
    database_backend = str(database_section.get("backend", "postgres"))
    database_url_env = str(database_section.get("url_env", "DATABASE_URL"))
    database_url = _resolve_database_url(
        backend=database_backend,
        configured_url=database_section.get("url"),
        url_env=database_url_env,
        state_dir=state_dir,
    )
    database_backend = _infer_database_backend(database_url, database_backend)
    skills_repo_root = _resolve_support_path(base_dir, skills_section.get("repo_root"), "config/skills")
    llm_provider = str(llm_section.get("provider", "mock"))

    static_defaults = StaticReviewConfig()

    return AppConfig(
        repo_root=repo_root,
        state_dir=state_dir,
        reports_dir=reports_dir,
        rules_path=rules_path,
        repo_source=repo_source,
        repo_ref=repo_ref,
        include=list(app_section.get("include", ["*", "**/*"])),
        exclude=list(
            app_section.get(
                "exclude",
                [
                    ".git/**",
                    ".venv/**",
                    "reports/**",
                    "state/**",
                    "__pycache__/**",
                    ".pytest_cache/**",
                    ".mypy_cache/**",
                    "node_modules/**",
                    "dist/**",
                    "build/**",
                    "target/**",
                    ".gradle/**",
                    "coverage/**",
                ],
            )
        ),
        scan_interval_minutes=int(app_section.get("scan_interval_minutes", 60)),
        log_level=str(app_section.get("log_level", "INFO")),
        log_agent_activity=bool(app_section.get("log_agent_activity", True)),
        auto_apply_patch=bool(app_section.get("auto_apply_patch", False)),
        llm=LLMConfig(
            provider=llm_provider,
            model=str(llm_section.get("model", "close-devs-mock")),
            base_url=_optional_str(llm_section.get("base_url")) or default_base_url_for_provider(llm_provider),
            api_key_env=_optional_str(llm_section.get("api_key_env")) or default_api_key_env_for_provider(llm_provider),
            timeout_seconds=int(llm_section.get("timeout_seconds", 30)),
            temperature=float(llm_section.get("temperature", 0.0)),
            max_retries=int(llm_section.get("max_retries", 2)),
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
            ruff_command=static_section.get("ruff_command", static_defaults.ruff_command),
            mypy_command=static_section.get("mypy_command", static_defaults.mypy_command),
            bandit_command=static_section.get("bandit_command", static_defaults.bandit_command),
            dependency_audit_command=static_section.get(
                "dependency_audit_command",
                static_defaults.dependency_audit_command,
            ),
            language_adapters_enabled=[
                str(item)
                for item in static_section.get(
                    "language_adapters_enabled",
                    static_defaults.language_adapters_enabled,
                )
            ],
            tool_policy=str(
                static_section.get("tool_policy", static_defaults.tool_policy)
            ),
            dependency_audit_mode=str(
                static_section.get(
                    "dependency_audit_mode",
                    static_defaults.dependency_audit_mode,
                )
            ),
            unsupported_language_mode=str(
                static_section.get(
                    "unsupported_language_mode",
                    static_defaults.unsupported_language_mode,
                )
            ),
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
            publish_retry_count=int(github_section.get("publish_retry_count", 2)),
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
            sqlite_busy_timeout_ms=int(
                database_section.get("sqlite_busy_timeout_ms", 5000)
            ),
        ),
        environment=EnvironmentConfig(
            enabled=bool(environment_section.get("enabled", True)),
            scope=str(environment_section.get("scope", "all_analysis")),
            install_mode=str(environment_section.get("install_mode", "auto_detect")),
            install_fail_policy=str(
                environment_section.get("install_fail_policy", "mark_degraded")
            ),
            python_executable=str(
                environment_section.get("python_executable", "python3")
            ),
            bootstrap_tools=bool(environment_section.get("bootstrap_tools", True)),
            git_auth_mode=str(environment_section.get("git_auth_mode", "auto")),
            git_https_token_env=str(
                environment_section.get("git_https_token_env", "GIT_AUTH_TOKEN")
            ),
            git_https_username=str(
                environment_section.get("git_https_username", "git")
            ),
            git_ssh_key_path=_resolve_optional_path_string(
                base_dir,
                _optional_str(environment_section.get("git_ssh_key_path")),
            ),
            git_ssh_key_path_env=str(
                environment_section.get("git_ssh_key_path_env", "GIT_SSH_KEY_PATH")
            ),
            git_known_hosts_path=_resolve_optional_path_string(
                base_dir,
                _optional_str(environment_section.get("git_known_hosts_path")),
            ),
            git_known_hosts_path_env=str(
                environment_section.get("git_known_hosts_path_env", "GIT_KNOWN_HOSTS_PATH")
            ),
            git_ssh_strict_host_key_checking=str(
                environment_section.get("git_ssh_strict_host_key_checking", "accept-new")
            ),
            git_clone_timeout_seconds=int(
                environment_section.get("git_clone_timeout_seconds", 900)
            ),
            dependency_sources_priority=[
                str(item)
                for item in environment_section.get(
                    "dependency_sources_priority",
                    [
                        "src/requirements.txt",
                        "requirements.txt",
                        "requirements-dev.txt",
                        "requirements-test.txt",
                        "pyproject.toml:project.dependencies",
                        "poetry.lock",
                        "pdm.lock",
                        "uv.lock",
                    ],
                )
            ],
        ),
        agents=AgentsConfig(
            static=AgentRuntimeConfig(
                provider=_optional_str(agents_section.get("static", {}).get("provider")),
                model=str(agents_section.get("static", {}).get("model"))
                if agents_section.get("static", {}).get("model") not in (None, "")
                else None,
                base_url=_optional_str(agents_section.get("static", {}).get("base_url")),
                api_key_env=_optional_str(agents_section.get("static", {}).get("api_key_env")),
                temperature=_optional_float(agents_section.get("static", {}).get("temperature")),
                timeout_seconds=_optional_int(agents_section.get("static", {}).get("timeout_seconds")),
                system_prompt=_optional_str(agents_section.get("static", {}).get("system_prompt")),
                max_steps=int(agents_section.get("static", {}).get("max_steps", 24)),
                max_tool_calls=int(agents_section.get("static", {}).get("max_tool_calls", 32)),
                max_wall_time_seconds=int(
                    agents_section.get("static", {}).get("max_wall_time_seconds", 900)
                ),
                max_consecutive_failures=int(
                    agents_section.get("static", {}).get("max_consecutive_failures", 3)
                ),
                allowed_tools=list(agents_section.get("static", {}).get("allowed_tools", [])),
                max_budget_ceiling=int(
                    agents_section.get("static", {}).get(
                        "max_budget_ceiling",
                        agents_section.get("static", {}).get("max_steps", 24),
                    )
                ),
                allowed_tool_superset=list(
                    agents_section.get("static", {}).get(
                        "allowed_tool_superset",
                        agents_section.get("static", {}).get("allowed_tools", []),
                    )
                ),
                safety_lock=bool(
                    agents_section.get("static", {}).get("safety_lock", True)
                ),
            ),
            dynamic=AgentRuntimeConfig(
                provider=_optional_str(agents_section.get("dynamic", {}).get("provider")),
                model=str(agents_section.get("dynamic", {}).get("model"))
                if agents_section.get("dynamic", {}).get("model") not in (None, "")
                else None,
                base_url=_optional_str(agents_section.get("dynamic", {}).get("base_url")),
                api_key_env=_optional_str(agents_section.get("dynamic", {}).get("api_key_env")),
                temperature=_optional_float(agents_section.get("dynamic", {}).get("temperature")),
                timeout_seconds=_optional_int(agents_section.get("dynamic", {}).get("timeout_seconds")),
                system_prompt=_optional_str(agents_section.get("dynamic", {}).get("system_prompt")),
                max_steps=int(agents_section.get("dynamic", {}).get("max_steps", 32)),
                max_tool_calls=int(agents_section.get("dynamic", {}).get("max_tool_calls", 40)),
                max_wall_time_seconds=int(
                    agents_section.get("dynamic", {}).get("max_wall_time_seconds", 1200)
                ),
                max_consecutive_failures=int(
                    agents_section.get("dynamic", {}).get("max_consecutive_failures", 4)
                ),
                allowed_tools=list(agents_section.get("dynamic", {}).get("allowed_tools", [])),
                max_budget_ceiling=int(
                    agents_section.get("dynamic", {}).get(
                        "max_budget_ceiling",
                        agents_section.get("dynamic", {}).get("max_steps", 32),
                    )
                ),
                allowed_tool_superset=list(
                    agents_section.get("dynamic", {}).get(
                        "allowed_tool_superset",
                        agents_section.get("dynamic", {}).get("allowed_tools", []),
                    )
                ),
                safety_lock=bool(
                    agents_section.get("dynamic", {}).get("safety_lock", True)
                ),
            ),
            maintenance=AgentRuntimeConfig(
                provider=_optional_str(agents_section.get("maintenance", {}).get("provider")),
                model=str(agents_section.get("maintenance", {}).get("model"))
                if agents_section.get("maintenance", {}).get("model") not in (None, "")
                else None,
                base_url=_optional_str(agents_section.get("maintenance", {}).get("base_url")),
                api_key_env=_optional_str(agents_section.get("maintenance", {}).get("api_key_env")),
                temperature=_optional_float(agents_section.get("maintenance", {}).get("temperature")),
                timeout_seconds=_optional_int(agents_section.get("maintenance", {}).get("timeout_seconds")),
                system_prompt=_optional_str(agents_section.get("maintenance", {}).get("system_prompt")),
                max_steps=int(agents_section.get("maintenance", {}).get("max_steps", 40)),
                max_tool_calls=int(agents_section.get("maintenance", {}).get("max_tool_calls", 48)),
                max_wall_time_seconds=int(
                    agents_section.get("maintenance", {}).get("max_wall_time_seconds", 1500)
                ),
                max_consecutive_failures=int(
                    agents_section.get("maintenance", {}).get("max_consecutive_failures", 4)
                ),
                allowed_tools=list(agents_section.get("maintenance", {}).get("allowed_tools", [])),
                max_budget_ceiling=int(
                    agents_section.get("maintenance", {}).get(
                        "max_budget_ceiling",
                        agents_section.get("maintenance", {}).get("max_steps", 40),
                    )
                ),
                allowed_tool_superset=list(
                    agents_section.get("maintenance", {}).get(
                        "allowed_tool_superset",
                        agents_section.get("maintenance", {}).get("allowed_tools", []),
                    )
                ),
                safety_lock=bool(
                    agents_section.get("maintenance", {}).get("safety_lock", True)
                ),
            ),
        ),
        skills=SkillsConfig(
            enabled=bool(skills_section.get("enabled", True)),
            repo_root=skills_repo_root,
            shadow_evaluation_enabled=bool(
                skills_section.get("shadow_evaluation_enabled", True)
            ),
            min_shadow_runs=int(skills_section.get("min_shadow_runs", 5)),
            promotion_margin=float(skills_section.get("promotion_margin", 0.10)),
            static=SkillAgentConfig(
                baseline=str(skills_section.get("static", {}).get("baseline", "baseline")),
                auto_upgrade=bool(skills_section.get("static", {}).get("auto_upgrade", True)),
            ),
            dynamic=SkillAgentConfig(
                baseline=str(skills_section.get("dynamic", {}).get("baseline", "baseline")),
                auto_upgrade=bool(skills_section.get("dynamic", {}).get("auto_upgrade", True)),
            ),
            maintenance=SkillAgentConfig(
                baseline=str(skills_section.get("maintenance", {}).get("baseline", "baseline")),
                auto_upgrade=bool(skills_section.get("maintenance", {}).get("auto_upgrade", True)),
            ),
        ),
    )
