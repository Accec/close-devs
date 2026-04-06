from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_config_parses_database_section_and_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"
log_agent_activity = false

[database]
backend = "postgres"
url = "postgres://configured:configured@127.0.0.1:5432/configured"
url_env = "DATABASE_URL"
echo = true

[environment]
enabled = true
scope = "all_analysis"
install_mode = "auto_detect"
install_fail_policy = "mark_degraded"
python_executable = "python3"
bootstrap_tools = false

[agents.static]
provider = "google_genai"
model = "gpt-5.4"
api_key_env = "GOOGLE_API_KEY"
temperature = 0.2
timeout_seconds = 99
system_prompt = "static-prompt"
max_steps = 24
max_tool_calls = 32
max_wall_time_seconds = 900
max_consecutive_failures = 3
allowed_tools = ["read_file", "run_static_review"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "postgres://env:env@127.0.0.1:5432/envdb")

    config = load_config(config_path)

    assert config.database.backend == "postgres"
    assert config.database.url == "postgres://env:env@127.0.0.1:5432/envdb"
    assert config.database.echo is True
    assert config.database.sqlite_busy_timeout_ms == 5000
    assert config.environment.enabled is True
    assert config.environment.scope == "all_analysis"
    assert config.environment.install_fail_policy == "mark_degraded"
    assert config.environment.bootstrap_tools is False
    assert config.environment.git_auth_mode == "auto"
    assert config.environment.git_https_token_env == "GIT_AUTH_TOKEN"
    assert config.environment.git_https_username == "git"
    assert config.environment.git_ssh_key_path is None
    assert config.environment.git_ssh_key_path_env == "GIT_SSH_KEY_PATH"
    assert config.environment.git_known_hosts_path is None
    assert config.environment.git_known_hosts_path_env == "GIT_KNOWN_HOSTS_PATH"
    assert config.environment.git_ssh_strict_host_key_checking == "accept-new"
    assert config.environment.git_clone_timeout_seconds == 900
    assert config.environment.dependency_sources_priority[0] == "src/requirements.txt"
    assert config.log_agent_activity is False
    assert config.llm.max_retries == 2
    assert config.llm.provider == "mock"
    assert config.static_review.bandit_command == "bandit -q -r -f json"
    assert config.static_review.dependency_audit_command == "pip-audit --format json"
    assert config.static_review.language_adapters_enabled[:3] == ["python", "javascript", "typescript"]
    assert config.static_review.tool_policy == "augment_only"
    assert config.static_review.dependency_audit_mode == "auto"
    assert config.static_review.unsupported_language_mode == "generic_review"
    assert config.agents.static.model == "gpt-5.4"
    assert config.agents.static.provider == "google_genai"
    assert config.agents.static.api_key_env == "GOOGLE_API_KEY"
    assert config.agents.static.temperature == 0.2
    assert config.agents.static.timeout_seconds == 99
    assert config.agents.static.system_prompt == "static-prompt"
    assert config.agents.static.max_steps == 24
    assert config.agents.static.allowed_tools == ["read_file", "run_static_review"]


def test_load_config_infers_sqlite_backend_from_env_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"

[database]
backend = "postgres"
url = "postgres://configured:configured@127.0.0.1:5432/configured"
url_env = "DATABASE_URL"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite://{(tmp_path / 'state' / 'agent.db').resolve()}")

    config = load_config(config_path)

    assert config.database.backend == "sqlite"
    assert config.database.url.startswith("sqlite://")


def test_load_config_parses_publish_retry_and_dependency_priority(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"

[github]
publish_retry_count = 4

[environment]
dependency_sources_priority = ["uv.lock", "requirements.txt"]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.github.publish_retry_count == 4
    assert config.environment.dependency_sources_priority == ["uv.lock", "requirements.txt"]


def test_load_config_parses_multilanguage_static_review_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"

[static_review]
language_adapters_enabled = ["python", "go", "rust"]
tool_policy = "augment_only"
dependency_audit_mode = "auto"
unsupported_language_mode = "generic_review"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.include == ["*", "**/*"]
    assert "node_modules/**" in config.exclude
    assert config.static_review.language_adapters_enabled == ["python", "go", "rust"]
    assert config.static_review.tool_policy == "augment_only"
    assert config.static_review.dependency_audit_mode == "auto"
    assert config.static_review.unsupported_language_mode == "generic_review"


def test_load_config_applies_provider_specific_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"

[llm]
provider = "anthropic"
model = "claude-sonnet-4"

[agents.dynamic]
provider = "ollama"
model = "qwen3:14b"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.provider == "anthropic"
    assert config.llm.api_key_env == "ANTHROPIC_API_KEY"
    assert config.llm.base_url is None
    assert config.agents.dynamic.provider == "ollama"
    assert config.agents.dynamic.base_url is None


def test_load_config_resolves_support_paths_against_app_root_for_external_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "external-config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "config/rules.toml"

[skills]
repo_root = "config/skills"
""".strip(),
        encoding="utf-8",
    )
    repo_override = tmp_path / "target-repo"
    repo_override.mkdir()

    config = load_config(config_path, repo_override=repo_override)

    assert config.rules_path == (PROJECT_ROOT / "config" / "rules.toml").resolve()
    assert config.skills.repo_root == (PROJECT_ROOT / "config" / "skills").resolve()


def test_load_config_supports_remote_repo_override_and_ref(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(
        config_path,
        repo_override="https://github.com/example/demo.git",
        repo_ref_override="main",
    )

    assert config.repo_is_remote is True
    assert config.repo_source == "https://github.com/example/demo.git"
    assert config.repo_ref == "main"
    assert "remote_sources" in str(config.repo_root)


def test_load_config_parses_remote_git_auth_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    key_path = tmp_path / "keys" / "id_ed25519"
    known_hosts_path = tmp_path / "ssh" / "known_hosts"
    config_path.write_text(
        f"""
[app]
repo_root = "."
state_dir = "state"
reports_dir = "reports"
rules_path = "rules.toml"

[environment]
git_auth_mode = "ssh_key"
git_https_token_env = "CUSTOM_GIT_TOKEN"
git_https_username = "oauth2"
git_ssh_key_path = "{key_path}"
git_ssh_key_path_env = "CUSTOM_GIT_SSH_KEY"
git_known_hosts_path = "{known_hosts_path}"
git_known_hosts_path_env = "CUSTOM_KNOWN_HOSTS"
git_ssh_strict_host_key_checking = "yes"
git_clone_timeout_seconds = 321
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.environment.git_auth_mode == "ssh_key"
    assert config.environment.git_https_token_env == "CUSTOM_GIT_TOKEN"
    assert config.environment.git_https_username == "oauth2"
    assert config.environment.git_ssh_key_path == str(key_path.resolve())
    assert config.environment.git_ssh_key_path_env == "CUSTOM_GIT_SSH_KEY"
    assert config.environment.git_known_hosts_path == str(known_hosts_path.resolve())
    assert config.environment.git_known_hosts_path_env == "CUSTOM_KNOWN_HOSTS"
    assert config.environment.git_ssh_strict_host_key_checking == "yes"
    assert config.environment.git_clone_timeout_seconds == 321
