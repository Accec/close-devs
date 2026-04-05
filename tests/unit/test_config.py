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
