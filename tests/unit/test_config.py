from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config


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
model = "gpt-5.4"
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
    assert config.environment.enabled is True
    assert config.environment.scope == "all_analysis"
    assert config.environment.install_fail_policy == "mark_degraded"
    assert config.environment.bootstrap_tools is False
    assert config.log_agent_activity is False
    assert config.agents.static.model == "gpt-5.4"
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
