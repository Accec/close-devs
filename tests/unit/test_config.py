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

[database]
backend = "postgres"
url = "postgres://configured:configured@127.0.0.1:5432/configured"
url_env = "DATABASE_URL"
echo = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "postgres://env:env@127.0.0.1:5432/envdb")

    config = load_config(config_path)

    assert config.database.backend == "postgres"
    assert config.database.url == "postgres://env:env@127.0.0.1:5432/envdb"
    assert config.database.echo is True
