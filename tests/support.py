from __future__ import annotations

from pathlib import Path

from core.config import DatabaseConfig


DEFAULT_POSTGRES_URL = "postgres://close_devs:close_devs@127.0.0.1:5432/close_devs"


def sqlite_database_config(tmp_path: Path) -> DatabaseConfig:
    return DatabaseConfig(
        backend="sqlite",
        url=f"sqlite://{(tmp_path / 'state' / 'agent_memory.db').resolve()}",
        url_env="DATABASE_URL",
        echo=False,
    )
