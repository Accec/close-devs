from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tortoise import Tortoise
from tortoise.connection import connections

from core.config import DatabaseConfig, DEFAULT_POSTGRES_URL


def _default_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_POSTGRES_URL)


def resolve_database_url(database: DatabaseConfig) -> str:
    return os.environ.get(database.url_env, database.url)


def build_tortoise_config(database: DatabaseConfig) -> dict[str, Any]:
    return {
        "connections": {"default": resolve_database_url(database)},
        "apps": {
            "models": {
                "models": ["memory.orm_models", "aerich.models"],
                "default_connection": "default",
            }
        },
        "use_tz": True,
        "timezone": "UTC",
    }


async def init_database(database: DatabaseConfig, *, ensure_schema: bool = False) -> None:
    if getattr(Tortoise, "_inited", False):
        await close_database()

    database_url = resolve_database_url(database)
    if database.backend == "sqlite" and database_url.startswith("sqlite://"):
        sqlite_path = Path(database_url.removeprefix("sqlite://"))
        if sqlite_path.name != ":memory:":
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    await Tortoise.init(config=build_tortoise_config(database))
    connection = connections.get("default")
    await connection.execute_query("SELECT 1")
    if ensure_schema:
        await Tortoise.generate_schemas(safe=True)


async def close_database() -> None:
    if getattr(Tortoise, "_inited", False):
        await connections.close_all(discard=True)
        Tortoise.apps.clear()
        Tortoise._inited = False


TORTOISE_ORM = build_tortoise_config(DatabaseConfig(url=_default_database_url()))
