from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tortoise import Tortoise
from tortoise.backends.base.executor import EXECUTOR_CACHE
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


async def _table_exists(table_name: str, *, backend: str) -> bool:
    connection = connections.get("default")
    if backend == "postgres":
        _, rows = await connection.execute_query(
            "SELECT tablename FROM pg_catalog.pg_tables "
            f"WHERE schemaname = current_schema() AND tablename = '{table_name}'",
        )
        return bool(rows)
    _, rows = await connection.execute_query(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        [table_name],
    )
    return bool(rows)


async def _validate_runtime_schema(database: DatabaseConfig) -> None:
    missing_tables = [
        table_name
        for table_name in (
            "aerich",
            "runs",
            "agent_sessions",
            "agent_steps",
            "agent_handoffs",
            "tool_calls",
            "skill_pack_records",
            "skill_binding_records",
            "skill_candidate_records",
            "skill_evaluation_records",
            "agent_reflection_records",
        )
        if not await _table_exists(table_name, backend=database.backend)
    ]
    if not missing_tables:
        return

    joined = ", ".join(missing_tables)
    raise RuntimeError(
        "Database schema is not initialized. Missing tables: "
        f"{joined}. Run './.venv/bin/aerich upgrade' before starting Close-Devs."
    )


async def init_database(database: DatabaseConfig, *, ensure_schema: bool = False) -> None:
    if getattr(Tortoise, "_inited", False):
        await close_database()
    else:
        EXECUTOR_CACHE.clear()

    database_url = resolve_database_url(database)
    if database.backend == "sqlite" and database_url.startswith("sqlite://"):
        sqlite_path = Path(database_url.removeprefix("sqlite://"))
        if sqlite_path.name != ":memory:":
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    await Tortoise.init(config=build_tortoise_config(database))
    try:
        connection = connections.get("default")
        await connection.execute_query("SELECT 1")
        if database.backend == "sqlite":
            await connection.execute_query(
                f"PRAGMA busy_timeout = {int(database.sqlite_busy_timeout_ms)}"
            )
        if ensure_schema:
            await Tortoise.generate_schemas(safe=True)
        else:
            await _validate_runtime_schema(database)
    except Exception:
        await close_database()
        raise


async def close_database() -> None:
    if getattr(Tortoise, "_inited", False):
        await connections.close_all(discard=True)
        Tortoise.apps.clear()
        Tortoise._inited = False
    EXECUTOR_CACHE.clear()


TORTOISE_ORM = build_tortoise_config(DatabaseConfig(url=_default_database_url()))
