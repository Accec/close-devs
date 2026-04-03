from __future__ import annotations

from typing import Any

from memory.state_store import StateStore


class RunHistory:
    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store

    async def latest(self, repo_root: str) -> dict[str, Any] | None:
        return await self.state_store.latest_run(repo_root)

    async def recent(self, repo_root: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.state_store.recent_runs(repo_root, limit=limit)
