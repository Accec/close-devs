from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import tempfile


class FileStore:
    async def read_text(self, path: Path) -> str:
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def write_text(self, path: Path, content: str) -> None:
        await asyncio.to_thread(self._write_text_sync, path, content)

    def _write_text_sync(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def materialize_workspace_copy(self, repo_root: Path) -> Path:
        return await asyncio.to_thread(self._materialize_workspace_copy_sync, repo_root)

    def _materialize_workspace_copy_sync(self, repo_root: Path) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="close_devs_validation_"))
        destination = tmp_dir / repo_root.name
        shutil.copytree(
            repo_root,
            destination,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "state",
                "reports",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
            ),
        )
        return destination
