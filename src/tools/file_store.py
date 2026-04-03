from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import tempfile


class FileStore:
    async def ensure_dir(self, path: Path) -> None:
        await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)

    async def read_text(self, path: Path) -> str:
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def write_text(self, path: Path, content: str) -> None:
        await asyncio.to_thread(self._write_text_sync, path, content)

    def _write_text_sync(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    async def materialize_workspace_copy(
        self,
        repo_root: Path,
        *,
        destination: Path | None = None,
    ) -> Path:
        return await asyncio.to_thread(
            self._materialize_workspace_copy_sync,
            repo_root,
            destination,
        )

    def _materialize_workspace_copy_sync(
        self,
        repo_root: Path,
        destination: Path | None = None,
    ) -> Path:
        target = destination
        if target is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix="close_devs_validation_"))
            target = tmp_dir / repo_root.name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            repo_root,
            target,
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
        return target
