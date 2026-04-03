from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from core.models import RepoSnapshot, utc_now
from repo.snapshot import hash_file, should_include


class RepositoryScanner:
    def __init__(self, include: list[str], exclude: list[str]) -> None:
        self.include = include
        self.exclude = exclude

    async def scan(self, repo_root: Path) -> RepoSnapshot:
        return await asyncio.to_thread(self._scan_sync, repo_root)

    def _scan_sync(self, repo_root: Path) -> RepoSnapshot:
        file_hashes: dict[str, str] = {}
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root).as_posix()
            if not should_include(relative, self.include, self.exclude):
                continue
            file_hashes[relative] = hash_file(path)

        return RepoSnapshot(
            repo_root=str(repo_root),
            scanned_at=utc_now(),
            revision=self._detect_revision(repo_root),
            file_hashes=file_hashes,
        )

    def _detect_revision(self, repo_root: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        revision = result.stdout.strip()
        return revision or None
