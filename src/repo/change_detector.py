from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from core.models import ChangeSet, RepoSnapshot


class ChangeDetector:
    async def detect(
        self,
        repo_root: Path,
        current: RepoSnapshot,
        previous: RepoSnapshot | None,
    ) -> ChangeSet:
        return await asyncio.to_thread(self._detect_sync, repo_root, current, previous)

    def _detect_sync(
        self,
        repo_root: Path,
        current: RepoSnapshot,
        previous: RepoSnapshot | None,
    ) -> ChangeSet:
        if previous is None:
            return ChangeSet(
                changed_files=[],
                added_files=current.files,
                removed_files=[],
                baseline_revision=None,
                current_revision=current.revision,
                reason="initial-scan",
            )

        git_changes = self._detect_via_git(repo_root, previous.revision, current.revision)
        if git_changes is not None:
            changed, added, removed = git_changes
            return ChangeSet(
                changed_files=sorted(changed),
                added_files=sorted(added),
                removed_files=sorted(removed),
                baseline_revision=previous.revision,
                current_revision=current.revision,
                reason="git-diff",
            )

        previous_paths = set(previous.file_hashes)
        current_paths = set(current.file_hashes)
        added = current_paths - previous_paths
        removed = previous_paths - current_paths
        changed = {
            path
            for path in current_paths & previous_paths
            if current.file_hashes[path] != previous.file_hashes[path]
        }
        return ChangeSet(
            changed_files=sorted(changed),
            added_files=sorted(added),
            removed_files=sorted(removed),
            baseline_revision=previous.revision,
            current_revision=current.revision,
            reason="hash-diff",
        )

    def _detect_via_git(
        self,
        repo_root: Path,
        previous_revision: str | None,
        current_revision: str | None,
    ) -> tuple[set[str], set[str], set[str]] | None:
        if not previous_revision or not current_revision or previous_revision == current_revision:
            return None

        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "diff",
                    "--name-status",
                    previous_revision,
                    current_revision,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None

        changed: set[str] = set()
        added: set[str] = set()
        removed: set[str] = set()
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status, path = parts[0], parts[-1]
            if status.startswith("A"):
                added.add(path)
            elif status.startswith("D"):
                removed.add(path)
            else:
                changed.add(path)
        return changed, added, removed
