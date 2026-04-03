from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.models import RepoSnapshot
from repo.change_detector import ChangeDetector


@pytest.mark.asyncio
async def test_change_detector_falls_back_to_hash_diff(tmp_path: Path) -> None:
    detector = ChangeDetector()
    previous = RepoSnapshot(
        repo_root=str(tmp_path),
        scanned_at=datetime.now(timezone.utc),
        revision=None,
        file_hashes={"a.py": "old-a", "b.py": "same"},
    )
    current = RepoSnapshot(
        repo_root=str(tmp_path),
        scanned_at=datetime.now(timezone.utc),
        revision=None,
        file_hashes={"a.py": "new-a", "c.py": "new-c", "b.py": "same"},
    )

    change_set = await detector.detect(tmp_path, current, previous)

    assert change_set.reason == "hash-diff"
    assert change_set.changed_files == ["a.py"]
    assert change_set.added_files == ["c.py"]
    assert change_set.removed_files == []
