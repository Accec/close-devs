from __future__ import annotations

from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path


def should_include(path: str, include: list[str], exclude: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    def matches(pattern: str) -> bool:
        return fnmatch(normalized, pattern) or (
            pattern.startswith("**/") and fnmatch(normalized, pattern[3:])
        )

    if include and not any(matches(pattern) for pattern in include):
        return False
    if any(matches(pattern) for pattern in exclude):
        return False
    return True


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
