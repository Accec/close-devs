from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess


def test_aerich_upgrade_bootstraps_fresh_sqlite_database(tmp_path: Path) -> None:
    database_path = tmp_path / "fresh.db"
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite://{database_path}"

    result = subprocess.run(
        [".venv/bin/aerich", "upgrade"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert database_path.exists()

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "aerich" in tables
    assert "runs" in tables
