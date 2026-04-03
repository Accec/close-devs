from __future__ import annotations

import os
import sys
from pathlib import Path
import shutil
import subprocess
import time

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.support import DEFAULT_POSTGRES_URL


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def postgres_database_url() -> str:
    if shutil.which("docker") is None:
        pytest.skip("docker is required for PostgreSQL integration tests")

    env = {**os.environ, "DATABASE_URL": DEFAULT_POSTGRES_URL}
    _run_command(["docker", "compose", "down", "-v"], env=env, check=False)
    try:
        _run_command(["docker", "compose", "up", "-d", "postgres"], env=env)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or "docker compose up failed"
        pytest.skip(f"PostgreSQL integration tests skipped: {stderr}")

    for _ in range(30):
        readiness = _run_command(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-U",
                "close_devs",
                "-d",
                "close_devs",
            ],
            env=env,
            check=False,
        )
        if readiness.returncode == 0:
            break
        time.sleep(1)
    else:
        _run_command(["docker", "compose", "logs", "postgres"], env=env, check=False)
        pytest.fail("PostgreSQL service failed to become ready")

    _run_command([sys.executable, "-m", "aerich", "upgrade"], env=env)
    yield DEFAULT_POSTGRES_URL
    _run_command(["docker", "compose", "down", "-v"], env=env, check=False)
