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


def _run_command_with_retry(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    attempts: int = 5,
    delay_seconds: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _run_command(command, env=env)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            stderr = exc.stderr.lower()
            transient = any(
                token in stderr
                for token in (
                    "connection reset by peer",
                    "connection refused",
                    "the database system is starting up",
                    "terminating connection",
                    "could not connect",
                )
            )
            if not transient or attempt == attempts:
                raise
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


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

    _run_command_with_retry(
        [sys.executable, "-m", "aerich", "upgrade"],
        env=env,
        attempts=10,
        delay_seconds=1.0,
    )
    yield DEFAULT_POSTGRES_URL
    _run_command(["docker", "compose", "down", "-v"], env=env, check=False)
