from __future__ import annotations

from pathlib import Path
import shlex
import sys

import pytest

from tools.command_runner import CommandRunner


@pytest.mark.asyncio
async def test_command_runner_captures_output(tmp_path: Path) -> None:
    runner = CommandRunner()
    command = f"{shlex.quote(sys.executable)} -c \"print('ok')\""

    result = await runner.run(command, cwd=tmp_path, timeout_seconds=5)

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_command_runner_handles_timeout(tmp_path: Path) -> None:
    runner = CommandRunner()
    command = f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(2.0)\""

    result = await runner.run(command, cwd=tmp_path, timeout_seconds=1)

    assert result.returncode == 124
    assert result.timed_out is True
