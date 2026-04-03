from __future__ import annotations

from pathlib import Path
import os
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


@pytest.mark.asyncio
async def test_command_runner_respects_env_path_override(tmp_path: Path) -> None:
    runner = CommandRunner()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script_path = bin_dir / "hello-tool"
    script_path.write_text("#!/bin/sh\nprintf 'from-env'\n", encoding="utf-8")
    script_path.chmod(0o755)
    command = (
        f"{shlex.quote(sys.executable)} -c "
        "\"import shutil; print(shutil.which('hello-tool') or '')\""
    )

    result = await runner.run(
        command,
        cwd=tmp_path,
        timeout_seconds=5,
        env={"PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", "")},
    )

    assert result.returncode == 0
    assert result.stdout.strip().endswith("hello-tool")
