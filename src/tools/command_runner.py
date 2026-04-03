from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import os
import time


@dataclass(slots=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False


class CommandRunner:
    async def run(
        self,
        command: str,
        cwd: Path,
        timeout_seconds: int = 120,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        started = time.perf_counter()
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
            duration = time.perf_counter() - started
            return CommandResult(
                command=command,
                returncode=process.returncode or 0,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_seconds=duration,
            )
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            duration = time.perf_counter() - started
            return CommandResult(
                command=command,
                returncode=124,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                duration_seconds=duration,
                timed_out=True,
            )
