from __future__ import annotations

from pathlib import Path

from tools.command_runner import CommandResult, CommandRunner


class TestRunner:
    __test__ = False

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    async def run(self, command: str, repo_root: Path, timeout_seconds: int) -> CommandResult:
        return await self.runner.run(command, cwd=repo_root, timeout_seconds=timeout_seconds)
