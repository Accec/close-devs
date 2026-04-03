from __future__ import annotations

from agents.base import BaseAgent
from core.models import AgentKind, AgentResult, Finding, RunContext, Severity, Task, TaskStatus, TaskType
from tools.test_runner import TestRunner
from tools.traceback_parser import TracebackParser


class DynamicDebugAgent(BaseAgent):
    kind = AgentKind.DYNAMIC_DEBUG
    allowed_task_types = frozenset({TaskType.DYNAMIC_DEBUG, TaskType.VALIDATION_DYNAMIC})

    def __init__(
        self,
        test_runner: TestRunner | None = None,
        traceback_parser: TracebackParser | None = None,
    ) -> None:
        self.test_runner = test_runner or TestRunner()
        self.traceback_parser = traceback_parser or TracebackParser()

    async def run(self, task: Task, context: RunContext) -> AgentResult:
        self.ensure_task_type(task)
        commands = list(task.payload.get("commands", []))
        if not commands:
            return AgentResult(
                task_id=task.task_id,
                agent_kind=self.kind,
                task_type=task.task_type,
                status=TaskStatus.SKIPPED,
                summary="No dynamic commands configured for execution.",
                artifacts={"commands": []},
            )

        findings: list[Finding] = []
        artifacts: list[dict[str, object]] = []
        for command in commands:
            result = await self.test_runner.run(
                command=command,
                repo_root=context.working_repo_root,
                timeout_seconds=context.config.dynamic_debug.timeout_seconds,
            )
            artifacts.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            if result.timed_out:
                findings.append(
                    Finding(
                        source_agent=self.kind,
                        severity=Severity.HIGH,
                        rule_id="command-timeout",
                        message=f"Command timed out after configured timeout: {command}",
                        category="runtime",
                        evidence={"command": command},
                    )
                )
                continue
            if result.returncode == 0:
                continue

            tracebacks = self.traceback_parser.parse("\n".join([result.stdout, result.stderr]))
            if tracebacks:
                for parsed in tracebacks:
                    findings.append(
                        Finding(
                            source_agent=self.kind,
                            severity=Severity.HIGH,
                            rule_id="python-traceback",
                            message=f"{parsed.exception_type}: {parsed.message}",
                            category="runtime",
                            evidence={"command": command, "traceback": parsed.text},
                        )
                    )
            else:
                findings.append(
                    Finding(
                        source_agent=self.kind,
                        severity=Severity.MEDIUM,
                        rule_id="command-failed",
                        message=f"Command failed with exit code {result.returncode}: {command}",
                        category="runtime",
                        evidence={"command": command},
                    )
                )

        summary = f"Dynamic debug executed {len(commands)} commands with {len(findings)} findings."
        return AgentResult(
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            status=TaskStatus.SUCCEEDED,
            summary=summary,
            findings=findings,
            artifacts={"commands": artifacts},
        )
