from __future__ import annotations

from agents.base import BaseAgent
from core.models import AgentKind, AgentResult, RunContext, Task, TaskStatus, TaskType
from tools.static_tooling import StaticTooling


class StaticReviewAgent(BaseAgent):
    kind = AgentKind.STATIC_REVIEW
    allowed_task_types = frozenset({TaskType.STATIC_REVIEW, TaskType.VALIDATION_STATIC})

    def __init__(self, tooling: StaticTooling | None = None) -> None:
        self.tooling = tooling or StaticTooling()

    async def run(self, task: Task, context: RunContext) -> AgentResult:
        self.ensure_task_type(task)
        findings, artifacts = await self.tooling.review(
            repo_root=context.working_repo_root,
            targets=task.targets,
            config=context.config,
            rules=context.rules,
        )
        summary = (
            f"Static review completed for {len(task.targets)} targets with {len(findings)} findings."
        )
        return AgentResult(
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            status=TaskStatus.SUCCEEDED,
            summary=summary,
            findings=findings,
            artifacts=artifacts,
        )
