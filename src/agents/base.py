from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import AgentKind, AgentResult, RunContext, Task, TaskType


class BaseAgent(ABC):
    kind: AgentKind
    allowed_task_types: frozenset[TaskType]

    def ensure_task_type(self, task: Task) -> None:
        if task.task_type not in self.allowed_task_types:
            raise ValueError(
                f"{self.kind.value} cannot handle task type {task.task_type.value}"
            )

    @abstractmethod
    async def run(self, task: Task, context: RunContext) -> AgentResult:
        raise NotImplementedError
