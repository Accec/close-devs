from __future__ import annotations

from uuid import uuid4

from core.models import AgentKind, FeedbackBundle, PatchProposal, Task, TaskType


class TaskDispatcher:
    def _new_task_id(self) -> str:
        return uuid4().hex

    def create_review_tasks(
        self,
        run_id: str,
        targets: list[str],
        commands: list[str],
    ) -> tuple[Task, Task]:
        static_task = Task(
            task_id=self._new_task_id(),
            run_id=run_id,
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.STATIC_REVIEW,
            targets=targets,
            payload={},
        )
        dynamic_task = Task(
            task_id=self._new_task_id(),
            run_id=run_id,
            agent_kind=AgentKind.DYNAMIC_DEBUG,
            task_type=TaskType.DYNAMIC_DEBUG,
            targets=[],
            payload={"commands": commands},
        )
        return static_task, dynamic_task

    def create_maintenance_task(
        self,
        run_id: str,
        feedback: FeedbackBundle,
        *,
        handoffs: list[dict[str, object]] | None = None,
    ) -> Task:
        return Task(
            task_id=self._new_task_id(),
            run_id=run_id,
            agent_kind=AgentKind.MAINTENANCE,
            task_type=TaskType.MAINTENANCE,
            targets=feedback.change_set.all_touched_files,
            payload={"feedback": feedback, "handoffs": list(handoffs or [])},
        )

    def create_validation_tasks(
        self,
        run_id: str,
        patch: PatchProposal,
        commands: list[str],
    ) -> tuple[Task, Task]:
        static_task = Task(
            task_id=self._new_task_id(),
            run_id=run_id,
            agent_kind=AgentKind.STATIC_REVIEW,
            task_type=TaskType.VALIDATION_STATIC,
            targets=patch.validation_targets or patch.touched_files,
            payload={"validation": True},
        )
        dynamic_task = Task(
            task_id=self._new_task_id(),
            run_id=run_id,
            agent_kind=AgentKind.DYNAMIC_DEBUG,
            task_type=TaskType.VALIDATION_DYNAMIC,
            targets=[],
            payload={"commands": commands, "validation": True},
        )
        return static_task, dynamic_task
