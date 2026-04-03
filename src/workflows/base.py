from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import WorkflowReport


class BaseWorkflow(ABC):
    name: str

    @abstractmethod
    async def run(self, orchestrator: object, **kwargs: object) -> WorkflowReport:
        raise NotImplementedError
