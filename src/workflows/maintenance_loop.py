from __future__ import annotations

from core.models import WorkflowReport
from workflows.base import BaseWorkflow


class MaintenanceLoopWorkflow(BaseWorkflow):
    name = "maintenance_loop"

    async def run(self, orchestrator: object, **kwargs: object) -> WorkflowReport:
        return await orchestrator.run_maintenance_cycle()
