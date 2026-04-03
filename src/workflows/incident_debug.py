from __future__ import annotations

from core.models import WorkflowReport
from workflows.base import BaseWorkflow


class IncidentDebugWorkflow(BaseWorkflow):
    name = "incident_debug"

    async def run(self, orchestrator: object, **kwargs: object) -> WorkflowReport:
        return await orchestrator.run_dynamic_debug_cycle()
