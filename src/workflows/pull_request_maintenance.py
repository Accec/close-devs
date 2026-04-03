from __future__ import annotations

from pathlib import Path

from core.models import PullRequestContext, WorkflowReport
from workflows.base import BaseWorkflow


class PullRequestMaintenanceWorkflow(BaseWorkflow):
    name = "pull_request_maintenance"

    async def run(
        self,
        orchestrator: object,
        *,
        pr_context: PullRequestContext | None = None,
        event_path: Path | None = None,
        pr_number: int | None = None,
    ) -> WorkflowReport:
        return await orchestrator.run_pull_request_maintenance(
            pr_context=pr_context,
            event_path=event_path,
            pr_number=pr_number,
        )
