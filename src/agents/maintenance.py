from __future__ import annotations

from agents.base import BaseAgent
from core.agent_kernel import AgentKernelResult
from core.config import AgentRuntimeConfig
from core.models import (
    AgentKind,
    AgentResult,
    CompletionReason,
    FilePatch,
    PatchProposal,
    RunContext,
    SafeFixPolicy,
    Task,
    TaskStatus,
    TaskType,
    ToolPermissionSet,
)
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient
from tools.agent_toolkit import AgentToolkitFactory
from tools.file_store import FileStore
from tools.patch_service import PatchService


class MaintenanceAgent(BaseAgent):
    kind = AgentKind.MAINTENANCE
    allowed_task_types = frozenset({TaskType.MAINTENANCE})
    default_tools = frozenset(
        {
            "read_file",
            "search_repo",
            "git_diff",
            "prepare_safe_patch",
            "write_file",
        }
    )

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        runtime_config: AgentRuntimeConfig | None = None,
        toolkit_factory: AgentToolkitFactory | None = None,
        patch_service: PatchService | None = None,
        file_store: FileStore | None = None,
        safe_fix_policy: SafeFixPolicy | None = None,
        permissions: ToolPermissionSet | None = None,
    ) -> None:
        runtime = runtime_config or AgentRuntimeConfig()
        permission_set = permissions or ToolPermissionSet(
            allowed_tools=frozenset(runtime.allowed_tools or self.default_tools),
            allow_write=True,
        )
        super().__init__(
            llm_client=llm_client or MockLLMClient(),
            runtime_config=runtime,
            permissions=permission_set,
        )
        self.toolkit_factory = toolkit_factory or AgentToolkitFactory(
            file_store=file_store,
            patch_service=patch_service,
            safe_fix_policy=safe_fix_policy,
        )
        self.patch_service = patch_service or PatchService(file_store)

    async def run(self, task: Task, context: RunContext) -> AgentResult:
        self.ensure_task_type(task)
        session = self.build_session(
            task=task,
            context=context,
            objective=(
                "Autonomously maintain the repository like a senior coding agent. "
                "Inspect upstream handoffs, prepare a safe patch candidate, and finalize a patch proposal."
            ),
        )
        tools = self.toolkit_factory.build_maintenance_toolkit(self.permissions, context=context)
        kernel_result = await self.kernel.run_session(
            session=session,
            tools=tools,
            permissions=self.permissions,
            context=context,
        )
        patch = self._extract_patch(kernel_result)
        artifacts = self.base_artifacts(kernel_result, handoffs=[])
        if patch is not None:
            if context.config.auto_apply_patch and patch.file_patches:
                await self.patch_service.apply(context.repo_root, patch)
                patch.applied = True
        latest_patch_output = self._latest_patch_output(kernel_result)
        artifacts.update(
            {
                "suggestions": list(kernel_result.final_response.get("suggestions", latest_patch_output.get("suggestions", []))),
                "safe_fix_only": bool(latest_patch_output.get("safe_fix_only", False)),
                "publishable": bool(latest_patch_output.get("publishable", False)),
                "unpublishable_reasons": self._unpublishable_reasons(latest_patch_output),
                "patch_kind": str(latest_patch_output.get("patch_kind", "advisory_only_patch")),
                "applied_rule_ids": list(latest_patch_output.get("applied_rule_ids", [])),
                "unsupported_findings": list(latest_patch_output.get("unsupported_findings", [])),
                "unresolved_handoffs": list(latest_patch_output.get("unresolved_handoffs", [])),
                "auto_fixed_blockers": list(latest_patch_output.get("auto_fixed_blockers", [])),
                "repair_scope": list(latest_patch_output.get("repair_scope", [])),
            }
        )
        summary = str(
            kernel_result.final_response.get(
                "summary",
                latest_patch_output.get("summary", "Autonomous maintenance completed."),
            )
        )
        status = (
            TaskStatus.FAILED
            if kernel_result.session.completion_reason == CompletionReason.FAILED
            else TaskStatus.SUCCEEDED
        )
        return AgentResult(
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            status=status,
            summary=summary,
            patch=patch,
            artifacts=artifacts,
            errors=list(kernel_result.errors),
        )

    def _extract_patch(self, kernel_result: AgentKernelResult) -> PatchProposal | None:
        output = self._latest_patch_output(kernel_result)
        file_patches_data = output.get("file_patches", [])
        if not file_patches_data and not output.get("summary"):
            return None
        file_patches = [
            FilePatch(
                path=str(item.get("path", "")),
                old_content=str(item.get("old_content", "")),
                new_content=str(item.get("new_content", "")),
                diff=str(item.get("diff", "")),
            )
            for item in file_patches_data
            if isinstance(item, dict)
        ]
        return PatchProposal(
            summary=str(output.get("summary", "Autonomous maintenance patch proposal.")),
            rationale=str(
                kernel_result.final_response.get(
                    "rationale",
                    output.get("rationale", "Autonomous maintenance did not provide a separate rationale."),
                )
            ),
            file_patches=file_patches,
            validation_targets=[str(item) for item in output.get("validation_targets", [])],
            suggestions=[str(item) for item in kernel_result.final_response.get("suggestions", output.get("suggestions", []))],
            metadata={
                "repair_scope": [str(item) for item in output.get("repair_scope", [])],
                "patch_kind": str(output.get("patch_kind", "advisory_only_patch")),
                "safe_fix_only": bool(output.get("safe_fix_only", False)),
                "unsupported_findings": list(output.get("unsupported_findings", [])),
                "unresolved_handoffs": list(output.get("unresolved_handoffs", [])),
                "auto_fixed_blockers": list(output.get("auto_fixed_blockers", [])),
            },
            applied=bool(output.get("applied", False)),
            diff_text=str(output.get("diff_text", "")),
        )

    def _latest_patch_output(self, kernel_result: AgentKernelResult) -> dict[str, object]:
        for call in reversed(kernel_result.session.tool_calls):
            if call.tool_name == "prepare_safe_patch" and call.status == "succeeded":
                return call.output
        return {}

    def _unpublishable_reasons(self, latest_patch_output: dict[str, object]) -> list[str]:
        reasons: list[str] = []
        if not latest_patch_output.get("file_patches"):
            reasons.append("no_safe_patch_generated")
        if latest_patch_output.get("file_patches") and not latest_patch_output.get("safe_fix_only", False):
            reasons.append("patch_contains_non_safe_rules")
        if latest_patch_output.get("unresolved_handoffs"):
            reasons.append("unresolved_high_value_handoffs")
        return reasons
