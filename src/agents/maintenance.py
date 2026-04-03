from __future__ import annotations

from collections import defaultdict

from agents.base import BaseAgent
from core.models import (
    AgentKind,
    AgentResult,
    FeedbackBundle,
    FilePatch,
    PatchProposal,
    RunContext,
    SafeFixPolicy,
    Task,
    TaskStatus,
    TaskType,
)
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient
from tools.file_store import FileStore
from tools.patch_service import PatchService


class MaintenanceAgent(BaseAgent):
    kind = AgentKind.MAINTENANCE
    allowed_task_types = frozenset({TaskType.MAINTENANCE})

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        patch_service: PatchService | None = None,
        file_store: FileStore | None = None,
        safe_fix_policy: SafeFixPolicy | None = None,
    ) -> None:
        self.file_store = file_store or FileStore()
        self.patch_service = patch_service or PatchService(self.file_store)
        self.llm_client = llm_client or MockLLMClient()
        self.safe_fix_policy = safe_fix_policy or SafeFixPolicy()

    async def run(self, task: Task, context: RunContext) -> AgentResult:
        self.ensure_task_type(task)
        feedback: FeedbackBundle = task.payload["feedback"]
        rationale, suggestions = await self.llm_client.summarize_feedback(feedback)

        updated_contents: dict[str, str] = {}
        original_contents: dict[str, str] = {}
        unsupported_findings: list[str] = []
        applied_rule_ids: set[str] = set()

        grouped_findings = defaultdict(list)
        for finding in feedback.static_findings:
            if finding.path:
                grouped_findings[finding.path].append(finding)

        for relative_path, findings in grouped_findings.items():
            absolute_path = context.working_repo_root / relative_path
            if not absolute_path.exists():
                continue

            if relative_path not in original_contents:
                original_contents[relative_path] = await self.file_store.read_text(absolute_path)
            current_content = updated_contents.get(relative_path, original_contents[relative_path])
            transformed_content = current_content

            for finding in findings:
                updated = self._apply_autofix(relative_path, finding.rule_id, transformed_content)
                if updated is None:
                    unsupported_findings.append(
                        f"{finding.rule_id} at {finding.path or 'unknown path'}"
                    )
                    continue
                transformed_content = updated
                applied_rule_ids.add(finding.rule_id)

            if transformed_content != current_content:
                updated_contents[relative_path] = transformed_content

        file_patches: list[FilePatch] = []
        for relative_path, new_content in sorted(updated_contents.items()):
            old_content = original_contents[relative_path]
            file_patches.append(
                self.patch_service.build_file_patch(relative_path, old_content, new_content)
            )

        if unsupported_findings:
            suggestions.extend(
                f"Manual follow-up required for {item}" for item in unsupported_findings[:10]
            )

        patch = PatchProposal(
            summary=(
                f"Generated {len(file_patches)} file patches from "
                f"{len(feedback.all_findings)} upstream findings."
            ),
            rationale=rationale,
            file_patches=file_patches,
            validation_targets=sorted(set(updated_contents.keys())),
            suggestions=suggestions,
        )
        patch.diff_text = self.patch_service.render_patch(file_patches)

        if context.config.auto_apply_patch and patch.file_patches:
            await self.patch_service.apply(context.repo_root, patch)
            patch.applied = True

        safe_fix_only = bool(applied_rule_ids) and all(
            self.safe_fix_policy.allows(rule_id) for rule_id in applied_rule_ids
        )
        publishable = bool(patch.file_patches) and safe_fix_only
        unpublishable_reasons: list[str] = []
        if not patch.file_patches:
            unpublishable_reasons.append("no_safe_patch_generated")
        if patch.file_patches and not safe_fix_only:
            unpublishable_reasons.append("patch_contains_non_safe_rules")

        return AgentResult(
            task_id=task.task_id,
            agent_kind=self.kind,
            task_type=task.task_type,
            status=TaskStatus.SUCCEEDED,
            summary=patch.summary,
            patch=patch,
            artifacts={
                "suggestions": patch.suggestions,
                "safe_fix_only": safe_fix_only,
                "publishable": publishable,
                "unpublishable_reasons": unpublishable_reasons,
                "patch_kind": "safe_autofix_patch" if patch.file_patches else "advisory_only_patch",
                "applied_rule_ids": sorted(applied_rule_ids),
                "unsupported_findings": unsupported_findings,
            },
        )

    def _apply_autofix(self, relative_path: str, rule_id: str, content: str) -> str | None:
        if rule_id in {"missing-module-docstring", "D100"} and relative_path.endswith(".py"):
            return self.patch_service.add_module_docstring(
                content,
                "Maintained by Close-Devs.",
            )
        if rule_id in {"missing-final-newline", "W292"}:
            return self.patch_service.ensure_final_newline(content)
        if rule_id in {"trailing-whitespace", "W291", "W293"}:
            return self.patch_service.strip_trailing_whitespace(content)
        if rule_id in {"excessive-eof-blank-lines", "W391"}:
            return self.patch_service.normalize_eof_blank_lines(content)
        return None
