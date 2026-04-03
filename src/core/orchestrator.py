from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langgraph.graph import END, START, StateGraph

from agents.dynamic_debug import DynamicDebugAgent
from agents.maintenance import MaintenanceAgent
from agents.static_review import StaticReviewAgent
from core.config import AgentRuntimeConfig, AppConfig, LLMConfig, load_config, load_rules
from core.dispatcher import TaskDispatcher
from core.logging import configure_logging
from core.models import (
    AgentKind,
    AgentResult,
    ArtifactReference,
    ChangeSet,
    ExecutionEnvironment,
    FeedbackBundle,
    PublishContext,
    PublishMode,
    PullRequestContext,
    RepoSnapshot,
    RunContext,
    SafeFixPolicy,
    Severity,
    Task,
    TaskStatus,
    TaskType,
    WorkflowReport,
    WorkflowState,
)
from github.adapter import GitHubAdapter
from github.rendering import build_companion_pr_payload, build_review_payload
from llm.factory import build_llm_client
from memory.issue_catalog import IssueCatalog
from memory.run_history import RunHistory
from memory.state_store import StateStore
from repo.change_detector import ChangeDetector
from repo.scanner import RepositoryScanner
from reports.markdown import write_markdown
from reports.serializer import publish_context_from_dict, read_json, workflow_report_from_dict, write_json
from skills.evolution import SkillEvolutionService
from skills.manager import SkillManager
from tools.file_store import FileStore
from tools.environment_manager import EnvironmentManager
from tools.patch_service import PatchService
from tools.agent_toolkit import AgentToolkitFactory
from workflows.incident_debug import IncidentDebugWorkflow
from workflows.maintenance_loop import MaintenanceLoopWorkflow
from workflows.pull_request_maintenance import PullRequestMaintenanceWorkflow


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        state_store: StateStore,
        github_adapter: GitHubAdapter | None = None,
    ) -> None:
        self.config = config
        self.logger = configure_logging(config.log_level)
        self.rules = load_rules(config.rules_path)
        self.state_store = state_store
        self.issue_catalog = IssueCatalog(self.state_store)
        self.run_history = RunHistory(self.state_store)
        self.scanner = RepositoryScanner(config.include, config.exclude)
        self.change_detector = ChangeDetector()
        self.dispatcher = TaskDispatcher()
        self.file_store = FileStore()
        self.patch_service = PatchService(self.file_store)
        self.safe_fix_policy = SafeFixPolicy()
        self.skill_manager = SkillManager(config, state_store)
        self.skill_evolution = SkillEvolutionService(config, state_store)
        self.toolkit_factory = AgentToolkitFactory(
            file_store=self.file_store,
            patch_service=self.patch_service,
            safe_fix_policy=self.safe_fix_policy,
        )
        self.environment_manager = EnvironmentManager(
            file_store=self.file_store,
            command_runner=self.toolkit_factory.command_runner,
            logger=self.logger,
        )
        self.static_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.static.model),
            self.logger,
        )
        self.dynamic_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.dynamic.model),
            self.logger,
        )
        self.maintenance_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.maintenance.model),
            self.logger,
        )
        self.static_agent = StaticReviewAgent(
            llm_client=self.static_llm_client,
            runtime_config=config.agents.static,
            toolkit_factory=self.toolkit_factory,
        )
        self.dynamic_agent = DynamicDebugAgent(
            llm_client=self.dynamic_llm_client,
            runtime_config=config.agents.dynamic,
            toolkit_factory=self.toolkit_factory,
        )
        self.maintenance_agent = MaintenanceAgent(
            llm_client=self.maintenance_llm_client,
            runtime_config=config.agents.maintenance,
            toolkit_factory=self.toolkit_factory,
            patch_service=self.patch_service,
            file_store=self.file_store,
            safe_fix_policy=self.safe_fix_policy,
        )
        self.github_adapter = github_adapter or GitHubAdapter(
            config.github,
            logger=self.logger,
        )
        self._log_agent_runtime("static_review", self.static_llm_client, config.agents.static)
        self._log_agent_runtime("dynamic_debug", self.dynamic_llm_client, config.agents.dynamic)
        self._log_agent_runtime("maintenance", self.maintenance_llm_client, config.agents.maintenance)

    def _agent_llm_config(self, model_override: str | None) -> LLMConfig:
        if not model_override:
            return self.config.llm
        return LLMConfig(
            provider=self.config.llm.provider,
            model=model_override,
            base_url=self.config.llm.base_url,
            api_key_env=self.config.llm.api_key_env,
            timeout_seconds=self.config.llm.timeout_seconds,
            temperature=self.config.llm.temperature,
            system_prompt=self.config.llm.system_prompt,
        )

    def _log_agent_runtime(
        self,
        agent_name: str,
        llm_client: object,
        runtime_config: AgentRuntimeConfig,
    ) -> None:
        self.logger.info(
            "Agent runtime ready: agent=%s client=%s model=%s max_steps=%s max_tool_calls=%s tools=%s",
            agent_name,
            type(llm_client).__name__,
            runtime_config.model or self.config.llm.model,
            runtime_config.max_steps,
            runtime_config.max_tool_calls,
            ",".join(runtime_config.allowed_tools),
        )

    @classmethod
    async def create(
        cls,
        config: AppConfig,
        github_adapter: GitHubAdapter | None = None,
        *,
        ensure_schema: bool = False,
    ) -> "Orchestrator":
        state_store = await StateStore.create(
            config.database,
            ensure_schema=ensure_schema or config.database.backend == "sqlite",
        )
        return cls(config, state_store, github_adapter=github_adapter)

    async def close(self) -> None:
        await self.state_store.close()

    async def run_workflow(
        self,
        name: str,
        *,
        pr_context: PullRequestContext | None = None,
        event_path: Path | None = None,
        pr_number: int | None = None,
    ) -> WorkflowReport:
        workflows = {
            "maintenance_loop": MaintenanceLoopWorkflow(),
            "incident_debug": IncidentDebugWorkflow(),
            "pull_request_maintenance": PullRequestMaintenanceWorkflow(),
        }
        if name not in workflows:
            raise ValueError(f"Unknown workflow: {name}")
        return await workflows[name].run(
            self,
            pr_context=pr_context,
            event_path=event_path,
            pr_number=pr_number,
        )

    async def scan_repository(
        self,
        repo_root: Path | None = None,
        pr_context: PullRequestContext | None = None,
    ) -> tuple[RepoSnapshot, ChangeSet]:
        target_root = repo_root or self.config.repo_root
        snapshot = await self.scanner.scan(target_root)
        if pr_context is not None and pr_context.changed_files:
            change_set = ChangeSet(
                changed_files=sorted(set(pr_context.changed_files)),
                added_files=[],
                removed_files=[],
                baseline_revision=None,
                current_revision=pr_context.head_sha,
                reason="pull-request-files",
            )
            return snapshot, change_set

        previous_snapshot = await self.state_store.get_latest_snapshot(str(target_root))
        change_set = await self.change_detector.detect(target_root, snapshot, previous_snapshot)
        return snapshot, change_set

    async def execute_task(self, task: Task, context: RunContext) -> AgentResult:
        await self.state_store.save_task(task)
        await self.state_store.update_task(task.task_id, TaskStatus.RUNNING, "Task started.")
        agent = {
            AgentKind.STATIC_REVIEW: self.static_agent,
            AgentKind.DYNAMIC_DEBUG: self.dynamic_agent,
            AgentKind.MAINTENANCE: self.maintenance_agent,
        }[task.agent_kind]
        task_logger = context.logger.getChild(f"agent.{task.agent_kind.value}")
        if context.config.log_agent_activity:
            task_logger.info(
                "Task dispatched: task=%s type=%s targets=%s working_repo=%s",
                task.task_id,
                task.task_type.value,
                len(task.targets),
                context.working_repo_root,
            )
        try:
            result = await agent.run(task, context)
        except Exception as exc:
            context.logger.exception("Task failed: %s", task.task_id)
            result = AgentResult(
                task_id=task.task_id,
                agent_kind=task.agent_kind,
                task_type=task.task_type,
                status=TaskStatus.FAILED,
                summary=f"Task failed: {exc}",
                errors=[str(exc)],
            )
        if context.config.log_agent_activity:
            task_logger.info(
                "Task finished: task=%s status=%s findings=%s patch_files=%s summary=%s",
                task.task_id,
                result.status.value,
                len(result.findings),
                len(result.patch.file_patches) if result.patch else 0,
                result.summary,
            )
        reflection, candidate = await self.skill_evolution.reflect_and_seed_candidate(
            repo_root=str(context.repo_root),
            run_id=context.run_id,
            task_id=task.task_id,
            session_id=str(result.artifacts.get("session_id", "")),
            result=result,
            active_skill=context.active_skill,
        )
        if reflection is not None:
            result.artifacts["reflection"] = {
                "summary": reflection.summary,
                "metrics": reflection.metrics,
                "upgrade_hints": reflection.upgrade_hints,
                "skill_version": reflection.skill_version,
            }
            result.artifacts["upgrade_hints"] = list(reflection.upgrade_hints)
        if candidate is not None:
            result.artifacts["candidate_skill_version"] = candidate.version
        await self.state_store.save_agent_result(context.run_id, task, result)
        return result

    async def run_maintenance_cycle(self) -> WorkflowReport:
        started_at = datetime.now(timezone.utc)
        run_id = await self.state_store.start_run(
            "maintenance_loop",
            str(self.config.repo_root),
            started_at=started_at,
        )
        try:
            graph = self._build_maintenance_graph()
            state = await graph.ainvoke(
                {
                    "run_id": run_id,
                    "workflow_name": "maintenance_loop",
                    "started_at": started_at,
                    "errors": [],
                }
            )
            return state["report"]
        except Exception as exc:
            await self.state_store.finish_run(run_id, TaskStatus.FAILED, f"Workflow failed: {exc}")
            raise

    async def run_static_review_cycle(self) -> WorkflowReport:
        started_at = datetime.now(timezone.utc)
        run_id = await self.state_store.start_run(
            "static_review",
            str(self.config.repo_root),
            started_at=started_at,
        )
        try:
            environment = await self._prepare_execution_environment(run_id, self.config.repo_root)
            active_skills, candidate_skills = await self._prepare_run_skills(self.config.repo_root)
            snapshot, change_set = await self.scan_repository()
            await self.state_store.save_snapshot(run_id, snapshot)
            analysis_root = (
                Path(environment.base_workspace_root)
                if environment is not None
                else self.config.repo_root
            )
            context = self._build_context(
                run_id,
                working_repo_root=analysis_root,
                repo_root=self.config.repo_root,
                execution_environment=environment,
                active_skill=active_skills.get(AgentKind.STATIC_REVIEW.value),
                candidate_skill=candidate_skills.get(AgentKind.STATIC_REVIEW.value),
            )
            task = Task(
                task_id=self.dispatcher._new_task_id(),
                run_id=run_id,
                agent_kind=AgentKind.STATIC_REVIEW,
                task_type=TaskType.STATIC_REVIEW,
                targets=sorted(set(change_set.added_files + change_set.changed_files)),
                payload={},
            )
            result = await self.execute_task(task, context)
            candidate_skills = await self._refresh_candidate_skills(
                {"candidate_skills": candidate_skills},
                self.config.repo_root,
            )
            await self.issue_catalog.record_findings(str(self.config.repo_root), run_id, result.findings)
            report = WorkflowReport(
                run_id=run_id,
                workflow_name="static_review",
                repo_root=str(self.config.repo_root),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                status=result.status,
                snapshot=snapshot,
                change_set=change_set,
                static_result=result,
                report_dir=str(Path(environment.report_dir)) if environment is not None else "",
                metadata=self._environment_metadata(environment),
            )
            report.metadata.update(
                await self.skill_evolution.evaluate_report(
                    repo_root=str(self.config.repo_root),
                    report=report,
                    active_skills=active_skills,
                    candidate_skills=candidate_skills,
                )
            )
            report_dir = await self._write_report_artifacts(report)
            report.report_dir = str(report_dir)
            await self.state_store.finish_run(
                run_id,
                report.status,
                self._report_summary(report),
                str(report_dir),
            )
            return report
        except Exception as exc:
            await self.state_store.finish_run(run_id, TaskStatus.FAILED, f"Workflow failed: {exc}")
            raise

    async def run_dynamic_debug_cycle(self) -> WorkflowReport:
        started_at = datetime.now(timezone.utc)
        run_id = await self.state_store.start_run(
            "incident_debug",
            str(self.config.repo_root),
            started_at=started_at,
        )
        try:
            graph = self._build_incident_debug_graph()
            state = await graph.ainvoke(
                {
                    "run_id": run_id,
                    "workflow_name": "incident_debug",
                    "started_at": started_at,
                    "errors": [],
                }
            )
            return state["report"]
        except Exception as exc:
            await self.state_store.finish_run(run_id, TaskStatus.FAILED, f"Workflow failed: {exc}")
            raise

    async def run_pull_request_maintenance(
        self,
        *,
        pr_context: PullRequestContext | None = None,
        event_path: Path | None = None,
        pr_number: int | None = None,
    ) -> WorkflowReport:
        started_at = datetime.now(timezone.utc)
        run_id = await self.state_store.start_run(
            "pull_request_maintenance",
            str(self.config.repo_root),
            started_at=started_at,
        )
        initial_state: WorkflowState = {
            "run_id": run_id,
            "workflow_name": "pull_request_maintenance",
            "started_at": started_at,
            "errors": [],
        }
        if pr_context is not None:
            initial_state["pr_context"] = pr_context
        if event_path is not None:
            initial_state["event_path"] = str(event_path)
        if pr_number is not None:
            initial_state["pr_number_override"] = pr_number
        try:
            graph = self._build_pull_request_graph()
            state = await graph.ainvoke(initial_state)
            return state["report"]
        except Exception as exc:
            await self.state_store.finish_run(run_id, TaskStatus.FAILED, f"Workflow failed: {exc}")
            raise

    async def publish_pull_request_results(
        self,
        *,
        report_path: Path | None = None,
        publish_context_path: Path | None = None,
    ) -> WorkflowReport:
        publish_context, report = await self._load_pr_publish_inputs(
            report_path=report_path,
            publish_context_path=publish_context_path,
        )
        pr_context = publish_context.pr_context
        workspace = await self.github_adapter.prepare_workspace(self.config.repo_root, pr_context)
        capabilities = await self.github_adapter.resolve_capabilities(
            pr_context,
            self.config.pr_workflow.allow_companion_pr,
        )
        artifact_references = await self.github_adapter.resolve_run_artifacts(
            pr_context,
            publish_context.artifact_name,
            publish_context.artifact_references,
        )
        publish_reasons = sorted(set(publish_context.publish_reasons + capabilities.reasons))
        publish_mode = (
            PublishMode.COMPANION_PR
            if publish_context.safe_to_publish
            and capabilities.can_push_fix_branch
            and capabilities.can_open_companion_pr
            else PublishMode.COMMENT_ONLY if capabilities.can_comment else PublishMode.ARTIFACT_ONLY
        )

        companion_pr_url: str | None = None
        companion_result: dict[str, object] = {"status": "skipped", "reason": "publish-mode"}
        if (
            publish_mode == PublishMode.COMPANION_PR
            and report.maintenance_result is not None
            and report.maintenance_result.patch is not None
            and report.maintenance_result.patch.file_patches
            and publish_context.branch_name
        ):
            await self.patch_service.apply(workspace, report.maintenance_result.patch)
            fix_branch_result = await self.github_adapter.publish_fix_branch(
                pr_context,
                workspace,
                publish_context.branch_name,
                report.maintenance_result.patch.touched_files,
            )
            if fix_branch_result.get("status") == "published":
                companion_payload = build_companion_pr_payload(
                    pr_context,
                    publish_context.branch_name,
                    report,
                    self.config.github.companion_pr_label,
                )
                companion_result = await self.github_adapter.create_or_update_companion_pr(
                    pr_context,
                    companion_payload,
                )
                if companion_result.get("status") == "published":
                    companion_pr_url = str(companion_result.get("url")) if companion_result.get("url") else None
                else:
                    publish_reasons.append("companion-pr-publish-failed")
                    publish_mode = PublishMode.COMMENT_ONLY if capabilities.can_comment else PublishMode.ARTIFACT_ONLY
            else:
                publish_reasons.append("fix-branch-publish-failed")
                publish_mode = PublishMode.COMMENT_ONLY if capabilities.can_comment else PublishMode.ARTIFACT_ONLY

        existing_inline_markers = (
            await self.github_adapter.find_existing_inline_comment_markers(pr_context)
            if capabilities.can_inline_review
            else set()
        )
        review_payload = build_review_payload(
            report,
            pr_context,
            publish_mode,
            artifact_references,
            companion_pr_url=companion_pr_url,
            publish_reasons=publish_reasons,
            existing_inline_markers=existing_inline_markers,
            inline_comment_limit=self.config.pr_workflow.inline_comment_limit,
        )

        review_result = (
            await self.github_adapter.create_or_update_summary_comment(pr_context, review_payload)
            if capabilities.can_comment
            else {"status": "skipped", "reason": "comment-capability-disabled"}
        )
        inline_comment_result = (
            await self.github_adapter.publish_inline_comments(pr_context, review_payload.inline_comments)
            if capabilities.can_inline_review
            else {"status": "skipped", "reason": "inline-capability-disabled", "published": 0}
        )

        report.metadata.update(
            {
                "review_comment_id": review_result.get("id"),
                "review_comment_url": review_result.get("url"),
                "companion_pr_url": companion_pr_url,
                "artifact_urls": {
                    reference.name: reference.url or reference.fallback_url
                    for reference in artifact_references
                },
                "publish_mode": publish_mode.value,
                "publish_reasons": sorted(set(publish_reasons)),
                "inline_comments_published": inline_comment_result.get("published", 0),
            }
        )
        if report.report_dir:
            report_dir = Path(report.report_dir)
            await write_json(report_dir / "report.json", report)
            await write_json(report_dir / "artifacts" / "review_payload.json", review_payload)
            summary_markdown = await self.file_store.read_text(report_dir / "summary.md")
            await self.github_adapter.write_step_summary(summary_markdown)
        return report

    async def latest_report_path(self) -> Path | None:
        latest = await self.run_history.latest(str(self.config.repo_root))
        if latest is None or not latest.get("report_dir"):
            return None
        return Path(str(latest["report_dir"]))

    def _build_context(
        self,
        run_id: str,
        working_repo_root: Path,
        repo_root: Path | None = None,
        execution_environment: ExecutionEnvironment | None = None,
        active_skill=None,
        candidate_skill=None,
    ) -> RunContext:
        return RunContext(
            run_id=run_id,
            repo_root=repo_root or self.config.repo_root,
            working_repo_root=working_repo_root,
            config=self.config,
            state_store=self.state_store,
            logger=self.logger,
            rules=self.rules,
            execution_environment=execution_environment,
            active_skill=active_skill,
            candidate_skill=candidate_skill,
        )

    async def _prepare_execution_environment(
        self,
        run_id: str,
        source_repo_root: Path,
    ) -> ExecutionEnvironment | None:
        if not self.config.environment.enabled:
            return None
        report_dir = self.config.reports_dir / run_id
        await self.file_store.ensure_dir(report_dir)
        return await self.environment_manager.prepare(
            report_dir=report_dir,
            source_repo_root=source_repo_root,
            config=self.config,
        )

    async def _prepare_run_skills(
        self,
        repo_root: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return await self.skill_manager.resolve_run_skills(
            repo_root,
            {
                AgentKind.STATIC_REVIEW: self.config.agents.static,
                AgentKind.DYNAMIC_DEBUG: self.config.agents.dynamic,
                AgentKind.MAINTENANCE: self.config.agents.maintenance,
            },
        )

    def _source_repo_root(self, state: WorkflowState) -> Path:
        if "pr_repo_root" in state:
            return Path(state["pr_repo_root"])
        return self.config.repo_root

    def _analysis_root(self, state: WorkflowState) -> Path:
        environment = state.get("execution_environment")
        if environment is not None:
            return Path(environment.base_workspace_root)
        return self._source_repo_root(state)

    def _maintenance_root(self, state: WorkflowState) -> Path:
        environment = state.get("execution_environment")
        if environment is not None:
            return Path(environment.maintenance_workspace_root)
        return self._analysis_root(state)

    def _validation_root(self, state: WorkflowState) -> Path:
        environment = state.get("execution_environment")
        if environment is not None:
            return Path(environment.validation_workspace_root)
        if "validation_workspace_root" in state:
            return Path(state["validation_workspace_root"])
        return self._analysis_root(state)

    def _skill_for_agent(self, state: WorkflowState, agent_kind: AgentKind):
        active_skill = state.get("active_skills", {}).get(agent_kind.value)
        candidate_skill = state.get("candidate_skills", {}).get(agent_kind.value)
        return active_skill, candidate_skill

    async def _refresh_candidate_skills(
        self,
        state: WorkflowState,
        repo_root: Path,
    ) -> dict[str, object]:
        refreshed = dict(state.get("candidate_skills", {}))
        for agent_kind in AgentKind:
            candidate = await self.state_store.get_open_skill_candidate(str(repo_root), agent_kind)
            if candidate is not None:
                refreshed[agent_kind.value] = candidate
            elif agent_kind.value in refreshed:
                refreshed.pop(agent_kind.value, None)
        return refreshed

    def _environment_metadata(
        self,
        environment: ExecutionEnvironment | None,
    ) -> dict[str, object]:
        if environment is None:
            return {}
        return {
            "environment_status": environment.status,
            "environment_degraded": environment.degraded,
            "dependency_sources": list(environment.detected_sources),
            "install_commands": list(environment.install_commands),
            "install_failures": list(environment.install_errors),
            "runtime_root": environment.runtime_root,
            "venv_root": environment.venv_root,
            "base_workspace_root": environment.base_workspace_root,
            "maintenance_workspace_root": environment.maintenance_workspace_root,
            "validation_workspace_root": environment.validation_workspace_root,
            "environment_json_path": environment.environment_json_path,
            "install_log_path": environment.install_log_path,
            "bootstrap_packages": list(environment.bootstrap_packages),
        }

    async def _skill_report_metadata(self, state: WorkflowState, report: WorkflowReport) -> dict[str, object]:
        candidate_skills = await self._refresh_candidate_skills(
            state,
            self._source_repo_root(state),
        )
        return await self.skill_evolution.evaluate_report(
            repo_root=str(self._source_repo_root(state)),
            report=report,
            active_skills=dict(state.get("active_skills", {})),
            candidate_skills=candidate_skills,
        )

    def _dynamic_commands(self) -> list[str]:
        return [*self.config.dynamic_debug.smoke_commands, *self.config.dynamic_debug.test_commands]

    def _build_maintenance_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("scan", self._node_scan_maintenance)
        graph.add_node("static_review", self._node_static_review)
        graph.add_node("dynamic_debug", self._node_dynamic_debug)
        graph.add_node("feedback_merge", self._node_feedback_merge)
        graph.add_node("maintenance", self._node_maintenance)
        graph.add_node("static_validate", self._node_static_validate)
        graph.add_node("dynamic_validate", self._node_dynamic_validate)
        graph.add_node("persist_report", self._node_persist_maintenance_report)
        graph.add_edge(START, "scan")
        graph.add_edge("scan", "static_review")
        graph.add_edge("scan", "dynamic_debug")
        graph.add_edge("static_review", "feedback_merge")
        graph.add_edge("dynamic_debug", "feedback_merge")
        graph.add_edge("feedback_merge", "maintenance")
        graph.add_edge("maintenance", "static_validate")
        graph.add_edge("maintenance", "dynamic_validate")
        graph.add_edge("static_validate", "persist_report")
        graph.add_edge("dynamic_validate", "persist_report")
        graph.add_edge("persist_report", END)
        return graph.compile()

    def _build_incident_debug_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("scan", self._node_scan_incident_debug)
        graph.add_node("dynamic_debug", self._node_dynamic_debug)
        graph.add_node("persist_report", self._node_persist_incident_report)
        graph.add_edge(START, "scan")
        graph.add_edge("scan", "dynamic_debug")
        graph.add_edge("dynamic_debug", "persist_report")
        graph.add_edge("persist_report", END)
        return graph.compile()

    def _build_pull_request_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("load_pr_context", self._node_load_pr_context)
        graph.add_node("checkout_pr_repo", self._node_checkout_pr_repo)
        graph.add_node("scan_pr_repo", self._node_scan_pr_repo)
        graph.add_node("static_review", self._node_static_review)
        graph.add_node("dynamic_debug", self._node_dynamic_debug)
        graph.add_node("feedback_merge", self._node_feedback_merge)
        graph.add_node("maintenance", self._node_maintenance)
        graph.add_node("static_validate", self._node_static_validate)
        graph.add_node("dynamic_validate", self._node_dynamic_validate)
        graph.add_node("assess_publishability", self._node_assess_publishability)
        graph.add_node("persist_pr_report", self._node_persist_pr_report)
        graph.add_node("finalize_pr_run", self._node_finalize_pr_run)
        graph.add_edge(START, "load_pr_context")
        graph.add_edge("load_pr_context", "checkout_pr_repo")
        graph.add_edge("checkout_pr_repo", "scan_pr_repo")
        graph.add_edge("scan_pr_repo", "static_review")
        graph.add_edge("scan_pr_repo", "dynamic_debug")
        graph.add_edge("static_review", "feedback_merge")
        graph.add_edge("dynamic_debug", "feedback_merge")
        graph.add_edge("feedback_merge", "maintenance")
        graph.add_edge("maintenance", "static_validate")
        graph.add_edge("maintenance", "dynamic_validate")
        graph.add_edge("static_validate", "assess_publishability")
        graph.add_edge("dynamic_validate", "assess_publishability")
        graph.add_edge("assess_publishability", "persist_pr_report")
        graph.add_edge("persist_pr_report", "finalize_pr_run")
        graph.add_edge("finalize_pr_run", END)
        return graph.compile()

    async def _node_scan_maintenance(self, state: WorkflowState) -> WorkflowState:
        environment = await self._prepare_execution_environment(state["run_id"], self.config.repo_root)
        active_skills, candidate_skills = await self._prepare_run_skills(self.config.repo_root)
        snapshot, change_set = await self.scan_repository()
        await self.state_store.save_snapshot(state["run_id"], snapshot)
        review_targets = sorted(set(change_set.changed_files + change_set.added_files))
        static_task, dynamic_task = self.dispatcher.create_review_tasks(
            run_id=state["run_id"],
            targets=review_targets,
            commands=self._dynamic_commands(),
        )
        return {
            "snapshot": snapshot,
            "change_set": change_set,
            "static_task": static_task,
            "dynamic_task": dynamic_task,
            "execution_environment": environment,
            "active_skills": active_skills,
            "candidate_skills": candidate_skills,
            "report_dir": environment.report_dir if environment is not None else str(self.config.reports_dir / state["run_id"]),
        }

    async def _node_scan_incident_debug(self, state: WorkflowState) -> WorkflowState:
        environment = await self._prepare_execution_environment(state["run_id"], self.config.repo_root)
        active_skills, candidate_skills = await self._prepare_run_skills(self.config.repo_root)
        snapshot, change_set = await self.scan_repository()
        await self.state_store.save_snapshot(state["run_id"], snapshot)
        dynamic_task = Task(
            task_id=self.dispatcher._new_task_id(),
            run_id=state["run_id"],
            agent_kind=AgentKind.DYNAMIC_DEBUG,
            task_type=TaskType.DYNAMIC_DEBUG,
            targets=[],
            payload={"commands": self._dynamic_commands()},
        )
        return {
            "snapshot": snapshot,
            "change_set": change_set,
            "dynamic_task": dynamic_task,
            "execution_environment": environment,
            "active_skills": active_skills,
            "candidate_skills": candidate_skills,
            "report_dir": environment.report_dir if environment is not None else str(self.config.reports_dir / state["run_id"]),
        }

    async def _node_load_pr_context(self, state: WorkflowState) -> WorkflowState:
        if "pr_context" in state:
            return {}
        event_path = state.get("event_path")
        if not event_path:
            raise ValueError("PR workflow requires an event path or explicit PullRequestContext.")
        pr_context = await self.github_adapter.load_pr_context(
            Path(event_path),
            pr_number_override=state.get("pr_number_override"),
        )
        if pr_context is None:
            raise ValueError("Unable to load pull request context from GitHub event payload.")
        return {"pr_context": pr_context}

    async def _node_checkout_pr_repo(self, state: WorkflowState) -> WorkflowState:
        workspace = await self.github_adapter.prepare_workspace(
            self.config.repo_root,
            state["pr_context"],
        )
        return {"pr_repo_root": str(workspace)}

    async def _node_scan_pr_repo(self, state: WorkflowState) -> WorkflowState:
        source_root = Path(state["pr_repo_root"])
        environment = await self._prepare_execution_environment(state["run_id"], source_root)
        active_skills, candidate_skills = await self._prepare_run_skills(source_root)
        snapshot, change_set = await self.scan_repository(source_root, state["pr_context"])
        await self.state_store.save_snapshot(state["run_id"], snapshot)
        review_targets = sorted(set(change_set.changed_files + change_set.added_files))
        static_task, dynamic_task = self.dispatcher.create_review_tasks(
            run_id=state["run_id"],
            targets=review_targets,
            commands=self._dynamic_commands(),
        )
        return {
            "snapshot": snapshot,
            "change_set": change_set,
            "static_task": static_task,
            "dynamic_task": dynamic_task,
            "execution_environment": environment,
            "active_skills": active_skills,
            "candidate_skills": candidate_skills,
            "report_dir": environment.report_dir if environment is not None else str(self.config.reports_dir / state["run_id"]),
        }

    async def _node_static_review(self, state: WorkflowState) -> WorkflowState:
        task = state["validation_static_task"] if "validation_static_task" in state else state["static_task"]
        root = self._validation_root(state) if task.task_type == TaskType.VALIDATION_STATIC else self._analysis_root(state)
        active_skill, candidate_skill = self._skill_for_agent(state, AgentKind.STATIC_REVIEW)
        context = self._build_context(
            state["run_id"],
            root,
            repo_root=self._source_repo_root(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
        )
        result = await self.execute_task(task, context)
        if task.task_type == TaskType.VALIDATION_STATIC:
            return {"validation_static_result": result}
        return {"static_result": result}

    async def _node_dynamic_debug(self, state: WorkflowState) -> WorkflowState:
        task = state["validation_dynamic_task"] if "validation_dynamic_task" in state else state["dynamic_task"]
        root = self._validation_root(state) if task.task_type == TaskType.VALIDATION_DYNAMIC else self._analysis_root(state)
        active_skill, candidate_skill = self._skill_for_agent(state, AgentKind.DYNAMIC_DEBUG)
        context = self._build_context(
            state["run_id"],
            root,
            repo_root=self._source_repo_root(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
        )
        result = await self.execute_task(task, context)
        if task.task_type == TaskType.VALIDATION_DYNAMIC:
            return {"validation_dynamic_result": result}
        return {"dynamic_result": result}

    async def _node_feedback_merge(self, state: WorkflowState) -> WorkflowState:
        initial_findings = [*state["static_result"].findings, *state["dynamic_result"].findings]
        await self.issue_catalog.record_findings(
            str(self._source_repo_root(state)),
            state["run_id"],
            initial_findings,
        )
        handoffs = [
            *[
                item
                for item in state["static_result"].artifacts.get("handoffs", [])
                if isinstance(item, dict)
            ],
            *[
                item
                for item in state["dynamic_result"].artifacts.get("handoffs", [])
                if isinstance(item, dict)
            ],
        ]
        feedback = FeedbackBundle(
            snapshot=state["snapshot"],
            change_set=state["change_set"],
            static_findings=state["static_result"].findings,
            dynamic_findings=state["dynamic_result"].findings,
        )
        maintenance_task = self.dispatcher.create_maintenance_task(
            state["run_id"],
            feedback,
            handoffs=handoffs,
        )
        return {
            "feedback": feedback,
            "maintenance_task": maintenance_task,
            "candidate_skills": await self._refresh_candidate_skills(
                state,
                self._source_repo_root(state),
            ),
            "artifacts": {"handoff_count": len(handoffs)},
        }

    async def _node_maintenance(self, state: WorkflowState) -> WorkflowState:
        maintenance_root = self._maintenance_root(state)
        active_skill, candidate_skill = self._skill_for_agent(state, AgentKind.MAINTENANCE)
        context = self._build_context(
            state["run_id"],
            maintenance_root,
            repo_root=self._source_repo_root(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
        )
        result = await self.execute_task(state["maintenance_task"], context)
        updates: WorkflowState = {
            "maintenance_result": result,
            "candidate_skills": await self._refresh_candidate_skills(
                state,
                self._source_repo_root(state),
            ),
        }
        patch = result.patch
        if patch is not None and patch.file_patches:
            if state.get("execution_environment") is not None:
                await self.patch_service.apply(maintenance_root, patch)
                validation_workspace_root = await self.environment_manager.refresh_validation_workspace(
                    state["execution_environment"]
                )
            else:
                base_root = self._analysis_root(state)
                validation_workspace_root = await self.file_store.materialize_workspace_copy(base_root)
                await self.patch_service.apply(validation_workspace_root, patch)
            validation_snapshot = await self.scanner.scan(validation_workspace_root)
            static_task, dynamic_task = self.dispatcher.create_validation_tasks(
                run_id=state["run_id"],
                patch=patch,
                commands=self._dynamic_commands(),
            )
            updates["validation_workspace_root"] = str(validation_workspace_root)
            updates["validation_snapshot"] = validation_snapshot
            updates["validation_static_task"] = static_task
            updates["validation_dynamic_task"] = dynamic_task
        return updates

    async def _node_static_validate(self, state: WorkflowState) -> WorkflowState:
        task = state.get("validation_static_task")
        if task is None:
            return {
                "validation_static_result": AgentResult(
                    task_id="validation-static-skipped",
                    agent_kind=AgentKind.STATIC_REVIEW,
                    task_type=TaskType.VALIDATION_STATIC,
                    status=TaskStatus.SKIPPED,
                    summary="Static validation skipped because no patch was generated.",
                )
            }
        return await self._node_static_review(state)

    async def _node_dynamic_validate(self, state: WorkflowState) -> WorkflowState:
        task = state.get("validation_dynamic_task")
        if task is None:
            return {
                "validation_dynamic_result": AgentResult(
                    task_id="validation-dynamic-skipped",
                    agent_kind=AgentKind.DYNAMIC_DEBUG,
                    task_type=TaskType.VALIDATION_DYNAMIC,
                    status=TaskStatus.SKIPPED,
                    summary="Dynamic validation skipped because no patch was generated.",
                )
            }
        return await self._node_dynamic_debug(state)

    async def _node_assess_publishability(self, state: WorkflowState) -> WorkflowState:
        reasons = list(state["maintenance_result"].artifacts.get("unpublishable_reasons", []))
        validation_results = {
            "static_validation": state["validation_static_result"],
            "dynamic_validation": state["validation_dynamic_result"],
        }
        patch = state["maintenance_result"].patch
        if patch is not None and patch.file_patches:
            static_original = [
                finding
                for finding in state["feedback"].static_findings
                if finding.path in set(patch.touched_files)
            ]
            dynamic_original = list(state["feedback"].dynamic_findings)
            if "comparison" not in state["validation_static_result"].artifacts:
                state["validation_static_result"].artifacts["comparison"] = await self.issue_catalog.reconcile(
                    str(self._source_repo_root(state)),
                    state["run_id"],
                    static_original,
                    state["validation_static_result"].findings,
                )
            if "comparison" not in state["validation_dynamic_result"].artifacts:
                state["validation_dynamic_result"].artifacts["comparison"] = await self.issue_catalog.reconcile(
                    str(self._source_repo_root(state)),
                    state["run_id"],
                    dynamic_original,
                    state["validation_dynamic_result"].findings,
                )
        if any(result.status == TaskStatus.FAILED for result in validation_results.values()):
            reasons.append("validation_task_failed")
        for result in validation_results.values():
            if any(finding.severity in {Severity.HIGH, Severity.CRITICAL} for finding in result.findings):
                reasons.append("validation_high_severity_findings")
                break
            comparison = result.artifacts.get("comparison", {})
            if comparison.get("regressed"):
                reasons.append("validation_regressed_findings")
                break

        safe_to_publish = not reasons and bool(
            state["maintenance_result"].artifacts.get("publishable", False)
        )
        publish_mode = (
            PublishMode.COMPANION_PR if safe_to_publish else
            PublishMode.COMMENT_ONLY if self.github_adapter.token else PublishMode.ARTIFACT_ONLY
        )
        branch_name = (
            f"{self.config.github.bot_branch_prefix}/"
            f"{state['pr_context'].pr_number}/{state['run_id'][:8]}"
        )
        publish_reasons = sorted(set(reasons))
        state["maintenance_result"].artifacts["unpublishable_reasons"] = publish_reasons
        state["maintenance_result"].artifacts["publishable"] = safe_to_publish
        return {
            "safe_to_publish": safe_to_publish,
            "publish_mode": publish_mode,
            "companion_pr_required": publish_mode == PublishMode.COMPANION_PR,
            "branch_name": branch_name,
            "artifacts": {"publish_reasons": publish_reasons},
        }

    async def _node_persist_maintenance_report(self, state: WorkflowState) -> WorkflowState:
        validation_results = {
            "static_validation": state["validation_static_result"],
            "dynamic_validation": state["validation_dynamic_result"],
        }
        patch = state["maintenance_result"].patch
        if patch is not None and patch.file_patches:
            static_original = [
                finding
                for finding in state["feedback"].static_findings
                if finding.path in set(patch.touched_files)
            ]
            dynamic_original = list(state["feedback"].dynamic_findings)
            static_comparison = await self.issue_catalog.reconcile(
                str(self._source_repo_root(state)),
                state["run_id"],
                static_original,
                state["validation_static_result"].findings,
            )
            dynamic_comparison = await self.issue_catalog.reconcile(
                str(self._source_repo_root(state)),
                state["run_id"],
                dynamic_original,
                state["validation_dynamic_result"].findings,
            )
            state["validation_static_result"].artifacts["comparison"] = static_comparison
            state["validation_dynamic_result"].artifacts["comparison"] = dynamic_comparison

        report = WorkflowReport(
            run_id=state["run_id"],
            workflow_name="maintenance_loop",
            repo_root=str(self._source_repo_root(state)),
            started_at=state["started_at"],
            finished_at=datetime.now(timezone.utc),
            status=self._workflow_status(
                state["static_result"],
                state["dynamic_result"],
                state["maintenance_result"],
                validation_results,
            ),
            snapshot=state["snapshot"],
            change_set=state["change_set"],
            static_result=state["static_result"],
            dynamic_result=state["dynamic_result"],
            maintenance_result=state["maintenance_result"],
            validation_results=validation_results,
            report_dir=str(state.get("report_dir", "")),
            metadata=self._environment_metadata(state.get("execution_environment")),
        )
        report.metadata.update(await self._skill_report_metadata(state, report))
        report_dir = await self._write_report_artifacts(report)
        report.report_dir = str(report_dir)
        await self.state_store.finish_run(
            state["run_id"],
            report.status,
            self._report_summary(report),
            str(report_dir),
        )
        return {"report": report, "report_dir": str(report_dir)}

    async def _node_persist_incident_report(self, state: WorkflowState) -> WorkflowState:
        await self.issue_catalog.record_findings(
            str(self._source_repo_root(state)),
            state["run_id"],
            state["dynamic_result"].findings,
        )
        report = WorkflowReport(
            run_id=state["run_id"],
            workflow_name="incident_debug",
            repo_root=str(self._source_repo_root(state)),
            started_at=state["started_at"],
            finished_at=datetime.now(timezone.utc),
            status=state["dynamic_result"].status,
            snapshot=state["snapshot"],
            change_set=state["change_set"],
            dynamic_result=state["dynamic_result"],
            report_dir=str(state.get("report_dir", "")),
            metadata=self._environment_metadata(state.get("execution_environment")),
        )
        report.metadata.update(await self._skill_report_metadata(state, report))
        report_dir = await self._write_report_artifacts(report)
        report.report_dir = str(report_dir)
        await self.state_store.finish_run(
            state["run_id"],
            report.status,
            self._report_summary(report),
            str(report_dir),
        )
        return {"report": report, "report_dir": str(report_dir)}

    async def _node_persist_pr_report(self, state: WorkflowState) -> WorkflowState:
        validation_results = {
            "static_validation": state["validation_static_result"],
            "dynamic_validation": state["validation_dynamic_result"],
        }
        patch = state["maintenance_result"].patch
        if patch is not None and patch.file_patches:
            static_original = [
                finding
                for finding in state["feedback"].static_findings
                if finding.path in set(patch.touched_files)
            ]
            dynamic_original = list(state["feedback"].dynamic_findings)
            static_comparison = await self.issue_catalog.reconcile(
                str(self._source_repo_root(state)),
                state["run_id"],
                static_original,
                state["validation_static_result"].findings,
            )
            dynamic_comparison = await self.issue_catalog.reconcile(
                str(self._source_repo_root(state)),
                state["run_id"],
                dynamic_original,
                state["validation_dynamic_result"].findings,
            )
            state["validation_static_result"].artifacts["comparison"] = static_comparison
            state["validation_dynamic_result"].artifacts["comparison"] = dynamic_comparison

        metadata = {
            "pr_number": state["pr_context"].pr_number,
            "head_sha": state["pr_context"].head_sha,
            "publish_mode": state["publish_mode"].value,
            "publish_reasons": list(state.get("artifacts", {}).get("publish_reasons", [])),
        }
        metadata.update(self._environment_metadata(state.get("execution_environment")))
        report = self._build_pr_report(state, metadata=metadata)
        report.metadata.update(await self._skill_report_metadata(state, report))
        report_dir = await self._write_report_artifacts(report)
        report.report_dir = str(report_dir)
        artifact_references = self._artifact_references(report_dir, include_patch=bool(patch and patch.diff_text))
        publish_context = PublishContext(
            run_id=state["run_id"],
            report_dir=str(report_dir),
            report_path=str(report_dir / "report.json"),
            artifact_name=f"close-devs-report-{state['run_id']}",
            artifact_retention_days=self.config.github.artifact_retention_days,
            pr_context=state["pr_context"],
            publish_mode=state["publish_mode"],
            safe_to_publish=state["safe_to_publish"],
            branch_name=state.get("branch_name"),
            companion_pr_required=state.get("companion_pr_required", False),
            publish_reasons=list(state.get("artifacts", {}).get("publish_reasons", [])),
            artifact_references=artifact_references,
        )
        await write_json(report_dir / "artifacts" / "publish_context.json", publish_context)
        report.metadata["publish_context_path"] = str(report_dir / "artifacts" / "publish_context.json")
        await write_json(report_dir / "report.json", report)
        return {
            "report": report,
            "report_dir": str(report_dir),
            "publish_context": publish_context,
            "artifacts": {"publish_context_path": str(report_dir / "artifacts" / "publish_context.json")},
        }

    async def _node_finalize_pr_run(self, state: WorkflowState) -> WorkflowState:
        report = state["report"]
        await self.state_store.finish_run(
            state["run_id"],
            report.status,
            self._report_summary(report),
            report.report_dir,
        )
        return {"report": report}

    def _build_pr_report(
        self,
        state: WorkflowState,
        *,
        metadata: dict[str, object] | None = None,
    ) -> WorkflowReport:
        validation_results = {
            "static_validation": state["validation_static_result"],
            "dynamic_validation": state["validation_dynamic_result"],
        }
        return WorkflowReport(
            run_id=state["run_id"],
            workflow_name="pull_request_maintenance",
            repo_root=str(self._source_repo_root(state)),
            started_at=state["started_at"],
            finished_at=datetime.now(timezone.utc),
            status=self._workflow_status(
                state["static_result"],
                state["dynamic_result"],
                state["maintenance_result"],
                validation_results,
            ),
            snapshot=state["snapshot"],
            change_set=state["change_set"],
            static_result=state["static_result"],
            dynamic_result=state["dynamic_result"],
            maintenance_result=state["maintenance_result"],
            validation_results=validation_results,
            report_dir=str(state.get("report_dir", "")),
            metadata=dict(metadata or {}),
        )

    async def _load_pr_publish_inputs(
        self,
        *,
        report_path: Path | None,
        publish_context_path: Path | None,
    ) -> tuple[PublishContext, WorkflowReport]:
        resolved_publish_context_path = publish_context_path
        if resolved_publish_context_path is None and report_path is not None:
            resolved_publish_context_path = report_path.parent / "artifacts" / "publish_context.json"
        if resolved_publish_context_path is None:
            raise ValueError("PR publish requires --publish-context or --report.")

        publish_context = publish_context_from_dict(
            await read_json(resolved_publish_context_path)
        )
        resolved_report_path = report_path or Path(publish_context.report_path)
        report = workflow_report_from_dict(await read_json(resolved_report_path))
        report.report_dir = publish_context.report_dir
        return publish_context, report

    def _artifact_references(self, report_dir: Path, *, include_patch: bool) -> list[ArtifactReference]:
        references = [
            ArtifactReference(name="summary_md", path="summary.md"),
            ArtifactReference(name="report_json", path="report.json"),
            ArtifactReference(name="findings_json", path="findings.json"),
            ArtifactReference(name="environment_json", path="artifacts/environment.json"),
            ArtifactReference(name="install_log", path="artifacts/install.log"),
        ]
        if include_patch:
            references.append(ArtifactReference(name="patch_diff", path="patch.diff"))
        workflow_url = self.github_adapter.workflow_run_url()
        if workflow_url:
            references.append(
                ArtifactReference(
                    name="workflow_run",
                    path="Actions run",
                    url=workflow_url,
                    fallback_url=workflow_url,
                )
            )
        return references

    def _workflow_status(self, *results: object) -> TaskStatus:
        for item in results:
            if isinstance(item, dict):
                if any(result.status == TaskStatus.FAILED for result in item.values()):
                    return TaskStatus.FAILED
                continue
            if item is not None and getattr(item, "status", None) == TaskStatus.FAILED:
                return TaskStatus.FAILED
        return TaskStatus.SUCCEEDED

    def _report_summary(self, report: WorkflowReport) -> str:
        parts = [
            f"static={len(report.static_result.findings) if report.static_result else 0}",
            f"dynamic={len(report.dynamic_result.findings) if report.dynamic_result else 0}",
            (
                f"patches={len(report.maintenance_result.patch.file_patches)}"
                if report.maintenance_result and report.maintenance_result.patch
                else "patches=0"
            ),
        ]
        if report.metadata.get("publish_mode"):
            parts.append(f"publish_mode={report.metadata['publish_mode']}")
        return ", ".join(parts)

    async def _write_report_artifacts(self, report: WorkflowReport) -> Path:
        report_dir = Path(report.report_dir) if report.report_dir else self.config.reports_dir / report.run_id
        await self.file_store.ensure_dir(report_dir)
        report.report_dir = str(report_dir)
        await write_markdown(report_dir / "summary.md", report)
        await write_json(report_dir / "report.json", report)
        await write_json(report_dir / "findings.json", report.all_findings)
        if report.maintenance_result and report.maintenance_result.patch:
            await self.file_store.write_text(
                report_dir / "patch.diff",
                report.maintenance_result.patch.diff_text,
            )
        await self.file_store.ensure_dir(report_dir / "artifacts")
        return report_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Close-Devs CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        default="config/default.toml",
        help="Path to the TOML configuration file.",
    )
    common_parser.add_argument(
        "--repo",
        default=None,
        help="Target repository root. Defaults to the config value.",
    )

    subparsers.add_parser("run-once", parents=[common_parser], help="Run the maintenance loop once.")
    subparsers.add_parser("scan", parents=[common_parser], help="Scan the repository and print the change summary.")
    subparsers.add_parser("review", parents=[common_parser], help="Run static review only.")
    subparsers.add_parser("debug", parents=[common_parser], help="Run dynamic debug only.")

    pr_parser = subparsers.add_parser(
        "pr-review",
        parents=[common_parser],
        help="Run GitHub pull request maintenance workflow.",
    )
    pr_parser.add_argument(
        "--event-path",
        required=True,
        help="Path to the GitHub event payload JSON.",
    )
    pr_parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help="Optional PR number override.",
    )

    publish_parser = subparsers.add_parser(
        "pr-publish",
        parents=[common_parser],
        help="Publish Close-Devs PR artifacts and comments after analysis.",
    )
    publish_parser.add_argument(
        "--event-path",
        default=None,
        help="Optional GitHub event payload path for publish-time context overrides.",
    )
    publish_parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help="Optional PR number override for publish-time context overrides.",
    )
    publish_inputs = publish_parser.add_mutually_exclusive_group(required=True)
    publish_inputs.add_argument(
        "--publish-context",
        default=None,
        help="Path to the publish context JSON generated by pr-review.",
    )
    publish_inputs.add_argument(
        "--report",
        default=None,
        help="Path to the report.json generated by pr-review.",
    )

    report_parser = subparsers.add_parser(
        "report",
        parents=[common_parser],
        help="Show the latest report path.",
    )
    report_parser.add_argument("--show", action="store_true", help="Print report markdown.")

    skill_status_parser = subparsers.add_parser(
        "skill-status",
        parents=[common_parser],
        help="Show active and candidate skill versions.",
    )
    skill_status_parser.add_argument(
        "--agent",
        choices=[item.value for item in AgentKind],
        default=None,
        help="Optional agent filter.",
    )

    skill_history_parser = subparsers.add_parser(
        "skill-history",
        parents=[common_parser],
        help="Show recent skill evaluations.",
    )
    skill_history_parser.add_argument(
        "--agent",
        choices=[item.value for item in AgentKind],
        required=True,
        help="Agent whose skill evaluation history should be shown.",
    )

    skill_freeze_parser = subparsers.add_parser(
        "skill-freeze",
        parents=[common_parser],
        help="Freeze or unfreeze an agent skill binding.",
    )
    skill_freeze_parser.add_argument(
        "--agent",
        choices=[item.value for item in AgentKind],
        required=True,
        help="Agent whose active skill binding should be updated.",
    )
    skill_freeze_parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Unfreeze instead of freezing.",
    )

    skill_promote_parser = subparsers.add_parser(
        "skill-promote",
        parents=[common_parser],
        help="Promote the current open candidate for an agent.",
    )
    skill_promote_parser.add_argument(
        "--agent",
        choices=[item.value for item in AgentKind],
        required=True,
        help="Agent whose candidate should be promoted.",
    )
    return parser


def _print_report(report: WorkflowReport) -> None:
    print(f"Run ID: {report.run_id}")
    print(f"Workflow: {report.workflow_name}")
    print(f"Status: {report.status.value}")
    print(f"Report Dir: {report.report_dir}")
    if report.metadata.get("review_comment_url"):
        print(f"Review URL: {report.metadata['review_comment_url']}")
    if report.metadata.get("companion_pr_url"):
        print(f"Companion PR: {report.metadata['companion_pr_url']}")


async def main_async(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(
        Path(args.config),
        repo_override=Path(args.repo).resolve() if args.repo else None,
    )
    orchestrator = await Orchestrator.create(config)
    try:
        if args.command == "run-once":
            report = await orchestrator.run_workflow("maintenance_loop")
            _print_report(report)
            return 0
        if args.command == "scan":
            snapshot, change_set = await orchestrator.scan_repository()
            print(f"Files tracked: {len(snapshot.files)}")
            print(f"Changed: {len(change_set.changed_files)}")
            print(f"Added: {len(change_set.added_files)}")
            print(f"Removed: {len(change_set.removed_files)}")
            return 0
        if args.command == "review":
            report = await orchestrator.run_static_review_cycle()
            _print_report(report)
            return 0
        if args.command == "debug":
            report = await orchestrator.run_dynamic_debug_cycle()
            _print_report(report)
            return 0
        if args.command == "pr-review":
            report = await orchestrator.run_workflow(
                "pull_request_maintenance",
                event_path=Path(args.event_path),
                pr_number=args.pr_number,
            )
            _print_report(report)
            return 0
        if args.command == "pr-publish":
            report = await orchestrator.publish_pull_request_results(
                report_path=Path(args.report) if args.report else None,
                publish_context_path=Path(args.publish_context) if args.publish_context else None,
            )
            _print_report(report)
            return 0
        if args.command == "report":
            latest = await orchestrator.latest_report_path()
            if latest is None:
                print("No reports have been generated yet.")
                return 1
            print(latest)
            if args.show:
                summary_path = latest / "summary.md"
                if summary_path.exists():
                    print(summary_path.read_text(encoding="utf-8"))
            return 0
        if args.command == "skill-status":
            rows = await orchestrator.skill_manager.skill_status(config.repo_root)
            if args.agent:
                rows = [row for row in rows if row["agent"] == args.agent]
            for row in rows:
                print(
                    f"{row['agent']}: active={row['active_version']} source={row['active_source']} "
                    f"candidate={row['candidate_version'] or '-'} shadow_runs={row['candidate_shadow_runs']} "
                    f"status={row['candidate_status'] or '-'} frozen={row['frozen']}"
                )
            return 0
        if args.command == "skill-history":
            history = await orchestrator.skill_manager.history(config.repo_root, AgentKind(args.agent))
            for item in history:
                print(
                    f"{item['created_at']} run={item['run_id']} active={item['active_version']} "
                    f"candidate={item['candidate_version'] or '-'} active_score={item['active_score']} "
                    f"candidate_score={item['candidate_score']} promoted={item['promoted']} "
                    f"reasons={','.join(item['reasons']) if item['reasons'] else '-'}"
                )
            return 0
        if args.command == "skill-freeze":
            await orchestrator.skill_manager.freeze(
                config.repo_root,
                AgentKind(args.agent),
                frozen=not bool(args.unfreeze),
            )
            print(f"{args.agent}: {'unfrozen' if args.unfreeze else 'frozen'}")
            return 0
        if args.command == "skill-promote":
            promoted = await orchestrator.skill_manager.manual_promote(
                config.repo_root,
                AgentKind(args.agent),
            )
            print(f"{args.agent}: {'promoted' if promoted else 'no-open-candidate'}")
            return 0
        return 1
    finally:
        await orchestrator.close()


def main(argv: Iterable[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
