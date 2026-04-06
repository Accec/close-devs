from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
from typing import Iterable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.dynamic_debug import DynamicDebugAgent
from agents.maintenance import MaintenanceAgent
from agents.static_review import StaticReviewAgent
from core.config import (
    AgentRuntimeConfig,
    AppConfig,
    LLMConfig,
    default_api_key_env_for_provider,
    default_base_url_for_provider,
    load_config,
    load_rules,
)
from core.dispatcher import TaskDispatcher
from core.logging import configure_logging
from core.models import (
    AgentKind,
    AgentResult,
    ArtifactReference,
    ChangeSet,
    ExecutionEnvironment,
    FeedbackBundle,
    FilePatch,
    Finding,
    StaticContextBundle,
    PatchProposal,
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
from reports.enrichment import enrich_report_snippets
from reports.serializer import publish_context_from_dict, read_json, to_jsonable, workflow_report_from_dict, write_json
from skills.evolution import SkillEvolutionService
from skills.manager import SkillManager
from tools.file_store import FileStore
from tools.environment_manager import EnvironmentManager
from tools.patch_service import PatchService
from tools.agent_toolkit import AgentToolkitFactory
from tools.dependency_audit import parse_dependency_audit_output, summarize_dependency_vulnerabilities
from tools.language_support import build_dependency_audit_adapters, detect_language_profile
from tools.static_context_builder import StaticContextBuilder
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
        self.static_context_builder = StaticContextBuilder(
            toolkit_factory=self.toolkit_factory,
            static_tooling=self.toolkit_factory.static_tooling,
        )
        self.static_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.static),
            self.logger,
        )
        self.dynamic_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.dynamic),
            self.logger,
        )
        self.maintenance_llm_client = build_llm_client(
            self._agent_llm_config(config.agents.maintenance),
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

    def _agent_llm_config(self, runtime_config: AgentRuntimeConfig) -> LLMConfig:
        provider = runtime_config.provider or self.config.llm.provider
        inherited_base_url = (
            self.config.llm.base_url
            if provider == self.config.llm.provider
            else default_base_url_for_provider(provider)
        )
        inherited_api_key_env = (
            self.config.llm.api_key_env
            if provider == self.config.llm.provider
            else default_api_key_env_for_provider(provider)
        )
        return LLMConfig(
            provider=provider,
            model=runtime_config.model or self.config.llm.model,
            base_url=runtime_config.base_url if runtime_config.base_url is not None else inherited_base_url,
            api_key_env=runtime_config.api_key_env if runtime_config.api_key_env is not None else inherited_api_key_env,
            timeout_seconds=runtime_config.timeout_seconds or self.config.llm.timeout_seconds,
            temperature=runtime_config.temperature if runtime_config.temperature is not None else self.config.llm.temperature,
            max_retries=self.config.llm.max_retries,
            system_prompt=runtime_config.system_prompt or self.config.llm.system_prompt,
        )

    def _log_agent_runtime(
        self,
        agent_name: str,
        llm_client: object,
        runtime_config: AgentRuntimeConfig,
    ) -> None:
        self.logger.info(
            "Agent runtime ready: agent=%s provider=%s client=%s model=%s base_url_custom=%s strict_failure=%s max_steps=%s max_tool_calls=%s tools=%s",
            agent_name,
            self._llm_provider_name(llm_client),
            type(llm_client).__name__,
            runtime_config.model or self._agent_llm_config(runtime_config).model,
            bool(getattr(llm_client, "config", None) and getattr(llm_client.config, "base_url", None)),
            True,
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
        repo_identity: str | None = None,
    ) -> tuple[RepoSnapshot, ChangeSet]:
        target_root = repo_root or self.config.repo_root
        identity = repo_identity or self._configured_repo_identity()
        snapshot = await self.scanner.scan(target_root)
        snapshot.repo_root = identity
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

        previous_snapshot = await self.state_store.get_latest_snapshot(identity)
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
                artifacts={
                    "failure_reason": str(exc),
                    "llm_failure_reason": str(exc),
                },
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
            repo_root=context.repo_identity,
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
            self._configured_repo_identity(),
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
            self._configured_repo_identity(),
            started_at=started_at,
        )
        try:
            environment = await self._prepare_execution_environment(
                run_id,
                None if self.config.repo_is_remote else self.config.repo_root,
                repo_source=self._configured_repo_identity(),
                repo_ref=self.config.repo_ref,
            )
            active_skills, candidate_skills = await self._prepare_run_skills(self._configured_repo_identity())
            scan_root = Path(environment.base_workspace_root) if environment is not None else self.config.repo_root
            snapshot, change_set = await self.scan_repository(
                scan_root,
                repo_identity=self._configured_repo_identity(),
            )
            await self.state_store.save_snapshot(run_id, snapshot)
            review_targets = sorted(set(change_set.added_files + change_set.changed_files))
            static_context = await self._build_static_context(
                repo_root=Path(environment.base_workspace_root) if environment is not None else self.config.repo_root,
                targets=review_targets,
                execution_environment=environment,
            )
            analysis_root = (
                Path(environment.base_workspace_root)
                if environment is not None
                else self.config.repo_root
            )
            context = self._build_context(
                run_id,
                working_repo_root=analysis_root,
                repo_root=self.config.repo_root,
                repo_identity=self._configured_repo_identity(),
                execution_environment=environment,
                active_skill=active_skills.get(AgentKind.STATIC_REVIEW.value),
                candidate_skill=candidate_skills.get(AgentKind.STATIC_REVIEW.value),
                startup_topology=static_context.startup_topology,
                project_topology=static_context.project_topology,
                language_profile=static_context.language_profile,
                static_context=static_context,
            )
            task = Task(
                task_id=self.dispatcher._new_task_id(),
                run_id=run_id,
                agent_kind=AgentKind.STATIC_REVIEW,
                task_type=TaskType.STATIC_REVIEW,
                targets=review_targets,
                payload={"static_context": static_context},
            )
            result = await self.execute_task(task, context)
            candidate_skills = await self._refresh_candidate_skills(
                {"candidate_skills": candidate_skills},
                self._configured_repo_identity(),
            )
            await self.issue_catalog.record_findings(self._configured_repo_identity(), run_id, result.findings)
            report = WorkflowReport(
                run_id=run_id,
                workflow_name="static_review",
                repo_root=self._configured_repo_identity(),
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
                await self._dependency_audit_metadata(
                    analysis_root=analysis_root,
                    execution_environment=environment,
                    static_context=static_context,
                )
            )
            await self._enrich_report_for_persistence(
                report,
                analysis_root=analysis_root,
            )
            report.metadata.update(self._runtime_report_metadata(report))
            report.metadata.update(self._static_context_metadata(static_context))
            report.metadata.update(
                await self.skill_evolution.evaluate_report(
                    repo_root=self._configured_repo_identity(),
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
            self._configured_repo_identity(),
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
            self._configured_repo_identity(),
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
        latest = await self.run_history.latest(self._configured_repo_identity())
        if latest is None or not latest.get("report_dir"):
            return None
        return Path(str(latest["report_dir"]))

    def _build_context(
        self,
        run_id: str,
        working_repo_root: Path,
        repo_root: Path | None = None,
        repo_identity: str | None = None,
        execution_environment: ExecutionEnvironment | None = None,
        active_skill=None,
        candidate_skill=None,
        startup_topology=None,
        project_topology=None,
        language_profile=None,
        static_context: StaticContextBundle | None = None,
    ) -> RunContext:
        return RunContext(
            run_id=run_id,
            repo_root=repo_root or self.config.repo_root,
            repo_identity=repo_identity or self._configured_repo_identity(),
            working_repo_root=working_repo_root,
            config=self.config,
            state_store=self.state_store,
            logger=self.logger,
            rules=self.rules,
            execution_environment=execution_environment,
            active_skill=active_skill,
            candidate_skill=candidate_skill,
            startup_topology=startup_topology,
            project_topology=project_topology,
            language_profile=language_profile,
            static_context=static_context,
        )

    async def _prepare_execution_environment(
        self,
        run_id: str,
        source_repo_root: Path | None,
        *,
        repo_source: str,
        repo_ref: str | None = None,
    ) -> ExecutionEnvironment | None:
        if not self.config.environment.enabled:
            if self.config.repo_is_remote:
                raise ValueError("Remote repository sources require environment.enabled = true.")
            return None
        report_dir = self.config.reports_dir / run_id
        await self.file_store.ensure_dir(report_dir)
        return await self.environment_manager.prepare(
            report_dir=report_dir,
            source_repo_root=source_repo_root,
            source_repo=repo_source,
            source_ref=repo_ref,
            config=self.config,
        )

    async def _prepare_run_skills(
        self,
        repo_root: Path | str,
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

    def _configured_repo_identity(self) -> str:
        return self.config.repo_source

    def _repo_identity(self, state: WorkflowState) -> str:
        if "pr_repo_root" in state:
            return str(state["pr_repo_root"])
        return self._configured_repo_identity()

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
        repo_root: Path | str,
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
            "repo_source": environment.source_repo,
            "repo_source_kind": environment.source_kind,
            "repo_ref": environment.source_ref,
            "repo_source_auth": environment.source_auth,
            "dependency_sources": list(environment.detected_sources),
            "install_commands": list(environment.install_commands),
            "install_failures": list(environment.install_errors),
            "environment_installer_summary": dict(environment.installer_summary),
            "runtime_root": environment.runtime_root,
            "venv_root": environment.venv_root,
            "base_workspace_root": environment.base_workspace_root,
            "maintenance_workspace_root": environment.maintenance_workspace_root,
            "validation_workspace_root": environment.validation_workspace_root,
            "environment_json_path": environment.environment_json_path,
            "install_log_path": environment.install_log_path,
            "bootstrap_packages": list(environment.bootstrap_packages),
        }

    async def _build_static_context(
        self,
        *,
        repo_root: Path,
        targets: list[str],
        execution_environment: ExecutionEnvironment | None = None,
    ) -> StaticContextBundle:
        bundle = await self.static_context_builder.build(
            repo_root=repo_root,
            targets=targets,
            config=self.config,
            rules=self.rules,
            execution_environment=execution_environment,
        )
        summary = bundle.summary()
        self.logger.info(
            "Static pre-context ready: target_count=%s top_target_count=%s startup_context_count=%s baseline_digest_count=%s primary_language=%s ecosystems=%s",
            len(targets),
            summary.get("top_target_count", 0),
            summary.get("startup_context_count", 0),
            summary.get("baseline_total_findings", 0),
            summary.get("primary_language", "unknown"),
            ",".join(summary.get("ecosystems", [])),
        )
        return bundle

    def _static_context_metadata(
        self,
        static_context: StaticContextBundle | None,
    ) -> dict[str, object]:
        if static_context is None:
            return {}
        summary = static_context.summary()
        return {
            "static_context_summary": summary,
            "language_profile_summary": {
                "primary_language": static_context.language_profile.primary_language,
                "languages": list(static_context.language_profile.languages),
                "primary_ecosystem": static_context.language_profile.primary_ecosystem,
                "ecosystems": list(static_context.language_profile.ecosystems),
                "enabled_adapters": list(static_context.language_profile.enabled_adapters),
                "generic_review": bool(static_context.language_profile.generic_review),
            },
            "project_topology_summary": summary.get("project_topology_summary", {}),
            "tool_coverage_summary": summary.get("tool_coverage_summary", {}),
            "generic_language_review": bool(static_context.language_profile.generic_review),
        }

    async def _dependency_audit_metadata(
        self,
        *,
        analysis_root: Path,
        execution_environment: ExecutionEnvironment | None,
        static_context: StaticContextBundle | None = None,
    ) -> dict[str, object]:
        if self.config.static_review.dependency_audit_mode == "disabled":
            return {
                "dependency_audit_status": "disabled",
                "dependency_vulnerabilities": [],
                "dependency_vulnerability_summary": {"total": 0, "blockers": 0, "severity_counts": {}, "packages": []},
                "dependency_vulnerability_count": 0,
                "dependency_vulnerability_blocker_count": 0,
                "dependency_audit_ecosystem": "disabled",
            }
        if execution_environment is None:
            return {
                "dependency_audit_status": "unavailable",
                "dependency_audit_error": "execution-environment-unavailable",
                "dependency_vulnerabilities": [],
                "dependency_vulnerability_summary": {"total": 0, "blockers": 0, "severity_counts": {}, "packages": []},
                "dependency_vulnerability_count": 0,
                "dependency_vulnerability_blocker_count": 0,
                "dependency_audit_ecosystem": "unavailable",
            }

        install_env = execution_environment.command_env()
        language_profile = (
            static_context.language_profile
            if static_context is not None
            else detect_language_profile(
                analysis_root,
                enabled_languages=self.config.static_review.language_adapters_enabled,
            )
        )
        adapters = build_dependency_audit_adapters(
            analysis_root,
            language_profile=language_profile,
            mode=self.config.static_review.dependency_audit_mode,
            python_command=self.config.static_review.dependency_audit_command,
        )
        if not adapters:
            return {
                "dependency_audit_status": "unavailable",
                "dependency_audit_error": "no-supported-ecosystem-or-adapter",
                "dependency_vulnerabilities": [],
                "dependency_vulnerability_summary": {"total": 0, "blockers": 0, "severity_counts": {}, "packages": []},
                "dependency_vulnerability_count": 0,
                "dependency_vulnerability_blocker_count": 0,
                "dependency_audit_ecosystem": "unavailable",
            }
        vulnerabilities: list[dict[str, object]] = []
        errors: list[str] = []
        unavailable: list[str] = []
        executed_ecosystems: list[str] = []
        for adapter in adapters:
            executable = shlex.split(adapter.command)[0]
            if shutil.which(executable, path=install_env.get("PATH")) is None:
                unavailable.append(f"{adapter.ecosystem}:{executable}-not-installed")
                continue
            result = await self.toolkit_factory.command_runner.run(
                command=adapter.command,
                cwd=analysis_root,
                timeout_seconds=300,
                env=install_env,
            )
            if result.returncode != 0:
                error_text = result.stderr.strip() or result.stdout.strip() or f"exit-{result.returncode}"
                errors.append(f"{adapter.ecosystem}:{error_text[:500]}")
                continue
            try:
                vulnerabilities.extend(parse_dependency_audit_output(adapter.parser, result.stdout))
                executed_ecosystems.append(adapter.ecosystem)
            except Exception as exc:
                errors.append(f"{adapter.ecosystem}:parse-error:{exc}")

        summary = summarize_dependency_vulnerabilities(vulnerabilities)
        if vulnerabilities:
            status = "executed"
        elif errors:
            status = "failed"
        elif unavailable:
            status = "unavailable"
        else:
            status = "executed"
        metadata: dict[str, object] = {
            "dependency_audit_status": status,
            "dependency_vulnerabilities": vulnerabilities,
            "dependency_vulnerability_summary": summary,
            "dependency_vulnerability_count": len(vulnerabilities),
            "dependency_vulnerability_blocker_count": int(summary.get("blockers", 0)),
            "dependency_audit_ecosystem": ",".join(executed_ecosystems) or ",".join(
                adapter.ecosystem for adapter in adapters
            ),
        }
        if errors:
            metadata["dependency_audit_error"] = "; ".join(errors)[:2000]
        elif unavailable and not vulnerabilities:
            metadata["dependency_audit_error"] = "; ".join(unavailable)[:2000]
        return metadata

    async def _enrich_report_for_persistence(
        self,
        report: WorkflowReport,
        *,
        analysis_root: Path,
        maintenance_root: Path | None = None,
        validation_root: Path | None = None,
    ) -> None:
        await enrich_report_snippets(
            report,
            analysis_root=analysis_root,
            maintenance_root=maintenance_root,
            validation_root=validation_root,
            file_store=self.file_store,
        )

    async def _skill_report_metadata(self, state: WorkflowState, report: WorkflowReport) -> dict[str, object]:
        candidate_skills = await self._refresh_candidate_skills(
            state,
            self._repo_identity(state),
        )
        shadow_replays = await self._shadow_replay_candidates(state, report, candidate_skills)
        return await self.skill_evolution.evaluate_report(
            repo_root=self._repo_identity(state),
            report=report,
            active_skills=dict(state.get("active_skills", {})),
            candidate_skills=candidate_skills,
            shadow_replays=shadow_replays,
        )

    async def _shadow_replay_candidates(
        self,
        state: WorkflowState,
        report: WorkflowReport,
        candidate_skills: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        if not self.config.skills.shadow_evaluation_enabled:
            return {}

        replays: dict[str, dict[str, object]] = {}
        for agent_kind in AgentKind:
            candidate = candidate_skills.get(agent_kind.value)
            if candidate is None:
                continue
            if getattr(candidate, "cooldown_until", None) and candidate.cooldown_until > report.finished_at:
                replays[agent_kind.value] = {
                    "mode": "heuristic",
                    "reasons": ["candidate-cooldown-active"],
                }
                continue
            try:
                replay = await self._shadow_replay_candidate(state, report, agent_kind, candidate)
            except Exception as exc:
                self.logger.warning(
                    "Shadow replay failed: agent=%s candidate=%s error=%s",
                    agent_kind.value,
                    getattr(candidate, "version", "-"),
                    exc,
                )
                replays[agent_kind.value] = {
                    "mode": "heuristic",
                    "reasons": [
                        "shadow-replay-failed",
                        str(exc),
                    ],
                }
                continue
            if replay:
                replays[agent_kind.value] = replay
        return replays

    async def _shadow_replay_candidate(
        self,
        state: WorkflowState,
        report: WorkflowReport,
        agent_kind: AgentKind,
        candidate_skill,
    ) -> dict[str, object] | None:
        task = self._primary_task_for_agent(state, agent_kind)
        if task is None:
            return None
        working_root = await self._shadow_working_root(state, agent_kind)
        context = self._build_context(
            state["run_id"],
            working_root,
            repo_root=self._source_repo_root(state),
            repo_identity=self._repo_identity(state),
            execution_environment=state.get("execution_environment"),
            active_skill=candidate_skill.skill_pack,
            candidate_skill=None,
            startup_topology=state.get("startup_topology"),
            project_topology=state.get("project_topology"),
            language_profile=state.get("language_profile"),
            static_context=state.get("static_context"),
        )
        shadow_task = Task(
            task_id=f"{task.task_id}-shadow-{uuid4().hex[:8]}",
            run_id=task.run_id,
            agent_kind=task.agent_kind,
            task_type=task.task_type,
            targets=list(task.targets),
            payload=dict(task.payload),
        )
        agent = {
            AgentKind.STATIC_REVIEW: self.static_agent,
            AgentKind.DYNAMIC_DEBUG: self.dynamic_agent,
            AgentKind.MAINTENANCE: self.maintenance_agent,
        }[agent_kind]
        result = await agent.run(shadow_task, context)
        return {
            "mode": "replay",
            "score": self.skill_evolution.score_result(agent_kind, report, result),
            "reasons": [
                "shadow-replay-completed",
                result.summary,
            ],
        }

    def _primary_task_for_agent(self, state: WorkflowState, agent_kind: AgentKind) -> Task | None:
        mapping = {
            AgentKind.STATIC_REVIEW: state.get("static_task"),
            AgentKind.DYNAMIC_DEBUG: state.get("dynamic_task"),
            AgentKind.MAINTENANCE: state.get("maintenance_task"),
        }
        return mapping.get(agent_kind)

    async def _shadow_working_root(self, state: WorkflowState, agent_kind: AgentKind) -> Path:
        if agent_kind == AgentKind.MAINTENANCE:
            return await self.file_store.materialize_workspace_copy(self._maintenance_root(state))
        return self._analysis_root(state)

    def _runtime_report_metadata(self, report: WorkflowReport) -> dict[str, object]:
        provider_map = {
            "static_review": self._llm_provider_name(self.static_llm_client),
            "dynamic_debug": self._llm_provider_name(self.dynamic_llm_client),
            "maintenance": self._llm_provider_name(self.maintenance_llm_client),
        }
        model_map = {
            "static_review": self.config.agents.static.model or self.config.llm.model,
            "dynamic_debug": self.config.agents.dynamic.model or self.config.llm.model,
            "maintenance": self.config.agents.maintenance.model or self.config.llm.model,
        }
        provider_values = set(provider_map.values())
        model_values = set(model_map.values())
        metadata = {
            "actual_llm_provider": provider_values.pop() if len(provider_values) == 1 else "mixed",
            "actual_llm_providers": provider_map,
            "actual_llm_model": model_values.pop() if len(model_values) == 1 else "mixed",
            "actual_llm_models": model_map,
            "llm_failure_reason": self._llm_failure_reason(report),
        }
        metadata.update(self._resolution_metadata(report))
        return metadata

    def _llm_provider_name(self, client: object) -> str:
        provider_name = getattr(client, "provider_name", None)
        if isinstance(provider_name, str) and provider_name:
            return provider_name
        return type(client).__name__

    def _llm_failure_reason(self, report: WorkflowReport) -> str:
        results = [
            result
            for result in (
                report.static_result,
                report.dynamic_result,
                report.maintenance_result,
                *report.validation_results.values(),
            )
            if result is not None
        ]
        for result in results:
            failure = str(result.artifacts.get("llm_failure_reason", "")).strip()
            if failure:
                return failure
        return ""

    def _resolution_metadata(self, report: WorkflowReport) -> dict[str, object]:
        resolved: set[str] = set()
        unresolved: set[str] = set()
        regressed: set[str] = set()
        for result in report.validation_results.values():
            comparison = result.artifacts.get("comparison", {})
            resolved.update(str(item) for item in comparison.get("resolved", []))
            unresolved.update(str(item) for item in comparison.get("unresolved", []))
            regressed.update(str(item) for item in comparison.get("regressed", []))

        root_blockers = self._root_blockers(report)
        root_blockers_by_class = self._group_blockers_by_class(root_blockers)
        startup_metadata = self._startup_topology_metadata(report)
        manual_follow_ups = self._manual_follow_ups(
            report,
            advisory_startup_handoffs=startup_metadata["advisory_startup_handoffs"],
        )
        repo_healthy = not root_blockers and not unresolved and not regressed
        reasons: list[str] = []
        root_blocker_classes = sorted(
            {
                str(item.get("root_cause_class"))
                for item in root_blockers
                if item.get("root_cause_class")
            }
        )
        if root_blockers:
            reasons.append("high_or_blocking_findings_remain")
        if unresolved:
            reasons.append("validated_findings_still_unresolved")
        if regressed:
            reasons.append("validation_regressions_detected")
        if report.metadata.get("environment_degraded"):
            reasons.append("execution_environment_degraded")
        explanation = ""
        if report.status == TaskStatus.SUCCEEDED and not repo_healthy:
            explanation = (
                "Workflow execution completed, but the repository is still unhealthy because "
                + ", ".join(reasons)
                + "."
            )
        return {
            "resolution_summary": {
                "resolved": len(resolved),
                "unresolved": len(unresolved),
                "regressed": len(regressed),
                "repo_healthy": repo_healthy,
                "root_blocker_classes": root_blocker_classes,
                "root_blockers_by_class": root_blockers_by_class,
                "workflow_succeeded_but_repo_unhealthy": bool(explanation),
                "explanation": explanation,
            },
            "root_blockers": root_blockers,
            "manual_follow_ups": manual_follow_ups,
            "auto_fixed_blockers": list(
                report.maintenance_result.artifacts.get("auto_fixed_blockers", [])
                if report.maintenance_result is not None
                else []
            ),
            "startup_topology_summary": startup_metadata["startup_topology_summary"],
            "confirmed_startup_blockers": startup_metadata["confirmed_startup_blockers"],
            "advisory_startup_handoffs": startup_metadata["advisory_startup_handoffs"],
            "environment_degraded_reason_summary": (
                "; ".join(str(item) for item in report.metadata.get("install_failures", []))
                if report.metadata.get("environment_degraded")
                else ""
            ),
        }

    def _group_blockers_by_class(self, blockers: list[dict[str, object]]) -> dict[str, int]:
        grouped: dict[str, int] = {}
        for blocker in blockers:
            name = str(blocker.get("root_cause_class") or "unknown")
            grouped[name] = grouped.get(name, 0) + 1
        return grouped

    def _root_blockers(self, report: WorkflowReport) -> list[dict[str, object]]:
        blockers: list[dict[str, object]] = []
        sources = list(report.validation_results.values()) or [
            item for item in (report.static_result, report.dynamic_result) if item is not None
        ]
        for result in sources:
            for finding in result.findings:
                if not self._is_root_blocker_finding(finding):
                    continue
                blockers.append(
                    {
                        "fingerprint": finding.fingerprint,
                        "source_agent": finding.source_agent.value,
                        "severity": finding.severity.value,
                        "rule_id": finding.rule_id,
                        "message": finding.message,
                        "path": finding.path,
                        "line": finding.line,
                        "root_cause_class": finding.root_cause_class,
                        "startup_context": str(finding.evidence.get("startup_context", "")) or None,
                        "entrypoint_path": str(finding.evidence.get("matched_entrypoint", "")) or None,
                        "config_anchor_path": str(finding.evidence.get("matched_config_anchor", "")) or None,
                        "repair_hint": str(finding.evidence.get("repair_hint", "")) or None,
                    }
                )
        if report.maintenance_result is not None:
            for handoff in report.maintenance_result.artifacts.get("unresolved_handoffs", [])[:10]:
                if not isinstance(handoff, dict):
                    continue
                blockers.append(
                    {
                        "fingerprint": "|".join(
                            [
                                "maintenance",
                                str(handoff.get("kind", "")),
                                str(handoff.get("reason", "unresolved-handoff")),
                                str(handoff.get("message", handoff.get("reason", "Unresolved maintenance blocker"))),
                            ]
                        ),
                        "source_agent": AgentKind.MAINTENANCE.value,
                        "severity": Severity.HIGH.value,
                        "rule_id": str(handoff.get("reason", "unresolved-handoff")),
                        "message": str(handoff.get("message", handoff.get("reason", "Unresolved maintenance blocker"))),
                        "path": None,
                        "root_cause_class": str(handoff.get("kind", "")) or None,
                        "guidance": str(handoff.get("guidance", "")) or None,
                    }
                )
        for vulnerability in report.metadata.get("dependency_vulnerabilities", []):
            if not isinstance(vulnerability, dict) or not bool(vulnerability.get("blocker", False)):
                continue
            package_name = str(vulnerability.get("package_name", "")).strip()
            installed_version = str(vulnerability.get("installed_version", "")).strip()
            vulnerability_id = str(vulnerability.get("vulnerability_id", "unknown-vulnerability")).strip()
            fixed_versions = [
                str(item).strip()
                for item in vulnerability.get("fix_versions", []) or []
                if str(item).strip()
            ]
            blockers.append(
                {
                    "fingerprint": "|".join(
                        [
                            "dependency",
                            package_name,
                            installed_version,
                            vulnerability_id,
                        ]
                    ),
                    "source_agent": "dependency_audit",
                    "severity": str(vulnerability.get("severity", "medium")),
                    "rule_id": vulnerability_id,
                    "message": (
                        f"{package_name} {installed_version} is affected by {vulnerability_id}: "
                        f"{str(vulnerability.get('summary', vulnerability_id)).strip()}"
                    ).strip(),
                    "path": package_name or None,
                    "root_cause_class": "dependency",
                    "guidance": (
                        f"Upgrade to one of: {', '.join(fixed_versions)}"
                        if fixed_versions
                        else None
                    ),
                    "package_name": package_name or None,
                    "installed_version": installed_version or None,
                    "fixed_versions": fixed_versions,
                }
            )
        deduped: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in blockers:
            key = str(item.get("fingerprint", "")).strip() or "|".join(
                [
                    str(item.get("source_agent", "")),
                    str(item.get("rule_id", "")),
                    str(item.get("message", "")),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:10]

    def _manual_follow_ups(
        self,
        report: WorkflowReport,
        *,
        advisory_startup_handoffs: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        follow_ups: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        if report.maintenance_result is not None:
            for item in report.maintenance_result.artifacts.get("unresolved_handoffs", [])[:20]:
                if not isinstance(item, dict):
                    continue
                reason = str(item.get("reason", "unresolved-handoff"))
                message = str(item.get("message", reason))
                guidance = str(item.get("guidance", "")).strip()
                key = (str(item.get("kind", "")), reason, message)
                if key in seen:
                    continue
                seen.add(key)
                follow_ups.append(
                    {
                        "kind": str(item.get("kind", "")) or "unknown",
                        "reason": reason,
                        "message": message,
                        "guidance": guidance,
                        "startup_context": str(item.get("startup_context", "")) or None,
                        "entrypoint_path": str(item.get("entrypoint_path", "")) or None,
                        "config_anchor_path": str(item.get("config_anchor_path", "")) or None,
                    }
                )
        for item in advisory_startup_handoffs or []:
            reason = str(item.get("reason", "startup-topology-advisory"))
            message = str(item.get("message", reason))
            key = (str(item.get("kind", "")), reason, message)
            if key in seen:
                continue
            seen.add(key)
            follow_ups.append(item)
        return follow_ups[:10]

    def _is_root_blocker_finding(self, finding: Finding) -> bool:
        if bool(finding.evidence.get("advisory", False)):
            return False
        if finding.severity in {Severity.HIGH, Severity.CRITICAL}:
            return True
        return finding.root_cause_class in {
            "dependency",
            "config",
            "startup",
            "environment",
            "runtime",
            "application",
        }

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
        graph.add_edge("static_review", "dynamic_debug")
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
        graph.add_edge("static_review", "dynamic_debug")
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
        environment = await self._prepare_execution_environment(
            state["run_id"],
            None if self.config.repo_is_remote else self.config.repo_root,
            repo_source=self._configured_repo_identity(),
            repo_ref=self.config.repo_ref,
        )
        active_skills, candidate_skills = await self._prepare_run_skills(self._configured_repo_identity())
        scan_root = (
            Path(environment.base_workspace_root)
            if environment is not None
            else self.config.repo_root
        )
        snapshot, change_set = await self.scan_repository(
            scan_root,
            repo_identity=self._configured_repo_identity(),
        )
        await self.state_store.save_snapshot(state["run_id"], snapshot)
        review_targets = sorted(set(change_set.changed_files + change_set.added_files))
        analysis_root = (
            Path(environment.base_workspace_root)
            if environment is not None
            else self.config.repo_root
        )
        static_context = await self._build_static_context(
            repo_root=analysis_root,
            targets=review_targets,
            execution_environment=environment,
        )
        static_task, dynamic_task = self.dispatcher.create_review_tasks(
            run_id=state["run_id"],
            targets=review_targets,
            commands=self._dynamic_commands(),
            static_context=static_context,
        )
        return {
            "snapshot": snapshot,
            "change_set": change_set,
            "startup_topology": static_context.startup_topology,
            "project_topology": static_context.project_topology,
            "language_profile": static_context.language_profile,
            "static_context": static_context,
            "static_task": static_task,
            "dynamic_task": dynamic_task,
            "execution_environment": environment,
            "active_skills": active_skills,
            "candidate_skills": candidate_skills,
            "report_dir": environment.report_dir if environment is not None else str(self.config.reports_dir / state["run_id"]),
        }

    async def _node_scan_incident_debug(self, state: WorkflowState) -> WorkflowState:
        environment = await self._prepare_execution_environment(
            state["run_id"],
            None if self.config.repo_is_remote else self.config.repo_root,
            repo_source=self._configured_repo_identity(),
            repo_ref=self.config.repo_ref,
        )
        active_skills, candidate_skills = await self._prepare_run_skills(self._configured_repo_identity())
        scan_root = (
            Path(environment.base_workspace_root)
            if environment is not None
            else self.config.repo_root
        )
        snapshot, change_set = await self.scan_repository(
            scan_root,
            repo_identity=self._configured_repo_identity(),
        )
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
        environment = await self._prepare_execution_environment(
            state["run_id"],
            source_root,
            repo_source=str(source_root),
        )
        active_skills, candidate_skills = await self._prepare_run_skills(source_root)
        snapshot, change_set = await self.scan_repository(
            source_root,
            state["pr_context"],
            repo_identity=str(source_root),
        )
        await self.state_store.save_snapshot(state["run_id"], snapshot)
        review_targets = sorted(set(change_set.changed_files + change_set.added_files))
        analysis_root = (
            Path(environment.base_workspace_root)
            if environment is not None
            else source_root
        )
        static_context = await self._build_static_context(
            repo_root=analysis_root,
            targets=review_targets,
            execution_environment=environment,
        )
        static_task, dynamic_task = self.dispatcher.create_review_tasks(
            run_id=state["run_id"],
            targets=review_targets,
            commands=self._dynamic_commands(),
            static_context=static_context,
        )
        return {
            "snapshot": snapshot,
            "change_set": change_set,
            "startup_topology": static_context.startup_topology,
            "project_topology": static_context.project_topology,
            "language_profile": static_context.language_profile,
            "static_context": static_context,
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
            repo_identity=self._repo_identity(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
            startup_topology=state.get("startup_topology"),
            project_topology=state.get("project_topology"),
            language_profile=state.get("language_profile"),
            static_context=state.get("static_context"),
        )
        result = await self.execute_task(task, context)
        if task.task_type == TaskType.VALIDATION_STATIC:
            return {"validation_static_result": result}
        return {
            "static_result": result,
            "startup_topology": result.artifacts.get("startup_topology") or state.get("startup_topology"),
            "project_topology": result.artifacts.get("project_topology") or state.get("project_topology"),
            "language_profile": result.artifacts.get("language_profile") or state.get("language_profile"),
        }

    async def _node_dynamic_debug(self, state: WorkflowState) -> WorkflowState:
        task = state["validation_dynamic_task"] if "validation_dynamic_task" in state else state["dynamic_task"]
        root = self._validation_root(state) if task.task_type == TaskType.VALIDATION_DYNAMIC else self._analysis_root(state)
        active_skill, candidate_skill = self._skill_for_agent(state, AgentKind.DYNAMIC_DEBUG)
        context = self._build_context(
            state["run_id"],
            root,
            repo_root=self._source_repo_root(state),
            repo_identity=self._repo_identity(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
            startup_topology=state.get("startup_topology"),
            project_topology=state.get("project_topology"),
            language_profile=state.get("language_profile"),
            static_context=state.get("static_context"),
        )
        result = await self.execute_task(task, context)
        if task.task_type == TaskType.VALIDATION_DYNAMIC:
            return {"validation_dynamic_result": result}
        return {"dynamic_result": result}

    async def _node_feedback_merge(self, state: WorkflowState) -> WorkflowState:
        initial_findings = [*state["static_result"].findings, *state["dynamic_result"].findings]
        await self.issue_catalog.record_findings(
            self._repo_identity(state),
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
            startup_topology=state.get("startup_topology"),
        )
        return {
            "feedback": feedback,
            "maintenance_task": maintenance_task,
            "candidate_skills": await self._refresh_candidate_skills(
                state,
                self._repo_identity(state),
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
            repo_identity=self._repo_identity(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
            startup_topology=state.get("startup_topology"),
            project_topology=state.get("project_topology"),
            language_profile=state.get("language_profile"),
            static_context=state.get("static_context"),
        )
        result = await self.execute_task(state["maintenance_task"], context)
        updates: WorkflowState = {
            "maintenance_result": result,
            "candidate_skills": await self._refresh_candidate_skills(
                state,
                self._repo_identity(state),
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
            validation_snapshot.repo_root = self._repo_identity(state)
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

    async def _maybe_followup_maintenance_round(self, state: WorkflowState) -> None:
        if state.get("artifacts", {}).get("repair_rounds", 1) >= 2:
            return
        followup_handoffs = self._followup_handoffs_from_validation(state)
        if not followup_handoffs:
            return

        feedback = FeedbackBundle(
            snapshot=state.get("validation_snapshot", state["snapshot"]),
            change_set=state["change_set"],
            static_findings=state["validation_static_result"].findings,
            dynamic_findings=state["validation_dynamic_result"].findings,
        )
        maintenance_task = self.dispatcher.create_maintenance_task(
            state["run_id"],
            feedback,
            handoffs=followup_handoffs,
            startup_topology=state.get("startup_topology"),
        )
        maintenance_root = self._maintenance_root(state)
        active_skill, candidate_skill = self._skill_for_agent(state, AgentKind.MAINTENANCE)
        context = self._build_context(
            state["run_id"],
            maintenance_root,
            repo_root=self._source_repo_root(state),
            repo_identity=self._repo_identity(state),
            execution_environment=state.get("execution_environment"),
            active_skill=active_skill,
            candidate_skill=candidate_skill,
            startup_topology=state.get("startup_topology"),
            project_topology=state.get("project_topology"),
            language_profile=state.get("language_profile"),
            static_context=state.get("static_context"),
        )
        followup_result = await self.execute_task(maintenance_task, context)
        merged_result = self._merge_maintenance_results(
            state["maintenance_result"],
            followup_result,
        )
        state["maintenance_task"] = maintenance_task
        state["maintenance_result"] = merged_result
        state["artifacts"] = {
            **dict(state.get("artifacts", {})),
            "repair_rounds": 2,
            "followup_handoff_count": len(followup_handoffs),
        }

        if followup_result.patch is None or not followup_result.patch.file_patches:
            return

        await self.patch_service.apply(maintenance_root, followup_result.patch)
        if state.get("execution_environment") is not None:
            validation_workspace_root = await self.environment_manager.refresh_validation_workspace(
                state["execution_environment"]
            )
        else:
            validation_workspace_root = Path(state["validation_workspace_root"])
            await self.patch_service.apply(validation_workspace_root, followup_result.patch)
        state["validation_workspace_root"] = str(validation_workspace_root)
        state["validation_snapshot"] = await self.scanner.scan(validation_workspace_root)
        state["validation_snapshot"].repo_root = self._repo_identity(state)
        static_task, dynamic_task = self.dispatcher.create_validation_tasks(
            run_id=state["run_id"],
            patch=merged_result.patch or followup_result.patch,
            commands=self._dynamic_commands(),
        )
        state["validation_static_task"] = static_task
        state["validation_dynamic_task"] = dynamic_task
        validation_static = await self._node_static_review(state)
        validation_dynamic = await self._node_dynamic_debug(state)
        state["validation_static_result"] = validation_static["validation_static_result"]
        state["validation_dynamic_result"] = validation_dynamic["validation_dynamic_result"]

    def _followup_handoffs_from_validation(self, state: WorkflowState) -> list[dict[str, object]]:
        handoffs: list[dict[str, object]] = []
        for result in (
            state.get("validation_dynamic_result"),
            state.get("validation_static_result"),
        ):
            if result is None:
                continue
            for finding in result.findings:
                if not self._is_followup_finding(finding):
                    continue
                handoffs.append(
                    {
                        "source_agent": finding.source_agent.value,
                        "title": f"Follow up {finding.root_cause_class or finding.category} blocker",
                        "description": finding.message,
                        "recommended_change": self._followup_recommended_change(finding),
                        "severity": finding.severity.value,
                        "kind": (
                            finding.root_cause_class
                            if finding.root_cause_class in {"dependency", "config", "startup", "runtime"}
                            else "code"
                        ),
                        "confidence": 0.9 if finding.root_cause_class in {"dependency", "config", "startup"} else 0.75,
                        "affected_files": [finding.path] if finding.path else [],
                        "metadata": {
                            "rule_id": finding.rule_id,
                            "category": finding.category,
                            "root_cause_class": finding.root_cause_class,
                        },
                        "evidence": [
                            {
                                "kind": "validation-finding",
                                "title": finding.rule_id,
                                "summary": finding.message,
                                "path": finding.path,
                                "data": finding.evidence,
                            }
                        ],
                    }
                )
        unresolved_handoffs = state["maintenance_result"].artifacts.get("unresolved_handoffs", [])
        for item in unresolved_handoffs:
            if not isinstance(item, dict):
                continue
            handoffs.append(
                {
                    "source_agent": AgentKind.MAINTENANCE.value,
                    "title": f"Follow up {item.get('kind', 'maintenance')} blocker",
                    "description": str(item.get("message", item.get("reason", "Unresolved maintenance blocker"))),
                    "recommended_change": "Apply a targeted manifest or configuration fix for the unresolved blocker.",
                    "severity": Severity.HIGH.value,
                    "kind": str(item.get("kind", "code")),
                    "confidence": 0.8,
                    "affected_files": [],
                    "metadata": {
                        "rule_id": str(item.get("reason", "unresolved-handoff")),
                        "root_cause_class": str(item.get("kind", "")) or None,
                    },
                    "evidence": [],
                }
            )

        deduped: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in handoffs:
            key = (
                str(item.get("kind", "")),
                str(item.get("title", "")),
                str(item.get("description", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:6]

    def _is_followup_finding(self, finding: Finding) -> bool:
        if bool(finding.evidence.get("advisory", False)):
            return False
        if finding.root_cause_class in {"dependency", "config", "startup", "environment"}:
            return True
        return finding.severity in {Severity.HIGH, Severity.CRITICAL}

    def _startup_topology_metadata(self, report: WorkflowReport) -> dict[str, object]:
        topology_value = None
        if report.static_result is not None:
            topology_value = report.static_result.artifacts.get("startup_topology")
        topology = to_jsonable(topology_value) if topology_value is not None else {}
        entrypoints = [
            item for item in topology.get("entrypoints", [])
            if isinstance(item, dict)
        ] if isinstance(topology, dict) else []
        anchors = [
            item for item in topology.get("config_anchors", [])
            if isinstance(item, dict)
        ] if isinstance(topology, dict) else []
        confirmed = self._confirmed_startup_blockers(report)
        advisory = self._advisory_startup_handoffs(report, confirmed)
        contexts = sorted(
            {
                str(item.get("context"))
                for item in [*entrypoints, *anchors]
                if item.get("context")
            }
        )
        return {
            "startup_topology_summary": {
                "src_layout": bool(topology.get("src_layout", False)) if isinstance(topology, dict) else False,
                "entrypoint_count": len(entrypoints),
                "config_anchor_count": len(anchors),
                "contexts": contexts,
                "entrypoints": entrypoints,
                "config_anchors": anchors,
                "repair_hints": list(topology.get("repair_hints", [])) if isinstance(topology, dict) else [],
                "confirmed_count": len(confirmed),
                "advisory_count": len(advisory),
            },
            "confirmed_startup_blockers": confirmed,
            "advisory_startup_handoffs": advisory,
        }

    def _confirmed_startup_blockers(self, report: WorkflowReport) -> list[dict[str, object]]:
        candidates: list[tuple[str, AgentResult]] = []
        if report.dynamic_result is not None:
            candidates.append(("analysis", report.dynamic_result))
        validation_dynamic = report.validation_results.get("dynamic_validation")
        if validation_dynamic is not None:
            candidates.append(("validation", validation_dynamic))

        confirmed: list[dict[str, object]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for phase, result in candidates:
            for finding in result.findings:
                if finding.root_cause_class != "startup":
                    continue
                entrypoint_path = str(finding.evidence.get("matched_entrypoint", "")).strip()
                config_anchor_path = str(finding.evidence.get("matched_config_anchor", "")).strip()
                startup_context = str(finding.evidence.get("startup_context", "")).strip()
                if not entrypoint_path and not config_anchor_path and not startup_context:
                    continue
                key = (
                    startup_context,
                    entrypoint_path,
                    config_anchor_path,
                    str(finding.evidence.get("repair_hint", "")).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                confirmed.append(
                    {
                        "phase": phase,
                        "source_agent": finding.source_agent.value,
                        "severity": finding.severity.value,
                        "rule_id": finding.rule_id,
                        "message": finding.message,
                        "startup_context": startup_context or None,
                        "entrypoint_path": entrypoint_path or None,
                        "config_anchor_path": config_anchor_path or None,
                        "repair_hint": str(finding.evidence.get("repair_hint", "")).strip() or None,
                    }
                )
        return confirmed

    def _advisory_startup_handoffs(
        self,
        report: WorkflowReport,
        confirmed_startup_blockers: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if report.static_result is None:
            return []
        advisory_items = report.static_result.artifacts.get("startup_handoffs", [])
        if not isinstance(advisory_items, list):
            return []
        confirmed_keys = {
            (
                str(item.get("startup_context", "") or ""),
                str(item.get("entrypoint_path", "") or ""),
                str(item.get("config_anchor_path", "") or ""),
                str(item.get("repair_hint", "") or ""),
            )
            for item in confirmed_startup_blockers
        }
        advisory: list[dict[str, object]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for handoff in advisory_items:
            if not isinstance(handoff, dict):
                continue
            metadata = handoff.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            key = (
                str(metadata.get("startup_context", "") or ""),
                str(metadata.get("entrypoint_path", "") or ""),
                str(metadata.get("config_anchor_path", "") or ""),
                str(metadata.get("repair_hint", "") or ""),
            )
            if key in confirmed_keys or key in seen:
                continue
            seen.add(key)
            advisory.append(
                {
                    "kind": "startup",
                    "reason": "startup-topology-advisory",
                    "message": str(handoff.get("description", handoff.get("title", "Startup topology advisory"))),
                    "guidance": str(handoff.get("recommended_change", "")).strip(),
                    "startup_context": key[0] or None,
                    "entrypoint_path": key[1] or None,
                    "config_anchor_path": key[2] or None,
                    "repair_hint": key[3] or None,
                }
            )
        return advisory

    def _followup_recommended_change(self, finding: Finding) -> str:
        if finding.root_cause_class == "dependency":
            return "Apply a dependency declaration fix in the project manifest or requirements file, then re-run validation."
        if finding.root_cause_class == "config":
            return "Apply a configuration or environment placeholder fix and re-run validation."
        if finding.root_cause_class == "startup":
            return "Apply a startup/bootstrap fix for the Python test entrypoint or path configuration, then re-run validation."
        if finding.root_cause_class == "environment":
            return "Stabilize the validation environment or bootstrap requirements before rerunning the check."
        return "Address the blocking validation finding with a focused follow-up patch."

    def _merge_maintenance_results(
        self,
        current: AgentResult,
        followup: AgentResult,
    ) -> AgentResult:
        merged_patch = self._merge_patch_proposals(current.patch, followup.patch)
        current_unresolved = list(current.artifacts.get("unresolved_handoffs", []))
        followup_unresolved = list(followup.artifacts.get("unresolved_handoffs", []))
        artifacts = {
            **current.artifacts,
            **followup.artifacts,
            "repair_rounds": 2,
            "unresolved_handoffs": followup_unresolved or current_unresolved,
            "auto_fixed_blockers": sorted(
                {
                    *current.artifacts.get("auto_fixed_blockers", []),
                    *followup.artifacts.get("auto_fixed_blockers", []),
                }
            ),
        }
        return AgentResult(
            task_id=followup.task_id,
            agent_kind=followup.agent_kind,
            task_type=followup.task_type,
            status=followup.status if followup.status == TaskStatus.FAILED else current.status,
            summary=followup.summary or current.summary,
            findings=followup.findings or current.findings,
            patch=merged_patch,
            artifacts=artifacts,
            errors=[*current.errors, *followup.errors],
        )

    def _merge_patch_proposals(
        self,
        current: PatchProposal | None,
        followup: PatchProposal | None,
    ) -> PatchProposal | None:
        if current is None:
            return followup
        if followup is None:
            return current
        merged_by_path: dict[str, FilePatch] = {item.path: item for item in current.file_patches}
        for patch in followup.file_patches:
            if patch.path in merged_by_path:
                original = merged_by_path[patch.path]
                merged_by_path[patch.path] = FilePatch(
                    path=patch.path,
                    old_content=original.old_content,
                    new_content=patch.new_content,
                    diff=self.patch_service.build_file_patch(
                        patch.path,
                        original.old_content,
                        patch.new_content,
                    ).diff,
                )
            else:
                merged_by_path[patch.path] = patch
        merged_file_patches = [merged_by_path[path] for path in sorted(merged_by_path)]
        metadata = {
            **current.metadata,
            **followup.metadata,
            "repair_scope": sorted(
                {
                    *current.metadata.get("repair_scope", []),
                    *followup.metadata.get("repair_scope", []),
                }
            ),
            "unresolved_handoffs": list(
                followup.metadata.get("unresolved_handoffs", current.metadata.get("unresolved_handoffs", []))
            ),
            "auto_fixed_blockers": sorted(
                {
                    *current.metadata.get("auto_fixed_blockers", []),
                    *followup.metadata.get("auto_fixed_blockers", []),
                }
            ),
        }
        return PatchProposal(
            summary=followup.summary or current.summary,
            rationale="\n".join(item for item in [current.rationale, followup.rationale] if item).strip(),
            file_patches=merged_file_patches,
            validation_targets=sorted(
                {
                    *current.validation_targets,
                    *followup.validation_targets,
                }
            ),
            suggestions=list(dict.fromkeys([*current.suggestions, *followup.suggestions])),
            metadata=metadata,
            applied=current.applied or followup.applied,
            diff_text=self.patch_service.render_patch(merged_file_patches),
        )

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
                    self._repo_identity(state),
                    state["run_id"],
                    static_original,
                    state["validation_static_result"].findings,
                )
            if "comparison" not in state["validation_dynamic_result"].artifacts:
                state["validation_dynamic_result"].artifacts["comparison"] = await self.issue_catalog.reconcile(
                    self._repo_identity(state),
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
        await self._maybe_followup_maintenance_round(state)
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
                self._repo_identity(state),
                state["run_id"],
                static_original,
                state["validation_static_result"].findings,
            )
            dynamic_comparison = await self.issue_catalog.reconcile(
                self._repo_identity(state),
                state["run_id"],
                dynamic_original,
                state["validation_dynamic_result"].findings,
            )
            state["validation_static_result"].artifacts["comparison"] = static_comparison
            state["validation_dynamic_result"].artifacts["comparison"] = dynamic_comparison

        report = WorkflowReport(
            run_id=state["run_id"],
            workflow_name="maintenance_loop",
            repo_root=self._repo_identity(state),
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
        report.metadata.update(
            await self._dependency_audit_metadata(
                analysis_root=self._analysis_root(state),
                execution_environment=state.get("execution_environment"),
                static_context=state.get("static_context"),
            )
        )
        await self._enrich_report_for_persistence(
            report,
            analysis_root=self._analysis_root(state),
            maintenance_root=self._maintenance_root(state),
            validation_root=self._validation_root(state),
        )
        report.metadata.update(self._runtime_report_metadata(report))
        report.metadata.update(self._static_context_metadata(state.get("static_context")))
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
            self._repo_identity(state),
            state["run_id"],
            state["dynamic_result"].findings,
        )
        report = WorkflowReport(
            run_id=state["run_id"],
            workflow_name="incident_debug",
            repo_root=self._repo_identity(state),
            started_at=state["started_at"],
            finished_at=datetime.now(timezone.utc),
            status=state["dynamic_result"].status,
            snapshot=state["snapshot"],
            change_set=state["change_set"],
            dynamic_result=state["dynamic_result"],
            report_dir=str(state.get("report_dir", "")),
            metadata=self._environment_metadata(state.get("execution_environment")),
        )
        await self._enrich_report_for_persistence(
            report,
            analysis_root=self._analysis_root(state),
        )
        report.metadata.update(self._runtime_report_metadata(report))
        report.metadata.update(self._static_context_metadata(state.get("static_context")))
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
        await self._maybe_followup_maintenance_round(state)
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
                self._repo_identity(state),
                state["run_id"],
                static_original,
                state["validation_static_result"].findings,
            )
            dynamic_comparison = await self.issue_catalog.reconcile(
                self._repo_identity(state),
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
        report.metadata.update(
            await self._dependency_audit_metadata(
                analysis_root=self._analysis_root(state),
                execution_environment=state.get("execution_environment"),
                static_context=state.get("static_context"),
            )
        )
        await self._enrich_report_for_persistence(
            report,
            analysis_root=self._analysis_root(state),
            maintenance_root=self._maintenance_root(state),
            validation_root=self._validation_root(state),
        )
        report.metadata.update(self._runtime_report_metadata(report))
        report.metadata.update(self._static_context_metadata(state.get("static_context")))
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
            repo_root=self._repo_identity(state),
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
            ArtifactReference(name="dependency_vulnerabilities_json", path="artifacts/dependency_vulnerabilities.json"),
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
        await self.file_store.ensure_dir(report_dir / "artifacts")
        await write_markdown(report_dir / "summary.md", report)
        await write_json(report_dir / "report.json", report)
        await write_json(report_dir / "findings.json", report.all_findings)
        if "dependency_audit_status" in report.metadata:
            await write_json(
                report_dir / "artifacts" / "dependency_vulnerabilities.json",
                {
                    "status": report.metadata.get("dependency_audit_status", "not-run"),
                    "error": report.metadata.get("dependency_audit_error", ""),
                    "summary": report.metadata.get("dependency_vulnerability_summary", {}),
                    "vulnerabilities": report.metadata.get("dependency_vulnerabilities", []),
                },
            )
        if report.maintenance_result and report.maintenance_result.patch:
            await self.file_store.write_text(
                report_dir / "patch.diff",
                report.maintenance_result.patch.diff_text,
            )
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
        help="Target repository root or remote Git URL. Defaults to the config value.",
    )
    common_parser.add_argument(
        "--repo-ref",
        default=None,
        help="Optional branch, tag, or commit SHA when --repo points to a remote Git repository.",
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
        repo_override=args.repo,
        repo_ref_override=args.repo_ref,
    )
    orchestrator = await Orchestrator.create(config)
    try:
        if args.command == "run-once":
            report = await orchestrator.run_workflow("maintenance_loop")
            _print_report(report)
            return 0
        if args.command == "scan":
            if config.repo_is_remote:
                environment = await orchestrator._prepare_execution_environment(
                    f"scan-{uuid4().hex}",
                    None,
                    repo_source=config.repo_source,
                    repo_ref=config.repo_ref,
                )
                snapshot, change_set = await orchestrator.scan_repository(
                    Path(environment.base_workspace_root),
                    repo_identity=config.repo_source,
                )
            else:
                snapshot, change_set = await orchestrator.scan_repository(
                    repo_identity=config.repo_source,
                )
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
            rows = await orchestrator.skill_manager.skill_status(config.repo_source)
            if args.agent:
                rows = [row for row in rows if row["agent"] == args.agent]
            for row in rows:
                print(
                    f"{row['agent']}: active={row['active_version']} source={row['active_source']} "
                    f"candidate={row['candidate_version'] or '-'} shadow_runs={row['candidate_shadow_runs']} "
                    f"status={row['candidate_status'] or '-'} cooldown_until={row['candidate_cooldown_until'] or '-'} "
                    f"frozen={row['frozen']}"
                )
            return 0
        if args.command == "skill-history":
            history = await orchestrator.skill_manager.history(config.repo_source, AgentKind(args.agent))
            for item in history:
                print(
                    f"{item['created_at']} run={item['run_id']} active={item['active_version']} "
                    f"candidate={item['candidate_version'] or '-'} active_score={item['active_score']} "
                    f"candidate_score={item['candidate_score']} mode={item['mode']} promoted={item['promoted']} "
                    f"reasons={','.join(item['reasons']) if item['reasons'] else '-'}"
                )
            return 0
        if args.command == "skill-freeze":
            await orchestrator.skill_manager.freeze(
                config.repo_source,
                AgentKind(args.agent),
                frozen=not bool(args.unfreeze),
            )
            print(f"{args.agent}: {'unfrozen' if args.unfreeze else 'frozen'}")
            return 0
        if args.command == "skill-promote":
            promoted = await orchestrator.skill_manager.manual_promote(
                config.repo_source,
                AgentKind(args.agent),
            )
            print(f"{args.agent}: {'promoted' if promoted else 'no-open-candidate'}")
            return 0
        return 1
    finally:
        await orchestrator.close()


def main(argv: Iterable[str] | None = None) -> int:
    return asyncio.run(main_async(argv))
