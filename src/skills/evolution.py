from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from uuid import uuid4

from core.config import AppConfig
from core.models import (
    AgentKind,
    AgentReflection,
    AgentResult,
    CompletionReason,
    Finding,
    Severity,
    SkillCandidate,
    SkillCandidateStatus,
    SkillEvaluation,
    SkillPack,
    SkillSource,
    WorkflowReport,
)
from memory.state_store import StateStore
from reports.serializer import to_jsonable


class SkillEvolutionService:
    def __init__(self, config: AppConfig, state_store: StateStore) -> None:
        self.config = config
        self.state_store = state_store

    async def reflect_and_seed_candidate(
        self,
        *,
        repo_root: str,
        run_id: str,
        task_id: str,
        session_id: str,
        result: AgentResult,
        active_skill: SkillPack | None,
    ) -> tuple[AgentReflection | None, SkillCandidate | None]:
        if not self.config.skills.enabled or active_skill is None:
            return None, None

        reflection = AgentReflection(
            reflection_id=uuid4().hex,
            repo_root=repo_root,
            run_id=run_id,
            task_id=task_id,
            session_id=session_id,
            agent_kind=result.agent_kind,
            skill_version=active_skill.version,
            summary=result.summary,
            metrics=self._result_metrics(result),
            upgrade_hints=self._upgrade_hints(result),
        )
        await self.state_store.save_agent_reflection(reflection)

        auto_upgrade = self._auto_upgrade_enabled(result.agent_kind)
        if not auto_upgrade:
            return reflection, None

        existing_candidate = await self.state_store.get_open_skill_candidate(repo_root, result.agent_kind)
        if existing_candidate is not None:
            return reflection, existing_candidate

        if not reflection.upgrade_hints:
            return reflection, None

        candidate_pack = self._build_candidate_pack(active_skill, reflection.upgrade_hints)
        await self.state_store.upsert_skill_pack(repo_root, candidate_pack)
        candidate = SkillCandidate(
            candidate_id=uuid4().hex,
            repo_root=repo_root,
            agent_kind=result.agent_kind,
            based_on_version=active_skill.version,
            version=candidate_pack.version,
            skill_pack=candidate_pack,
            status=SkillCandidateStatus.CANDIDATE,
            notes=list(reflection.upgrade_hints),
        )
        await self.state_store.save_skill_candidate(candidate)
        return reflection, candidate

    async def evaluate_report(
        self,
        *,
        repo_root: str,
        report: WorkflowReport,
        active_skills: dict[str, SkillPack],
        candidate_skills: dict[str, SkillCandidate],
    ) -> dict[str, object]:
        if not self.config.skills.enabled:
            return {
                "agent_skills": {},
                "active_skill_versions": {},
                "candidate_skill_versions": {},
                "skill_upgrade_events": [],
                "shadow_evaluation_summary": {},
            }

        upgrade_events: list[dict[str, object]] = []
        shadow_summary: dict[str, object] = {}
        active_versions = {name: pack.version for name, pack in active_skills.items()}
        candidate_versions = {
            name: candidate.version for name, candidate in candidate_skills.items()
        }
        agent_skills = {
            name: {
                "active_version": pack.version,
                "active_source": pack.source.value,
                "candidate_version": candidate_versions.get(name),
            }
            for name, pack in active_skills.items()
        }

        for agent_kind, result in self._primary_results(report).items():
            active_skill = active_skills.get(agent_kind.value)
            candidate = candidate_skills.get(agent_kind.value)
            if active_skill is None:
                continue
            active_score = self._score(agent_kind, report, result)
            evaluation_payload: dict[str, object] = {
                "active_score": active_score,
                "candidate_score": None,
                "promoted": False,
                "reasons": [],
            }
            if candidate is not None and self.config.skills.shadow_evaluation_enabled:
                binding = await self.state_store.get_skill_binding(repo_root, agent_kind)
                frozen = bool(binding.frozen) if binding is not None else False
                candidate_score, reasons = self._shadow_candidate_score(
                    agent_kind,
                    report,
                    result,
                    candidate,
                    active_score,
                )
                promoted = False
                next_shadow_runs = candidate.shadow_runs + 1
                await self.state_store.save_skill_evaluation(
                    SkillEvaluation(
                        evaluation_id=uuid4().hex,
                        repo_root=repo_root,
                        agent_kind=agent_kind,
                        run_id=report.run_id,
                        active_version=active_skill.version,
                        candidate_version=candidate.version,
                        active_score=active_score,
                        candidate_score=candidate_score,
                        promoted=False,
                        reasons=reasons,
                    )
                )
                if frozen:
                    reasons.append("binding-frozen")
                if not frozen and next_shadow_runs >= self.config.skills.min_shadow_runs:
                    margin_score = active_score * (1.0 + self.config.skills.promotion_margin)
                    if candidate_score >= margin_score and not self._has_high_regression(report, agent_kind):
                        await self.state_store.set_skill_binding(
                            self._promoted_binding(repo_root, agent_kind, candidate.version)
                        )
                        await self.state_store.update_skill_candidate(
                            candidate.candidate_id,
                            status=SkillCandidateStatus.PROMOTED,
                            shadow_runs=next_shadow_runs,
                            notes=[*candidate.notes, "Promoted after shadow evaluation."],
                        )
                        promoted = True
                        upgrade_events.append(
                            {
                                "agent": agent_kind.value,
                                "event": "promoted",
                                "candidate_version": candidate.version,
                                "active_version": active_skill.version,
                                "candidate_score": candidate_score,
                                "active_score": active_score,
                            }
                        )
                    else:
                        await self.state_store.update_skill_candidate(
                            candidate.candidate_id,
                            status=SkillCandidateStatus.REJECTED,
                            shadow_runs=next_shadow_runs,
                            notes=[*candidate.notes, "Rejected after shadow evaluation."],
                        )
                        upgrade_events.append(
                            {
                                "agent": agent_kind.value,
                                "event": "rejected",
                                "candidate_version": candidate.version,
                                "candidate_score": candidate_score,
                                "active_score": active_score,
                                "reasons": reasons,
                            }
                        )
                else:
                    await self.state_store.update_skill_candidate(
                        candidate.candidate_id,
                        shadow_runs=next_shadow_runs,
                        notes=candidate.notes,
                    )
                evaluation_payload = {
                    "active_score": active_score,
                    "candidate_score": candidate_score,
                    "promoted": promoted,
                    "reasons": reasons,
                    "shadow_runs": next_shadow_runs,
                }
            shadow_summary[agent_kind.value] = evaluation_payload

        return {
            "agent_skills": agent_skills,
            "active_skill_versions": active_versions,
            "candidate_skill_versions": candidate_versions,
            "skill_upgrade_events": upgrade_events,
            "shadow_evaluation_summary": shadow_summary,
        }

    def _primary_results(self, report: WorkflowReport) -> dict[AgentKind, AgentResult]:
        results: dict[AgentKind, AgentResult] = {}
        if report.static_result is not None:
            results[AgentKind.STATIC_REVIEW] = report.static_result
        if report.dynamic_result is not None:
            results[AgentKind.DYNAMIC_DEBUG] = report.dynamic_result
        if report.maintenance_result is not None:
            results[AgentKind.MAINTENANCE] = report.maintenance_result
        return results

    def _result_metrics(self, result: AgentResult) -> dict[str, object]:
        session_summary = dict(result.artifacts.get("session_summary", {}))
        high_value_findings = int(result.artifacts.get("high_value_findings", 0) or 0)
        patch_files = len(result.patch.file_patches) if result.patch else 0
        return {
            "status": result.status.value,
            "finding_count": len(result.findings),
            "handoff_count": len(result.artifacts.get("handoffs", [])),
            "tool_call_count": int(session_summary.get("tool_call_count", 0) or 0),
            "step_count": int(session_summary.get("step_count", 0) or 0),
            "budget_exhausted": bool(session_summary.get("budget_exhausted", False)),
            "high_value_findings": high_value_findings,
            "patch_files": patch_files,
        }

    def _upgrade_hints(self, result: AgentResult) -> list[str]:
        hints: list[str] = []
        session_summary = dict(result.artifacts.get("session_summary", {}))
        handoff_count = len(result.artifacts.get("handoffs", []))
        if result.agent_kind == AgentKind.STATIC_REVIEW:
            low_value = [
                finding for finding in result.findings if finding.severity == Severity.LOW
            ]
            high_value = int(result.artifacts.get("high_value_findings", 0) or 0)
            if len(result.findings) >= 10 and len(low_value) / max(len(result.findings), 1) > 0.6:
                hints.append("suppress_low_value_noise")
            if high_value == 0 and "correctness" in result.summary.lower():
                hints.append("promote_summary_only_risks_to_findings")
            if high_value > 0 and handoff_count == 0:
                hints.append("emit_fix_requests_for_high_value_findings")
            if int(session_summary.get("tool_call_count", 0) or 0) <= 1:
                hints.append("expand_cross_file_investigation")
        elif result.agent_kind == AgentKind.DYNAMIC_DEBUG:
            if any(f.rule_id == "command-failed" for f in result.findings) and handoff_count == 0:
                hints.append("always_emit_runtime_fix_requests")
            if int(session_summary.get("tool_call_count", 0) or 0) <= 1:
                hints.append("inspect_traceback_before_finalize")
            if "module not found" in result.summary.lower() or "dependency" in result.summary.lower():
                hints.append("prioritize_dependency_contract_diagnosis")
        else:
            if result.patch is None or not result.patch.file_patches:
                hints.append("expand_safe_fix_coverage")
            unsupported = result.artifacts.get("unsupported_findings", [])
            if unsupported:
                hints.append("target_high_value_unresolved_handoffs")
            if result.patch is not None and len(result.patch.file_patches) > 0 and not result.artifacts.get("publishable", False):
                hints.append("improve_publishability_filters")
        return sorted(dict.fromkeys(hints))

    def _build_candidate_pack(self, active_skill: SkillPack, hints: list[str]) -> SkillPack:
        policy = replace(active_skill.policy)
        planning = list(policy.planning_heuristics)
        completion = list(policy.completion_checklist)
        noise = list(policy.noise_suppression)
        tool_preferences = list(policy.tool_preferences)

        for hint in hints:
            if hint == "suppress_low_value_noise":
                noise.append("missing-module-docstring")
                noise.append("trailing-whitespace")
                policy.rule_weights["missing-module-docstring"] = -0.5
            elif hint == "promote_summary_only_risks_to_findings":
                completion.append("Promote any higher-value summary-only concern into a structured finding.")
            elif hint == "emit_fix_requests_for_high_value_findings":
                completion.append("Emit at least one fix request for each medium/high correctness issue cluster.")
            elif hint == "expand_cross_file_investigation":
                planning.append("After the first deterministic pass, inspect at least one neighboring module or dependency edge.")
                tool_preferences.extend(["search_repo", "read_file", "ast_summary"])
            elif hint == "always_emit_runtime_fix_requests":
                completion.append("Runtime blockers must always become fix requests.")
            elif hint == "inspect_traceback_before_finalize":
                tool_preferences = ["run_test_command", "parse_traceback", *tool_preferences]
            elif hint == "prioritize_dependency_contract_diagnosis":
                planning.append("When collection/import fails, inspect dependency declarations before finalizing.")
            elif hint == "expand_safe_fix_coverage":
                planning.append("Investigate dependency declarations and config changes before limiting to cosmetic fixes.")
            elif hint == "target_high_value_unresolved_handoffs":
                completion.append("Prioritize unresolved high-severity handoffs over low-risk cosmetic fixes.")
            elif hint == "improve_publishability_filters":
                planning.append("Prefer smaller publishable patches over broader low-risk batches.")

        policy.planning_heuristics = sorted(dict.fromkeys(planning))
        policy.completion_checklist = sorted(dict.fromkeys(completion))
        policy.noise_suppression = sorted(dict.fromkeys(noise))
        policy.tool_preferences = [tool for i, tool in enumerate(tool_preferences) if tool not in tool_preferences[:i]]

        candidate = SkillPack(
            agent_kind=active_skill.agent_kind,
            name=f"{active_skill.name}-candidate",
            version=f"{active_skill.version}-cand-{uuid4().hex[:8]}",
            description=f"Candidate derived from {active_skill.version}",
            status="candidate",
            source=SkillSource.DATABASE,
            system_prompt=active_skill.system_prompt,
            skill_markdown=active_skill.skill_markdown,
            examples=[dict(item) for item in active_skill.examples],
            policy=policy,
        )
        candidate.profile_hash = hashlib.sha256(
            json.dumps(
                {
                    "agent_kind": candidate.agent_kind.value,
                    "name": candidate.name,
                    "version": candidate.version,
                    "description": candidate.description,
                    "status": candidate.status,
                    "source": candidate.source.value,
                    "system_prompt": candidate.system_prompt,
                    "skill_markdown": candidate.skill_markdown,
                    "examples": candidate.examples,
                    "policy": to_jsonable(candidate.policy),
                },
                sort_keys=True,
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        return candidate

    def _score(self, agent_kind: AgentKind, report: WorkflowReport, result: AgentResult) -> float:
        if agent_kind == AgentKind.STATIC_REVIEW:
            high_value = int(result.artifacts.get("high_value_findings", 0) or 0)
            low_value = len([finding for finding in result.findings if finding.severity == Severity.LOW])
            handoffs = len(result.artifacts.get("handoffs", []))
            regressed = len(
                report.validation_results.get("static_validation", AgentResult("", agent_kind, result.task_type, result.status, "")).artifacts.get("comparison", {}).get("regressed", [])
            ) if report.validation_results.get("static_validation") else 0
            return round(50.0 + high_value * 10.0 + handoffs * 4.0 - low_value * 0.25 - regressed * 8.0, 2)
        if agent_kind == AgentKind.DYNAMIC_DEBUG:
            blocker = 1 if any(finding.severity in {Severity.HIGH, Severity.CRITICAL} for finding in result.findings) else 0
            handoffs = len(result.artifacts.get("handoffs", []))
            failed_only = 1 if any(f.rule_id == "command-failed" for f in result.findings) and len(result.findings) == 1 else 0
            tool_calls = int(result.artifacts.get("session_summary", {}).get("tool_call_count", 0) or 0)
            return round(55.0 + blocker * 12.0 + handoffs * 5.0 - failed_only * 4.0 - max(0, 2 - tool_calls) * 2.0, 2)
        patch_files = len(result.patch.file_patches) if result.patch else 0
        publishable = 1 if result.artifacts.get("publishable", False) else 0
        unresolved = len(result.artifacts.get("unsupported_findings", []))
        regressed = 0
        for validation in report.validation_results.values():
            regressed += len(validation.artifacts.get("comparison", {}).get("regressed", []))
        return round(50.0 + patch_files * 3.0 + publishable * 10.0 - unresolved * 2.0 - regressed * 6.0, 2)

    def _shadow_candidate_score(
        self,
        agent_kind: AgentKind,
        report: WorkflowReport,
        result: AgentResult,
        candidate: SkillCandidate,
        active_score: float,
    ) -> tuple[float, list[str]]:
        hints = set(candidate.notes)
        score = active_score
        reasons: list[str] = []
        low_value_ratio = (
            len([finding for finding in result.findings if finding.severity == Severity.LOW]) / max(len(result.findings), 1)
        ) if result.findings else 0.0
        if agent_kind == AgentKind.STATIC_REVIEW:
            if "suppress_low_value_noise" in hints and low_value_ratio > 0.5:
                score += 8.0
                reasons.append("candidate reduces low-value static noise")
            if "emit_fix_requests_for_high_value_findings" in hints and int(result.artifacts.get("high_value_findings", 0) or 0) > 0:
                score += 6.0
                reasons.append("candidate strengthens high-value handoffs")
            if "expand_cross_file_investigation" in hints:
                score += 4.0
                reasons.append("candidate encourages broader architectural inspection")
        elif agent_kind == AgentKind.DYNAMIC_DEBUG:
            if "inspect_traceback_before_finalize" in hints:
                score += 5.0
                reasons.append("candidate would deepen traceback inspection")
            if "always_emit_runtime_fix_requests" in hints and not result.artifacts.get("handoffs"):
                score += 7.0
                reasons.append("candidate would improve runtime handoff completeness")
            if "prioritize_dependency_contract_diagnosis" in hints and "dependency" in result.summary.lower():
                score += 5.0
                reasons.append("candidate better targets dependency blockers")
        else:
            if "expand_safe_fix_coverage" in hints and not (result.patch and result.patch.file_patches):
                score += 8.0
                reasons.append("candidate expands maintenance coverage")
            if "target_high_value_unresolved_handoffs" in hints and result.artifacts.get("unsupported_findings"):
                score += 6.0
                reasons.append("candidate targets unresolved maintenance handoffs")
            if "improve_publishability_filters" in hints and not result.artifacts.get("publishable", False):
                score += 4.0
                reasons.append("candidate improves publishability focus")
        return round(score, 2), reasons

    def _has_high_regression(self, report: WorkflowReport, agent_kind: AgentKind) -> bool:
        if agent_kind == AgentKind.MAINTENANCE:
            return any(
                any(finding.severity in {Severity.HIGH, Severity.CRITICAL} for finding in result.findings)
                or bool(result.artifacts.get("comparison", {}).get("regressed"))
                for result in report.validation_results.values()
            )
        validation_key = "static_validation" if agent_kind == AgentKind.STATIC_REVIEW else "dynamic_validation"
        result = report.validation_results.get(validation_key)
        if result is None:
            return False
        return any(finding.severity in {Severity.HIGH, Severity.CRITICAL} for finding in result.findings) or bool(
            result.artifacts.get("comparison", {}).get("regressed")
        )

    def _auto_upgrade_enabled(self, agent_kind: AgentKind) -> bool:
        skill_config = {
            AgentKind.STATIC_REVIEW: self.config.skills.static,
            AgentKind.DYNAMIC_DEBUG: self.config.skills.dynamic,
            AgentKind.MAINTENANCE: self.config.skills.maintenance,
        }[agent_kind]
        return self.config.skills.enabled and bool(skill_config.auto_upgrade)

    def _promoted_binding(self, repo_root: str, agent_kind: AgentKind, version: str):
        from core.models import SkillBinding

        return SkillBinding(
            repo_root=repo_root,
            agent_kind=agent_kind,
            active_version=version,
            source=SkillSource.DATABASE,
            frozen=False,
        )
