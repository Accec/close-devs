from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any

from core.config import AgentRuntimeConfig, AppConfig
from core.models import AgentKind, SkillCandidate, SkillCandidateStatus, SkillPack, SkillPolicy, SkillSource
from memory.state_store import StateStore
from reports.serializer import to_jsonable


AGENT_SKILL_DIRS: dict[AgentKind, str] = {
    AgentKind.STATIC_REVIEW: "static",
    AgentKind.DYNAMIC_DEBUG: "dynamic",
    AgentKind.MAINTENANCE: "maintenance",
}


class SkillManager:
    def __init__(self, config: AppConfig, state_store: StateStore) -> None:
        self.config = config
        self.state_store = state_store

    async def resolve_run_skills(
        self,
        repo_root: Path | str,
        runtime_configs: dict[AgentKind, AgentRuntimeConfig],
    ) -> tuple[dict[str, SkillPack], dict[str, SkillCandidate]]:
        active: dict[str, SkillPack] = {}
        candidates: dict[str, SkillCandidate] = {}
        repo_key = str(repo_root)
        for agent_kind, runtime_config in runtime_configs.items():
            baseline = self._load_baseline_skill(agent_kind)
            baseline = self._clamp_pack(baseline, runtime_config)
            await self.state_store.upsert_skill_pack(repo_key, baseline)

            binding = await self.state_store.get_skill_binding(repo_key, agent_kind)
            if binding is None:
                await self.state_store.set_skill_binding(
                    self._default_binding(repo_key, agent_kind, baseline.version)
                )
                active_pack = baseline
            else:
                active_pack = await self._resolve_bound_pack(repo_key, agent_kind, binding.active_version, baseline)

            active[str(agent_kind.value)] = self._clamp_pack(active_pack, runtime_config)
            candidate = await self.state_store.get_open_skill_candidate(repo_key, agent_kind)
            if candidate is not None:
                clamped_candidate_pack = self._clamp_pack(candidate.skill_pack, runtime_config)
                candidates[str(agent_kind.value)] = SkillCandidate(
                    candidate_id=candidate.candidate_id,
                    repo_root=candidate.repo_root,
                    agent_kind=candidate.agent_kind,
                    based_on_version=candidate.based_on_version,
                    version=candidate.version,
                    skill_pack=clamped_candidate_pack,
                    status=candidate.status,
                    created_at=candidate.created_at,
                    shadow_runs=candidate.shadow_runs,
                    notes=list(candidate.notes),
                )
        return active, candidates

    async def skill_status(self, repo_root: Path | str) -> list[dict[str, Any]]:
        status_rows: list[dict[str, Any]] = []
        active, candidates = await self.resolve_run_skills(
            repo_root,
            {
                AgentKind.STATIC_REVIEW: self.config.agents.static,
                AgentKind.DYNAMIC_DEBUG: self.config.agents.dynamic,
                AgentKind.MAINTENANCE: self.config.agents.maintenance,
            },
        )
        repo_key = str(repo_root)
        for agent_kind in AgentKind:
            binding = await self.state_store.get_skill_binding(repo_key, agent_kind)
            candidate = candidates.get(agent_kind.value)
            active_pack = active[agent_kind.value]
            status_rows.append(
                {
                    "agent": agent_kind.value,
                    "active_version": active_pack.version,
                    "active_source": active_pack.source.value,
                    "frozen": bool(binding.frozen) if binding is not None else False,
                    "candidate_version": candidate.version if candidate is not None else None,
                    "candidate_shadow_runs": candidate.shadow_runs if candidate is not None else 0,
                    "candidate_status": candidate.status.value if candidate is not None else None,
                    "candidate_cooldown_until": candidate.cooldown_until.isoformat() if candidate and candidate.cooldown_until else None,
                }
            )
        return status_rows

    async def freeze(self, repo_root: Path | str, agent_kind: AgentKind, frozen: bool) -> None:
        await self.state_store.set_skill_binding_frozen(str(repo_root), agent_kind, frozen)

    async def manual_promote(self, repo_root: Path | str, agent_kind: AgentKind) -> bool:
        candidate = await self.state_store.get_open_skill_candidate(str(repo_root), agent_kind)
        if candidate is None:
            return False
        binding = await self.state_store.get_skill_binding(str(repo_root), agent_kind)
        if binding is not None and binding.frozen:
            return False
        await self.state_store.upsert_skill_pack(str(repo_root), candidate.skill_pack)
        await self.state_store.set_skill_binding(
            self._default_binding(str(repo_root), agent_kind, candidate.version, source=SkillSource.DATABASE)
        )
        await self.state_store.update_skill_candidate(
            candidate.candidate_id,
            status=SkillCandidateStatus.PROMOTED,
            shadow_runs=candidate.shadow_runs,
            notes=[*candidate.notes, "Manually promoted."],
        )
        return True

    async def history(self, repo_root: Path | str, agent_kind: AgentKind) -> list[dict[str, Any]]:
        evaluations = await self.state_store.recent_skill_evaluations(str(repo_root), agent_kind, limit=20)
        return [
            {
                "evaluation_id": item.evaluation_id,
                "run_id": item.run_id,
                "active_version": item.active_version,
                "candidate_version": item.candidate_version,
                "active_score": item.active_score,
                "candidate_score": item.candidate_score,
                "mode": item.mode.value,
                "promoted": item.promoted,
                "reasons": item.reasons,
                "created_at": item.created_at.isoformat(),
            }
            for item in evaluations
        ]

    def _default_binding(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        version: str,
        *,
        source: SkillSource = SkillSource.REPO,
    ):
        from core.models import SkillBinding

        return SkillBinding(
            repo_root=repo_root,
            agent_kind=agent_kind,
            active_version=version,
            source=source,
            frozen=False,
        )

    async def _resolve_bound_pack(
        self,
        repo_root: str,
        agent_kind: AgentKind,
        version: str,
        baseline: SkillPack,
    ) -> SkillPack:
        if version == baseline.version:
            return baseline
        pack = await self.state_store.get_skill_pack(repo_root, agent_kind, version)
        if pack is not None:
            return pack
        return baseline

    def _load_baseline_skill(self, agent_kind: AgentKind) -> SkillPack:
        skill_dir = self.config.skills.repo_root / AGENT_SKILL_DIRS[agent_kind]
        manifest = self._load_toml(skill_dir / "manifest.toml")
        policy_raw = self._load_toml(skill_dir / "policy.toml")
        skill_markdown = (skill_dir / "skill.md").read_text(encoding="utf-8")
        examples = json.loads((skill_dir / "examples.json").read_text(encoding="utf-8"))
        policy = SkillPolicy(
            planning_heuristics=[str(item) for item in policy_raw.get("planning_heuristics", [])],
            tool_preferences=[str(item) for item in policy_raw.get("tool_preferences", [])],
            forbidden_ordering=[str(item) for item in policy_raw.get("forbidden_ordering", [])],
            severity_bias={str(key): float(value) for key, value in dict(policy_raw.get("severity_bias", {})).items()},
            rule_weights={str(key): float(value) for key, value in dict(policy_raw.get("rule_weights", {})).items()},
            command_preferences=[str(item) for item in policy_raw.get("command_preferences", [])],
            completion_checklist=[str(item) for item in policy_raw.get("completion_checklist", [])],
            handoff_style=str(policy_raw.get("handoff_style", "")),
            patch_style=str(policy_raw.get("patch_style", "")),
            recommended_max_steps=int(policy_raw["recommended_max_steps"]) if policy_raw.get("recommended_max_steps") is not None else None,
            recommended_max_tool_calls=int(policy_raw["recommended_max_tool_calls"]) if policy_raw.get("recommended_max_tool_calls") is not None else None,
            recommended_max_wall_time_seconds=int(policy_raw["recommended_max_wall_time_seconds"]) if policy_raw.get("recommended_max_wall_time_seconds") is not None else None,
            allowed_tools=[str(item) for item in policy_raw.get("allowed_tools", [])],
            environment_preferences=[str(item) for item in policy_raw.get("environment_preferences", [])],
            upgrade_constraints=[str(item) for item in policy_raw.get("upgrade_constraints", [])],
            noise_suppression=[str(item) for item in policy_raw.get("noise_suppression", [])],
        )
        system_prompt = str(manifest.get("system_prompt", "")).strip()
        pack = SkillPack(
            agent_kind=agent_kind,
            name=str(manifest.get("name", f"{agent_kind.value}-baseline")),
            version=str(manifest.get("version", "baseline-v1")),
            description=str(manifest.get("description", "")),
            status=str(manifest.get("status", "active")),
            source=SkillSource.REPO,
            system_prompt=system_prompt,
            skill_markdown=skill_markdown,
            examples=[dict(item) for item in examples if isinstance(item, dict)],
            policy=policy,
        )
        pack.profile_hash = self._hash_pack(pack)
        return pack

    def _clamp_pack(self, pack: SkillPack, runtime_config: AgentRuntimeConfig) -> SkillPack:
        allowed_superset = runtime_config.allowed_tool_superset or runtime_config.allowed_tools
        allowed_tools = (
            [tool for tool in pack.policy.allowed_tools if tool in allowed_superset]
            if pack.policy.allowed_tools
            else list(allowed_superset)
        )
        recommended_steps = self._clamp_int(
            pack.policy.recommended_max_steps,
            lower=1,
            upper=runtime_config.max_budget_ceiling or runtime_config.max_steps,
            default=runtime_config.max_steps,
        )
        recommended_tool_calls = self._clamp_int(
            pack.policy.recommended_max_tool_calls,
            lower=1,
            upper=runtime_config.max_tool_calls,
            default=runtime_config.max_tool_calls,
        )
        recommended_wall_time = self._clamp_int(
            pack.policy.recommended_max_wall_time_seconds,
            lower=30,
            upper=runtime_config.max_wall_time_seconds,
            default=runtime_config.max_wall_time_seconds,
        )
        clamped = SkillPack(
            agent_kind=pack.agent_kind,
            name=pack.name,
            version=pack.version,
            description=pack.description,
            status=pack.status,
            source=pack.source,
            system_prompt=pack.system_prompt,
            skill_markdown=pack.skill_markdown,
            examples=[dict(item) for item in pack.examples],
            policy=SkillPolicy(
                planning_heuristics=list(pack.policy.planning_heuristics),
                tool_preferences=[tool for tool in pack.policy.tool_preferences if tool in allowed_tools],
                forbidden_ordering=list(pack.policy.forbidden_ordering),
                severity_bias=dict(pack.policy.severity_bias),
                rule_weights=dict(pack.policy.rule_weights),
                command_preferences=list(pack.policy.command_preferences),
                completion_checklist=list(pack.policy.completion_checklist),
                handoff_style=pack.policy.handoff_style,
                patch_style=pack.policy.patch_style,
                recommended_max_steps=recommended_steps,
                recommended_max_tool_calls=recommended_tool_calls,
                recommended_max_wall_time_seconds=recommended_wall_time,
                allowed_tools=allowed_tools,
                environment_preferences=list(pack.policy.environment_preferences),
                upgrade_constraints=list(pack.policy.upgrade_constraints),
                noise_suppression=list(pack.policy.noise_suppression),
            ),
        )
        clamped.profile_hash = self._hash_pack(clamped)
        return clamped

    def _hash_pack(self, pack: SkillPack) -> str:
        payload = {
            "agent_kind": pack.agent_kind.value,
            "name": pack.name,
            "version": pack.version,
            "description": pack.description,
            "status": pack.status,
            "source": pack.source.value,
            "system_prompt": pack.system_prompt,
            "skill_markdown": pack.skill_markdown,
            "examples": pack.examples,
            "policy": to_jsonable(pack.policy),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def _clamp_int(self, value: int | None, *, lower: int, upper: int, default: int) -> int:
        if value is None:
            return default
        return max(lower, min(int(value), upper))

    def _load_toml(self, path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            return tomllib.load(handle)
