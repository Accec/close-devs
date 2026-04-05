from __future__ import annotations

from abc import ABC, abstractmethod
import json
from json import JSONDecodeError
import re

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import LLMConfig
from core.models import AgentActionType, AgentKind, AgentSession, AgentStep, SkillCandidate, SkillPack, ToolSpec
from llm.base import BaseLLMClient
from reports.serializer import to_jsonable


class StructuredLangChainLLMClient(BaseLLMClient, ABC):
    provider_name = "unknown"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = self._create_chat_model(config)

    @abstractmethod
    def _create_chat_model(self, config: LLMConfig) -> object:
        raise NotImplementedError

    async def complete_agent_step(
        self,
        *,
        session: AgentSession,
        available_tools: list[ToolSpec],
        skill_profile: SkillPack | None = None,
        candidate_skill: SkillCandidate | None = None,
        agent_policy: dict[str, object] | None = None,
        skill_examples: list[dict[str, object]] | None = None,
        skill_version: str | None = None,
    ) -> AgentStep:
        payload = {
            "session": {
                "session_id": session.session_id,
                "run_id": session.run_id,
                "task_id": session.task_id,
                "agent_kind": session.agent_kind.value,
                "task_type": session.task_type.value,
                "objective": session.objective,
                "targets": session.targets,
                "payload": session.payload,
                "step_count": session.step_count,
                "tool_call_count": session.tool_call_count,
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "requires_write": tool.requires_write,
                }
                for tool in available_tools
            ],
            "history": [
                {
                    "step_index": step.step_index,
                    "decision_summary": step.decision_summary,
                    "action_type": step.action_type.value,
                    "tool_name": step.tool_name,
                    "tool_input": step.tool_input,
                }
                for step in session.steps[-8:]
            ],
            "recent_tool_calls": [
                {
                    "step_index": call.step_index,
                    "tool_name": call.tool_name,
                    "status": call.status,
                    "summary": call.summary,
                    "error": call.error,
                    "output": call.output,
                }
                for call in session.tool_calls[-8:]
            ],
            "skill_profile": to_jsonable(skill_profile) if skill_profile is not None else None,
            "skill_examples": to_jsonable(skill_examples or []),
            "skill_version": skill_version or (skill_profile.version if skill_profile is not None else None),
            "candidate_skill_version": candidate_skill.version if candidate_skill is not None else None,
            "agent_policy": to_jsonable(agent_policy or {}),
        }
        prompt_payload = self._truncate_payload(to_jsonable(payload))
        role_guidance = self._role_specific_guidance(session, skill_profile)
        skill_system_prompt = skill_profile.system_prompt if skill_profile is not None else ""
        prompt = (
            "You are an autonomous software maintenance agent. "
            "Choose exactly one next action.\n"
            "Return strict JSON with keys:\n"
            "- decision_summary: string\n"
            "- action_type: 'tool_call' or 'finalize'\n"
            "- tool_name: string|null\n"
            "- tool_input: object\n"
            "- final_response: object\n\n"
            "When finalizing, include final_response.summary and any role-specific structured output.\n"
            "Structured findings should include category, severity, message, and root_cause_class when applicable.\n"
            "Structured fix requests should include kind, confidence, affected_files, and metadata.root_cause_class when applicable.\n"
            f"{role_guidance}\n"
            "Do not include markdown fences.\n\n"
            f"Context JSON:\n{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
        )
        response = await self.client.ainvoke(
            [
                SystemMessage(
                    content="\n\n".join(
                        item for item in [self.config.system_prompt, skill_system_prompt] if item
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        data = self._parse_json_response(self._message_text(response.content))
        return AgentStep(
            step_index=session.step_count + 1,
            decision_summary=str(data.get("decision_summary") or "Model-selected next step."),
            action_type=AgentActionType(data.get("action_type", AgentActionType.FINALIZE.value)),
            tool_name=str(data["tool_name"]) if data.get("tool_name") else None,
            tool_input=dict(data.get("tool_input", {})),
            final_response=dict(data.get("final_response", {})),
        )

    def _message_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)

    def _parse_json_response(self, text: str) -> dict[str, object]:
        candidates = self._json_candidates(text)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except JSONDecodeError as exc:
                last_error = exc
                try:
                    data = self._decode_first_json_object(candidate)
                except JSONDecodeError as nested_exc:
                    last_error = nested_exc
                    continue
            if not isinstance(data, dict):
                last_error = ValueError("Model response was not a JSON object.")
                continue
            return data
        if last_error is not None:
            raise last_error
        raise ValueError("Model response did not contain JSON.")

    def _json_candidates(self, text: str) -> list[str]:
        stripped = text.strip()
        candidates: list[str] = []
        if stripped:
            candidates.append(stripped)
        fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        for match in fence_matches:
            fenced = match.strip()
            if fenced:
                candidates.append(fenced)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _decode_first_json_object(self, text: str) -> dict[str, object]:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text[index:])
            except JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise JSONDecodeError("No JSON object found in model response.", text, 0)

    def _truncate_payload(
        self,
        value: object,
        *,
        path: tuple[str, ...] = (),
    ) -> object:
        if isinstance(value, str):
            return value[:4000]
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, list):
            limit = self._list_limit_for_path(path)
            return [
                self._truncate_payload(item, path=(*path, str(index)))
                for index, item in enumerate(value[:limit])
            ]
        if isinstance(value, dict):
            truncated: dict[str, object] = {}
            for key, item in list(value.items())[: self._dict_limit_for_path(path)]:
                truncated[str(key)] = self._truncate_payload(item, path=(*path, str(key)))
            return truncated
        return str(value)

    def _list_limit_for_path(self, path: tuple[str, ...]) -> int:
        normalized = ".".join(path)
        limits = {
            "session.payload.static_context.top_targets": 40,
            "session.payload.static_context.high_signal_targets": 20,
            "session.payload.static_context.related_files": 20,
            "session.payload.static_context.baseline_static_digest.top_findings": 50,
            "session.payload.static_context.baseline_tool_digest.top_findings": 50,
            "session.payload.static_context.startup_topology.entrypoints": 20,
            "session.payload.static_context.startup_topology.config_anchors": 20,
            "session.payload.static_context.project_topology.entrypoints": 24,
            "session.payload.static_context.project_topology.config_anchors": 24,
            "session.payload.static_context.project_topology.dependency_manifests": 24,
            "session.payload.static_context.project_topology.lockfiles": 20,
            "session.payload.static_context.config_anchor_digest": 20,
            "session.payload.static_context.import_adjacency_digest.related_files": 20,
            "session.payload.static_context.import_adjacency_digest.edges": 20,
            "session.payload.static_context.target_digest.prioritized_targets": 40,
            "session.payload.static_context.target_digest.top_targets": 40,
            "session.payload.static_context.target_digest.high_signal_targets": 20,
            "session.payload.static_context.repo_map_summary.languages": 12,
            "session.payload.static_context.repo_map_summary.ecosystems": 12,
            "session.payload.static_context.language_profile.languages": 12,
            "session.payload.static_context.language_profile.ecosystems": 12,
            "session.payload.static_context.tool_coverage_summary.enabled_tools": 20,
            "session.payload.static_context.tool_coverage_summary.unavailable_tools": 20,
            "session.payload.static_context.tool_coverage_summary.executed_tools": 20,
        }
        return limits.get(normalized, 12)

    def _dict_limit_for_path(self, path: tuple[str, ...]) -> int:
        normalized = ".".join(path)
        if normalized == "session.payload.static_context":
            return 32
        if normalized in {
            "session.payload.static_context.repo_map_summary",
            "session.payload.static_context.baseline_static_digest",
            "session.payload.static_context.import_adjacency_digest",
        }:
            return 24
        return 20

    def _role_specific_guidance(self, session: AgentSession, skill_profile: SkillPack | None) -> str:
        skill_lines: list[str] = []
        if skill_profile is not None:
            if skill_profile.policy.planning_heuristics:
                skill_lines.append(
                    "Skill planning heuristics:\n- " + "\n- ".join(skill_profile.policy.planning_heuristics)
                )
            if skill_profile.policy.tool_preferences:
                skill_lines.append(
                    "Preferred tool order:\n- " + "\n- ".join(skill_profile.policy.tool_preferences)
                )
            if skill_profile.policy.completion_checklist:
                skill_lines.append(
                    "Completion checklist:\n- " + "\n- ".join(skill_profile.policy.completion_checklist)
                )
        if session.agent_kind == AgentKind.STATIC_REVIEW:
            role_text = (
                "Static review policy:\n"
                "- Prioritize correctness, architecture, dependency contracts, initialization order, unsafe exception handling, and configuration risks over documentation lint.\n"
                "- Read session.payload.static_context before choosing tools; it contains ranked targets, project topology, language profile, related files, and baseline digest context.\n"
                "- Treat project_topology, dependency manifests, config anchors, and entrypoints as first-round repo map inputs even when language-specific tools are unavailable.\n"
                "- For unsupported languages, stay in generic review mode and base findings on code, topology, and snippets rather than pretending deep language-tool coverage.\n"
                "- Use run_static_review to validate or drill deeper, not as the only source of first-round repository context.\n"
                "- If you mention any higher-value issue in summary, you must emit at least one structured finding in final_response.findings.\n"
                "- If you emit any medium/high correctness or architecture finding, you must also emit at least one fix request in final_response.fix_requests.\n"
                "- Do not let docstring noise dominate the final answer when stronger issues exist.\n"
                "- Prefer concrete paths, lines, and evidence from inspected files.\n"
            )
        elif session.agent_kind == AgentKind.DYNAMIC_DEBUG:
            role_text = (
                "Dynamic debug policy:\n"
                "- Prioritize the first reproducible blocker and convert it into a concrete root-cause finding.\n"
                "- If a runtime blocker prevents test execution, summarize that blocker explicitly and emit a fix request.\n"
                "- Classify blockers as dependency, config, environment, test, or application whenever the evidence supports it.\n"
            )
        else:
            role_text = (
                "Maintenance policy:\n"
                "- Prefer the smallest safe patch that addresses upstream findings or handoffs.\n"
                "- If no safe fix exists, explain the blocker instead of fabricating a patch.\n"
                "- When a dependency or configuration blocker is explicit, prefer targeted manifest/config changes over cosmetic edits.\n"
            )
        if skill_lines:
            role_text = f"{role_text}\n" + "\n".join(skill_lines)
        return role_text
