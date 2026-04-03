from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import LLMConfig
from core.models import AgentActionType, AgentKind, AgentSession, AgentStep, SkillCandidate, SkillPack, ToolSpec
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient
from reports.serializer import to_jsonable


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key env var: {config.api_key_env}")

        kwargs: dict[str, object] = {
            "model": config.model,
            "api_key": api_key,
            "temperature": config.temperature,
            "timeout": config.timeout_seconds,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url

        self.config = config
        self.client = ChatOpenAI(**kwargs)
        self.fallback = MockLLMClient()

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
            f"{role_guidance}\n"
            "Do not include markdown fences.\n\n"
            f"Context JSON:\n{json.dumps(prompt_payload, ensure_ascii=True, indent=2)}"
        )
        try:
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
        except Exception:
            return await self.fallback.complete_agent_step(
                session=session,
                available_tools=available_tools,
                skill_profile=skill_profile,
                candidate_skill=candidate_skill,
                agent_policy=agent_policy,
                skill_examples=skill_examples,
                skill_version=skill_version,
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
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("Model response was not a JSON object.")
        return data

    def _truncate_payload(self, value: object) -> object:
        if isinstance(value, str):
            return value[:4000]
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, list):
            return [self._truncate_payload(item) for item in value[:12]]
        if isinstance(value, dict):
            truncated: dict[str, object] = {}
            for key, item in list(value.items())[:20]:
                truncated[str(key)] = self._truncate_payload(item)
            return truncated
        return str(value)

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
            )
        else:
            role_text = (
                "Maintenance policy:\n"
                "- Prefer the smallest safe patch that addresses upstream findings or handoffs.\n"
                "- If no safe fix exists, explain the blocker instead of fabricating a patch.\n"
            )
        if skill_lines:
            role_text = f"{role_text}\n" + "\n".join(skill_lines)
        return role_text
