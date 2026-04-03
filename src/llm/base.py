from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import AgentSession, AgentStep, SkillCandidate, SkillPack, ToolSpec


class BaseLLMClient(ABC):
    @abstractmethod
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
        """Return the next agent step for the current session."""
