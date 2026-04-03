from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import LLMConfig
from core.models import FeedbackBundle
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient


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

    async def summarize_feedback(self, feedback: FeedbackBundle) -> tuple[str, list[str]]:
        if not feedback.all_findings:
            return await self.fallback.summarize_feedback(feedback)

        payload = {
            "change_reason": feedback.change_set.reason,
            "changed_files": feedback.change_set.all_touched_files[:20],
            "findings": [
                {
                    "agent": finding.source_agent.value,
                    "rule_id": finding.rule_id,
                    "severity": finding.severity.value,
                    "category": finding.category,
                    "path": finding.path,
                    "message": finding.message,
                }
                for finding in feedback.all_findings[:20]
            ],
        }
        prompt = (
            "Return plain text with exactly this format:\n"
            "RATIONALE: <one concise paragraph>\n"
            "SUGGESTIONS:\n"
            "- <suggestion 1>\n"
            "- <suggestion 2>\n"
            "- <suggestion 3>\n\n"
            f"Feedback JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
        )
        try:
            response = await self.client.ainvoke(
                [
                    SystemMessage(content=self.config.system_prompt),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception:
            return await self.fallback.summarize_feedback(feedback)

        content = self._message_text(response.content)
        rationale, suggestions = self._parse_response(content)
        if not rationale:
            return await self.fallback.summarize_feedback(feedback)
        if not suggestions:
            _, suggestions = await self.fallback.summarize_feedback(feedback)
        return rationale, suggestions

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

    def _parse_response(self, text: str) -> tuple[str, list[str]]:
        rationale = ""
        suggestions: list[str] = []
        in_suggestions = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()
                in_suggestions = False
                continue
            if line.startswith("SUGGESTIONS:"):
                in_suggestions = True
                continue
            if in_suggestions and line.startswith("-"):
                suggestions.append(line.lstrip("- ").strip())
        return rationale, suggestions

