from __future__ import annotations

import os

from core.config import LLMConfig
from llm.langchain_structured import StructuredLangChainLLMClient


class GoogleGenAILLMClient(StructuredLangChainLLMClient):
    provider_name = "google_genai"

    def _create_chat_model(self, config: LLMConfig) -> object:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key env var: {config.api_key_env}")

        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, object] = {
            "model": config.model,
            "api_key": api_key,
            "temperature": config.temperature,
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
        return ChatGoogleGenerativeAI(**kwargs)
