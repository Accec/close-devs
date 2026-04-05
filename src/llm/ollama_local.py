from __future__ import annotations

from core.config import LLMConfig, default_base_url_for_provider
from llm.langchain_structured import StructuredLangChainLLMClient


class OllamaLocalLLMClient(StructuredLangChainLLMClient):
    provider_name = "ollama"

    def _create_chat_model(self, config: LLMConfig) -> object:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, object] = {
            "model": config.model,
            "temperature": config.temperature,
            "base_url": config.base_url or default_base_url_for_provider("ollama"),
        }
        return ChatOllama(**kwargs)
