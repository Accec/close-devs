from __future__ import annotations

import logging

from core.config import LLMConfig
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient


def build_llm_client(config: LLMConfig, logger: logging.Logger) -> BaseLLMClient:
    if config.provider == "mock":
        return MockLLMClient()
    if config.provider == "openai":
        from llm.openai_native import OpenAINativeLLMClient

        return OpenAINativeLLMClient(config)
    if config.provider == "openai_compatible":
        from llm.openai_compatible import OpenAICompatibleLLMClient

        return OpenAICompatibleLLMClient(config)
    if config.provider == "anthropic":
        from llm.anthropic_native import AnthropicNativeLLMClient

        return AnthropicNativeLLMClient(config)
    if config.provider == "google_genai":
        from llm.google_genai import GoogleGenAILLMClient

        return GoogleGenAILLMClient(config)
    if config.provider == "ollama":
        from llm.ollama_local import OllamaLocalLLMClient

        return OllamaLocalLLMClient(config)

    logger.error("Unknown LLM provider '%s'.", config.provider)
    raise ValueError(f"Unknown LLM provider: {config.provider}")
