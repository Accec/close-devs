from __future__ import annotations

import logging

from core.config import LLMConfig
from llm.base import BaseLLMClient
from llm.mock import MockLLMClient


def build_llm_client(config: LLMConfig, logger: logging.Logger) -> BaseLLMClient:
    if config.provider == "mock":
        return MockLLMClient()
    if config.provider == "openai_compatible":
        try:
            from llm.openai_compatible import OpenAICompatibleLLMClient

            return OpenAICompatibleLLMClient(config)
        except Exception as exc:
            logger.warning(
                "Failed to initialize openai-compatible client (%s). Falling back to mock.",
                exc,
            )
            return MockLLMClient()

    logger.warning("Unknown LLM provider '%s'. Falling back to mock.", config.provider)
    return MockLLMClient()
