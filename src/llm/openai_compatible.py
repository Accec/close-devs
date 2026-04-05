from __future__ import annotations

from llm.openai_native import OpenAINativeLLMClient


class OpenAICompatibleLLMClient(OpenAINativeLLMClient):
    provider_name = "openai_compatible"
