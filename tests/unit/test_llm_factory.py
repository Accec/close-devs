from __future__ import annotations

import logging

from core.config import LLMConfig
from llm.factory import build_llm_client
from llm.mock import MockLLMClient


def test_llm_factory_falls_back_to_mock_when_env_missing() -> None:
    client = build_llm_client(
        LLMConfig(
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.invalid/v1",
            api_key_env="MISSING_REPO_GUARDIAN_KEY",
        ),
        logging.getLogger("test"),
    )

    assert isinstance(client, MockLLMClient)
