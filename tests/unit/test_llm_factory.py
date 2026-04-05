from __future__ import annotations

import importlib
import logging

import pytest

from core.config import LLMConfig
from llm.factory import build_llm_client
from llm.mock import MockLLMClient


def test_llm_factory_returns_mock_for_mock_provider() -> None:
    client = build_llm_client(LLMConfig(provider="mock"), logging.getLogger("test"))

    assert isinstance(client, MockLLMClient)


@pytest.mark.parametrize(
    ("provider", "api_key_env"),
    [
        ("openai", "MISSING_OPENAI_KEY"),
        ("openai_compatible", "MISSING_OPENAI_COMPAT_KEY"),
        ("anthropic", "MISSING_ANTHROPIC_KEY"),
        ("google_genai", "MISSING_GOOGLE_KEY"),
    ],
)
def test_llm_factory_raises_when_required_env_missing(
    provider: str,
    api_key_env: str,
) -> None:
    with pytest.raises(ValueError, match=api_key_env):
        build_llm_client(
            LLMConfig(
                provider=provider,
                model="gpt-4.1-mini",
                base_url="https://example.invalid/v1",
                api_key_env=api_key_env,
            ),
            logging.getLogger("test"),
        )


@pytest.mark.parametrize(
    ("provider", "module_name", "class_name", "api_key_env"),
    [
        ("openai", "llm.openai_native", "OpenAINativeLLMClient", "OPENAI_API_KEY"),
        ("openai_compatible", "llm.openai_compatible", "OpenAICompatibleLLMClient", "OPENAI_API_KEY"),
        ("anthropic", "llm.anthropic_native", "AnthropicNativeLLMClient", "ANTHROPIC_API_KEY"),
        ("google_genai", "llm.google_genai", "GoogleGenAILLMClient", "GOOGLE_API_KEY"),
        ("ollama", "llm.ollama_local", "OllamaLocalLLMClient", ""),
    ],
)
def test_llm_factory_builds_supported_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    module_name: str,
    class_name: str,
    api_key_env: str,
) -> None:
    module = importlib.import_module(module_name)
    client_cls = getattr(module, class_name)
    monkeypatch.setattr(client_cls, "_create_chat_model", lambda self, config: object())
    if api_key_env:
        monkeypatch.setenv(api_key_env, "test-key")

    client = build_llm_client(
        LLMConfig(
            provider=provider,
            model="demo-model",
            base_url="http://127.0.0.1:11434" if provider == "ollama" else None,
            api_key_env=api_key_env,
        ),
        logging.getLogger("test"),
    )

    assert getattr(client, "provider_name", None) == provider


def test_llm_factory_raises_for_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm_client(
            LLMConfig(provider="mystery-provider", model="demo-model"),
            logging.getLogger("test"),
        )
