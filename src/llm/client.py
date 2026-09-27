"""LLMClient — single entry point for LLM calls across providers.

Routes by model prefix (gemini → Gemini, deepseek → DeepSeek, fallback →
OpenRouter). Centralises retry policy and per-provider singleton clients.
"""

from __future__ import annotations

import logging

from src.llm.messages import ChatMessage, LLMResponse
from src.llm.providers import get_provider as _get_provider_fn
from src.llm.retry import transient_retry

logger = logging.getLogger(__name__)


def _provider_for(model: str) -> str:
    """Return 'gemini', 'deepseek', or 'openrouter' based on model name prefix.

    Identical to the inlined copies in decompose.py, contextualize.py, and
    generate.py — lifted into one canonical location.
    """
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openrouter"


class LLMClient:
    """Provider-routing LLM client with retry.

    Usage::

        from src.llm import LLMClient, ChatMessage

        client = LLMClient(model="gemini-3.1-flash-lite-preview")
        resp = client.generate(
            messages=[ChatMessage(role="user", content="Hello.")],
            temperature=0.0,
            max_tokens=200,
        )
        print(resp.text)
    """

    def __init__(self, *, model: str) -> None:
        self._model = model
        self._provider_name = _provider_for(model)
        self._generate_fn = _get_provider_fn(self._provider_name)

    def generate(
        self,
        *,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a single chat completion, with retry on transient errors."""

        @transient_retry()
        def _call() -> LLMResponse:
            return self._generate_fn(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

        return _call()
