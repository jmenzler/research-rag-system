"""Shared dataclasses for the LLM client package.

Provider-agnostic message / response / error types. No SDK imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatMessage:
    """A single message in a chat conversation (OpenAI-compat shape)."""

    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Normalised response from any provider.

    ``latency_ms`` is wall-clock duration of the provider call (excluding retry
    sleeps), set by the orchestrator in ``client.py``.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float


class LLMError(Exception):
    """Wraps any provider exception into a single type callers can depend on.

    ``original`` is the underlying SDK exception so advanced handlers can
    introspect provider-specific attributes (e.g. ``genai.APIError.code``,
    ``openai.RateLimitError``) without importing every SDK.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        code: int | None,
        original: BaseException,
    ) -> None:
        self.provider = provider
        self.model = model
        self.code = code
        self.original = original
        super().__init__(str(self))

    def __str__(self) -> str:
        parts = [f"[{self.provider}] model={self.model}"]
        if self.code is not None:
            parts.append(f"code={self.code}")
        parts.append(str(self.original))
        return " ".join(parts)
