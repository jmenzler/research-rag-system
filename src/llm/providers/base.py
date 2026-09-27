"""Provider abstract interface.

Every LLM provider (Gemini, DeepSeek, OpenRouter) implements this protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.llm.messages import ChatMessage, LLMResponse


class Provider(Protocol):
    """Generate a single chat completion.

    Implementations MUST:
    - Raise ``LLMError`` on any failure (no raw SDK exceptions).
    - Set ``json_mode`` → enforce structured JSON output per-provider conventions.
    - Handle API key presence internally (raise RuntimeError if missing).
    """

    def generate(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        ...
