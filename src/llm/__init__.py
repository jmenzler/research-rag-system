"""LLM client package — provider-routing facade with centralised retry.

Public surface::

    from src.llm import LLMClient, ChatMessage, LLMResponse, LLMError

    client = LLMClient(model="gemini-2.5-flash")
    resp = client.generate(
        messages=[ChatMessage(role="user", content="Hello.")],
        temperature=0.0,
        max_tokens=200,
    )
    print(resp.text)
"""

from __future__ import annotations

from src.llm.client import LLMClient
from src.llm.messages import ChatMessage, LLMError, LLMResponse
from src.llm.retry import is_retryable_error, transient_retry

__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "is_retryable_error",
    "transient_retry",
]
