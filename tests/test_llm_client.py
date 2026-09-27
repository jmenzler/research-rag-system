"""Acceptance tests for src/llm/ — LLM client facade + provider routing.

These MUST fail before src/llm/ is implemented (TDD gate)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Dataclass smoke — should import cleanly once messages.py exists
# ---------------------------------------------------------------------------


def test_chat_message_creation() -> None:
    """ChatMessage dataclass with role + content."""
    from src.llm.messages import ChatMessage

    msg = ChatMessage(role="system", content="You are helpful.")
    assert msg.role == "system"
    assert msg.content == "You are helpful."


def test_llm_response_fields() -> None:
    """LLMResponse carries text + usage + model + latency."""
    from src.llm.messages import LLMResponse

    resp = LLMResponse(
        text="OK",
        input_tokens=5,
        output_tokens=1,
        model="gemini-2.5-flash",
        latency_ms=123.4,
    )
    assert resp.text == "OK"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 1
    assert resp.model == "gemini-2.5-flash"
    assert resp.latency_ms == 123.4


def test_llm_error_fields() -> None:
    """LLMError carries provider / model / code / original exception."""
    from src.llm.messages import LLMError

    orig = ValueError("boom")
    err = LLMError(
        provider="gemini",
        model="gemini-2.5-flash",
        code=500,
        original=orig,
    )
    assert err.provider == "gemini"
    assert err.model == "gemini-2.5-flash"
    assert err.code == 500
    assert err.original is orig
    assert "gemini" in str(err)
    assert "500" in str(err)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def test_provider_routing_gemini() -> None:
    """gemini-* prefix routes to Gemini provider."""
    from src.llm.client import LLMClient

    client = LLMClient(model="gemini-2.5-flash")
    assert client._provider_name == "gemini"


def test_provider_routing_deepseek() -> None:
    """deepseek-* prefix routes to DeepSeek provider."""
    from src.llm.client import LLMClient

    client = LLMClient(model="deepseek-v4-flash")
    assert client._provider_name == "deepseek"


def test_provider_routing_openrouter() -> None:
    """Anything else routes to OpenRouter."""
    from src.llm.client import LLMClient

    client = LLMClient(model="anthropic/claude-sonnet-4")
    assert client._provider_name == "openrouter"


def test_provider_routing_empty_fallback() -> None:
    """Empty model string still falls back to openrouter."""
    from src.llm.client import LLMClient

    client = LLMClient(model="")
    assert client._provider_name == "openrouter"


# ---------------------------------------------------------------------------
# End-to-end smoke — Gemini
# ---------------------------------------------------------------------------

_GEMINI_SMOKE_REASON = "Requires GEMINI_API_KEY in .env"


@pytest.mark.skipif(
    "not __import__('os').getenv('GEMINI_API_KEY')",
    reason=_GEMINI_SMOKE_REASON,
)
def test_gemini_smoke_roundtrip() -> None:
    """Real Gemini call returns 'OK'."""
    from src.llm import LLMClient
    from src.llm.messages import ChatMessage

    client = LLMClient(model="gemini-2.5-flash")
    resp = client.generate(
        messages=[ChatMessage(role="user", content="Reply with the word OK only.")],
        temperature=0.0,
        max_tokens=10,
    )
    assert resp.text.strip().upper() == "OK"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.latency_ms > 0


@pytest.mark.skipif(
    "not __import__('os').getenv('GEMINI_API_KEY')",
    reason=_GEMINI_SMOKE_REASON,
)
def test_gemini_json_mode() -> None:
    """Gemini with json_mode=True produces parseable JSON."""
    import json

    from src.llm import LLMClient
    from src.llm.messages import ChatMessage

    client = LLMClient(model="gemini-2.5-flash")
    resp = client.generate(
        messages=[
            ChatMessage(
                role="user",
                content='Return JSON: {"word": "hello", "count": 1}',
            )
        ],
        temperature=0.0,
        max_tokens=50,
        json_mode=True,
    )
    parsed = json.loads(resp.text)
    assert isinstance(parsed, dict)
    assert "word" in parsed


# ---------------------------------------------------------------------------
# End-to-end smoke — DeepSeek
# ---------------------------------------------------------------------------

_DEEPSEEK_SMOKE_REASON = "Requires DEEPSEEK_API_KEY in .env"


@pytest.mark.skipif(
    "not __import__('os').getenv('DEEPSEEK_API_KEY')",
    reason=_DEEPSEEK_SMOKE_REASON,
)
def test_deepseek_smoke_roundtrip() -> None:
    """Real DeepSeek call returns 'OK' (thinking disabled so no reasoning_content)."""
    from src.llm import LLMClient
    from src.llm.messages import ChatMessage

    client = LLMClient(model="deepseek-v4-flash")
    resp = client.generate(
        messages=[ChatMessage(role="user", content="Reply with the word OK only.")],
        temperature=0.0,
        max_tokens=10,
    )
    assert resp.text.strip().upper() == "OK"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
