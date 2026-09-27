"""OpenRouter provider — OpenAI-compat SDK with OpenRouter base URL."""

from __future__ import annotations

import time

import openai as _openai

from src import config
from src.llm.messages import ChatMessage, LLMError, LLMResponse

_OR_CLIENT: _openai.OpenAI | None = None


def _get_client() -> _openai.OpenAI:
    """Return a cached singleton OpenRouter OpenAI-compat client."""
    global _OR_CLIENT
    if _OR_CLIENT is None:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        _OR_CLIENT = _openai.OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _OR_CLIENT


def _openrouter_generate(
    *,
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> LLMResponse:
    """Generate a single completion via OpenRouter."""
    try:
        client = _get_client()
    except Exception as exc:
        raise LLMError(provider="openrouter", model=model, code=None, original=exc) from exc

    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": m.role, "content": m.content} for m in messages
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.monotonic()
    try:
        response = client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        elapsed = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
            latency_ms=round(elapsed, 1),
        )
    except _openai.OpenAIError as exc:
        code: int | None = getattr(exc, "status_code", None)
        raise LLMError(
            provider="openrouter",
            model=model,
            code=code,
            original=exc,
        ) from exc
    except Exception as exc:
        raise LLMError(
            provider="openrouter",
            model=model,
            code=None,
            original=exc,
        ) from exc
