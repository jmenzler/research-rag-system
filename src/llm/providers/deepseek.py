"""DeepSeek provider — OpenAI-compat SDK with thinking disabled.

DeepSeek V4 models default to reasoning/thinking mode, which produces
``reasoning_content`` that consumes the entire ``max_tokens`` budget before
emitting the actual response. We disable it on every call via the
``extra_body`` injection pattern from ``src/evaluate.py:113-134``.
"""

from __future__ import annotations

import time

import openai as _openai

from src import config
from src.llm.messages import ChatMessage, LLMError, LLMResponse

_DS_CLIENT: _openai.OpenAI | None = None


def _get_client() -> _openai.OpenAI:
    """Return a cached singleton DeepSeek OpenAI-compat client."""
    global _DS_CLIENT
    if _DS_CLIENT is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        _DS_CLIENT = _openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
    return _DS_CLIENT


def _deepseek_generate(
    *,
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> LLMResponse:
    """Generate a single completion via DeepSeek."""
    try:
        client = _get_client()
    except Exception as exc:
        raise LLMError(provider="deepseek", model=model, code=None, original=exc) from exc

    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": m.role, "content": m.content} for m in messages
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Always disable thinking — reasoning_content burns the output budget.
        "extra_body": {"thinking": {"type": "disabled"}},
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
            provider="deepseek",
            model=model,
            code=code,
            original=exc,
        ) from exc
    except Exception as exc:
        raise LLMError(
            provider="deepseek",
            model=model,
            code=None,
            original=exc,
        ) from exc
