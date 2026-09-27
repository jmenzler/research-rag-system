"""Gemini provider — google-genai client with singleton caching."""

from __future__ import annotations

import time

import httpx
from google import genai
from google.genai.errors import APIError
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig

from src import config
from src.llm.messages import ChatMessage, LLMError, LLMResponse

_GEMINI_CLIENT: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a cached singleton Gemini client with tuned httpx pools.

    Mirrors the pattern in ``src/contextualize.py:202-224``.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        config.validate_api_key()
        http_options = {
            "async_client_args": {
                "limits": httpx.Limits(max_connections=60, max_keepalive_connections=30),
                "timeout": httpx.Timeout(120.0, connect=10.0),
            },
            "client_args": {
                "limits": httpx.Limits(max_connections=60, max_keepalive_connections=30),
                "timeout": httpx.Timeout(120.0, connect=10.0),
            },
        }
        _GEMINI_CLIENT = genai.Client(api_key=config.GEMINI_API_KEY, http_options=http_options)  # type: ignore[arg-type]
    return _GEMINI_CLIENT


def _gemini_generate(
    *,
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> LLMResponse:
    """Generate a single completion via Gemini.

    Maps ChatMessage[] → contents (user messages) + system_instruction
    (first system message). Gemini does not support multiple system messages
    or interleaved assistant/user turns in this API — we concatenate.
    """
    try:
        client = _get_client()
    except Exception as exc:
        raise LLMError(provider="gemini", model=model, code=None, original=exc) from exc

    # Split: first system message → system_instruction; everything else → contents.
    system_instruction: str | None = None
    parts: list[Part] = []
    for msg in messages:
        if msg.role == "system" and system_instruction is None:
            system_instruction = msg.content
        else:
            parts.append(Part.from_text(text=f"[{msg.role}] {msg.content}"))

    user_text = "\n\n".join(
        p.text for p in parts if hasattr(p, "text") and p.text is not None
    )

    t0 = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[Part.from_text(text=user_text)] if user_text else parts,  # type: ignore[arg-type]
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json" if json_mode else None,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
        elapsed = (time.monotonic() - t0) * 1000

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return LLMResponse(
            text=(response.text or "").strip(),
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            model=model,
            latency_ms=round(elapsed, 1),
        )
    except APIError as exc:
        raise LLMError(
            provider="gemini",
            model=model,
            code=getattr(exc, "code", None),
            original=exc,
        ) from exc
    except Exception as exc:
        raise LLMError(
            provider="gemini",
            model=model,
            code=None,
            original=exc,
        ) from exc
