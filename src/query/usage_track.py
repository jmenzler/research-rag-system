"""LLM call wrapper that extracts Usage + latency from both Gemini and OpenAI-compat responses.

Centralises the response-object handling that was previously scattered across
``generate.py`` (synthesis path) and ``decompose.py`` / ``grader.py`` (which
discarded the response altogether).

Typical usage::

    from src.query.usage_track import call_text, LLMCallResult
    result = call_text(model, system, user, json_mode=False, temperature=0.0)
    # result.text  — the string the caller needs
    # result.usage — token counts (input / output / reasoning / total + cache split)
    # result.cost_usd — float or None if model not in pricing table
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import openai as _openai
from google import genai
from google.genai import types
from pydantic import BaseModel

from src import config
from src.config.models import compute_cost_usd
from src.models import Usage

# Per-call HTTP timeout (seconds). Default 120s. Real synthesis calls on
# DeepSeek V4 Pro routinely take 60-80s; we want to fail-fast on hung calls
# but not on legitimate slow ones. Judge calls (Gemini Flash) finish in
# 1-5s, so a 120s ceiling burns ~115s of slack on a stuck judge — acceptable
# given the application-level retry then takes over with backoff.
# Override via env var LLM_CALL_TIMEOUT_S.
_LLM_CALL_TIMEOUT_S: float = float(os.getenv("LLM_CALL_TIMEOUT_S", "180"))

# Per-process cache of OpenAI-compat models that have rejected
# `response_format={"type":"json_schema",...}` with a 400. After the first
# rejection we drop straight to `{"type":"json_object"}` for that model.
# DeepSeek's V3+ chat models are the typical culprits — they support JSON
# mode but not strict JSON schema.
#
# _json_schema_unsupported_lock guards all check-and-add operations. GIL
# covers individual set.add() on CPython, but the check-then-add pattern in
# _create_with_schema is not atomic without an explicit lock, and the language
# spec gives no GIL guarantee for future interpreters (free-threading, PyPy).
_json_schema_unsupported: set[str] = set()
_json_schema_unsupported_lock: threading.Lock = threading.Lock()


@dataclass
class LLMCallResult:
    """Everything returned from a single LLM call that callers might want to log."""

    text: str
    latency_ms: int
    usage: Usage
    raw_response: Any  # provider response object (for downstream inspection)
    cost_usd: float | None
    model: str


# ---------------------------------------------------------------------------
# Usage extractors (duplicated from generate.py so this module is self-contained;
# generate.py's copies still exist for the synthesis path which is separate).
# ---------------------------------------------------------------------------


def _usage_from_gemini(response: Any, model: str) -> Usage:  # noqa: ANN401  — duck-typed SDK response
    """Extract token usage from a google-genai GenerateContentResponse.

    Also extracts ``cached_content_token_count`` → ``input_cache_hit`` when the
    Gemini response exposes it (prompt caching opt-in).
    """
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return Usage()
    inp = int(getattr(um, "prompt_token_count", 0) or 0)
    out = int(getattr(um, "candidates_token_count", 0) or 0)
    reasoning = int(getattr(um, "thoughts_token_count", 0) or 0)
    total = int(getattr(um, "total_token_count", 0) or 0)
    if total == 0:
        total = inp + out + reasoning

    cache_hit = int(getattr(um, "cached_content_token_count", 0) or 0)
    cache_miss = max(inp - cache_hit, 0) if cache_hit > 0 else 0

    return Usage(
        input=inp,
        output=out,
        reasoning=reasoning,
        total=total,
        input_cache_hit=cache_hit,
        input_cache_miss=cache_miss,
    )


def _usage_from_openai_compat(response: Any) -> Usage:  # noqa: ANN401  — duck-typed SDK response
    """Extract token usage from an OpenAI-compatible chat completion response.

    Reads DeepSeek cache-split fields (``prompt_cache_hit_tokens`` /
    ``prompt_cache_miss_tokens``) when present.
    """
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    inp = int(getattr(u, "prompt_tokens", 0) or 0)
    out = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", 0) or 0)
    reasoning = 0
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
    if total == 0:
        total = inp + out + reasoning

    # DeepSeek-specific cache accounting (50× price differential).
    cache_hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)

    return Usage(
        input=inp,
        output=out,
        reasoning=reasoning,
        total=total,
        input_cache_hit=cache_hit,
        input_cache_miss=cache_miss,
    )


# ---------------------------------------------------------------------------
# Provider routing (mirrors decompose.py — kept local to avoid circular import)
# ---------------------------------------------------------------------------


def _provider_for(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openrouter"


def _openai_compat_client(provider: str) -> _openai.OpenAI:
    if provider == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        return _openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return _openai.OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def call_text(
    model: str,
    system: str,
    user: str,
    *,
    json_mode: bool,
    temperature: float,
    max_tokens: int = 16384,
    extra_body: dict[str, Any] | None = None,
    schema: type[BaseModel] | None = None,
) -> LLMCallResult:
    """Single-shot text generation; returns full ``LLMCallResult`` for logging.

    Wraps ``_generate_text_via_provider`` logic but keeps the response object
    so usage can be extracted and cost computed. ``text`` is stripped of
    leading/trailing whitespace.

    Args:
        model:        Provider model identifier.
        system:       System prompt string.
        user:         User message string.
        json_mode:    Request JSON-mode output from the provider.
        temperature:  Sampling temperature.
        max_tokens:   Max completion tokens (default 1024).
        extra_body:   Extra JSON body fields for OpenAI-compat calls (e.g. DeepSeek thinking).
        schema:       Optional Pydantic model for provider-side JSON schema
                      enforcement. When set:
                      - Gemini: passes ``response_schema=schema`` and forces
                        ``response_mime_type="application/json"``.
                      - OpenAI-compat: tries
                        ``response_format={"type":"json_schema","json_schema":{...},"strict":True}``;
                        on ``BadRequestError`` (e.g. DeepSeek which lacks
                        json_schema), caches the model in
                        ``_json_schema_unsupported`` and falls back to
                        ``{"type":"json_object"}``.
                      The caller is responsible for parsing the returned
                      ``text`` via ``schema.model_validate_json(text)``. We do
                      NOT retry on parse failure — the whole point is to skip
                      RAGAS-style retry storms.
    """
    provider = _provider_for(model)
    t0 = time.perf_counter()

    if provider == "gemini":
        config.validate_api_key()
        # Bound the HTTP timeout — google-genai SDK defaults are effectively
        # infinite under cold conditions. With concurrent calls and an
        # application-level retry layer, stuck calls would burn many minutes
        # silently. _LLM_CALL_TIMEOUT_S (default 120s) lets legitimate slow
        # calls finish (Pro synth takes 60-80s) while failing fast on real
        # hangs.
        gemini_client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options={"timeout": int(_LLM_CALL_TIMEOUT_S * 1000)},
        )
        # Schema mode forces JSON mime; otherwise honor json_mode flag.
        mime = "application/json" if (json_mode or schema is not None) else None
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "response_mime_type": mime,
            # Honor max_tokens on Gemini path too — google-genai's default is
            # tied to model context, but for our schema-output use case we've
            # seen 1k-token caps truncate JSON mid-array. Default 16k is plenty
            # for context_recall classifications + faithfulness verdicts on
            # 50+ statements.
            "max_output_tokens": max_tokens,
        }
        if schema is not None:
            cfg_kwargs["response_schema"] = schema
        response = gemini_client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=user)],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = (response.text or "").strip()
        usage = _usage_from_gemini(response, model)
    else:
        oai_client = _openai_compat_client(provider)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": _LLM_CALL_TIMEOUT_S,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        elif provider == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        # Pick response_format. Schema wins over json_mode.
        if schema is not None:
            response = _create_with_schema(oai_client, kwargs, model, schema)
        else:
            if json_mode:
                kwargs["messages"] = _ensure_json_keyword(kwargs["messages"])
                kwargs["response_format"] = {"type": "json_object"}
            response = oai_client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = (response.choices[0].message.content or "").strip()
        usage = _usage_from_openai_compat(response)

    cost = compute_cost_usd(model, usage)
    return LLMCallResult(
        text=text,
        latency_ms=latency_ms,
        usage=usage,
        raw_response=response,
        cost_usd=cost,
        model=model,
    )


def _ensure_json_keyword(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure the literal lowercase word ``json`` appears in the prompt.

    DeepSeek's `response_format={"type":"json_object"}` rejects requests
    with "Prompt must contain the word 'json' in some form". Our schema-
    output prompts say "JSON" (uppercase) which fails the lowercase check.
    Append a one-line directive to the last user message; this is a
    no-op for models that don't enforce this rule.
    """
    out: list[dict[str, str]] = [dict(m) for m in messages]
    # Find the last user message; append the directive to its content.
    for m in reversed(out):
        if m.get("role") == "user":
            content = m.get("content") or ""
            if "json" not in content.lower():
                m["content"] = content + "\n\nReturn the response as a JSON object (json)."
            break
    return out


def _create_with_schema(
    oai_client: _openai.OpenAI,
    base_kwargs: dict[str, Any],
    model: str,
    schema: type[BaseModel],
) -> Any:  # noqa: ANN401  — duck-typed SDK response
    """Issue a chat.completions.create with ``response_format`` derived from
    ``schema``, transparently falling back to ``json_object`` for providers
    that don't support strict ``json_schema``.

    Cached per-process via ``_json_schema_unsupported``. The call is never
    retried more than once; parse-time failures bubble up to the caller.
    """
    with _json_schema_unsupported_lock:
        already_unsupported = model in _json_schema_unsupported

    if already_unsupported:
        kwargs = dict(base_kwargs)
        kwargs["messages"] = _ensure_json_keyword(kwargs["messages"])
        kwargs["response_format"] = {"type": "json_object"}
        return oai_client.chat.completions.create(**kwargs)

    schema_kwargs = dict(base_kwargs)
    schema_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }
    try:
        return oai_client.chat.completions.create(**schema_kwargs)
    except _openai.BadRequestError:
        with _json_schema_unsupported_lock:
            _json_schema_unsupported.add(model)
        fallback_kwargs = dict(base_kwargs)
        fallback_kwargs["messages"] = _ensure_json_keyword(fallback_kwargs["messages"])
        fallback_kwargs["response_format"] = {"type": "json_object"}
        return oai_client.chat.completions.create(**fallback_kwargs)
