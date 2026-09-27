"""Anthropic-style Contextual Retrieval — generate 1-2 sentence chunk-context
summaries that get prepended to chunk text before embedding + BM25 indexing.

Per-notebook prompt customization via `prompts/contextualize/`:
- `base.md` — template with {audience} {glossary} {examples}
- `glossary_<domain>.md` — domain-specific terminology
- `examples_<domain>.md` — few-shot examples
- `audiences.py` — notebook → audience-string + domain mapping

For the SAMPLE pass we issue one generate call per chunk with the full document
inline, no caching. The bulk pipeline will swap in `client.caches.create(...)`
per source for the retrofit run.
"""

from __future__ import annotations

# pyright: reportMissingImports=false
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import openai as _openai
from google import genai
from google.genai.errors import APIError
from google.genai.types import (
    Content,
    CreateCachedContentConfig,
    GenerateContentConfig,
    Part,
    ThinkingConfig,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src import config


def _is_retryable_error(exc: BaseException) -> bool:
    # Gemini APIError
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code in (408, 429, 500, 502, 503, 504):
            return True
        status = (getattr(exc, "status", "") or "").upper()
        if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"):
            return True
    # OpenAI-compat path (DeepSeek, OpenRouter): RateLimitError, APIStatusError,
    # APIConnectionError, APITimeoutError, InternalServerError. The OpenAI SDK
    # exposes these as direct subclasses we can match on, plus status_code
    # attributes when underlying HTTP responses are available.
    try:
        from openai import (  # noqa: PLC0415
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        retry_types = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        if isinstance(exc, retry_types):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code in (408, 429, 500, 502, 503, 504):
                return True
    except ImportError:
        pass
    # Network-layer transients (DNS, conn reset, read timeout) bubble as plain
    # OSError / TimeoutError under httpx — retry these too.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return False


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "contextualize"
sys.path.insert(0, str(PROMPTS_DIR.parent.parent))
from prompts.contextualize.audiences import AudienceSpec, for_notebook  # noqa: E402

CONTEXTUALIZE_MODEL: str = os.environ.get("CONTEXTUALIZE_MODEL", "gemini-2.5-flash-lite")
# Gemini explicit-cache minimum (tokens). Below this caches.create fails;
# fall back to inline calls (cost penalty acceptable: short docs have few chunks).
CACHE_MIN_TOKENS: int = 1024
# 30-min TTL: covers occasional mega-docs that take >10min to fully contextualize
# (e.g. 1MB+ pdf.txt with 200+ chunks). Storage cost stays negligible.
CACHE_TTL_SECONDS: int = 1800

# Per-doc parallel chunk fan-out. Defaults sized for DeepSeek (no published
# RPM cap) — Gemini Tier 1 RPM=4K means each chunk worker @ ~30s/call burns
# 2 RPM, so 20 workers × 8 procs = 320 concurrent → under cap.
# Override via CHUNK_WORKERS env var; lower (10) if hitting Gemini quotas.
CHUNK_WORKERS: int = int(os.environ.get("CHUNK_WORKERS", "20"))
# Global cap on concurrent in-flight calls across the whole process.
# Bumped 30→60 for DeepSeek (no rate cap; bottleneck is wall latency, not RPM).
# Override via CTX_INFLIGHT env var.

_INFLIGHT = threading.Semaphore(int(os.environ.get("CTX_INFLIGHT", "60")))

# Cross-process pause coordination: when ANY worker hits a 429 with retryDelay,
# write the resume timestamp here. ALL workers (in this process or sibling procs)
# check before each Gemini call and sleep until the deadline. Lockless: we just
# read what's there; stale values are harmless (they're in the past → no sleep).
_PAUSE_FILE = Path(os.environ.get("CTX_PAUSE_FILE", "/tmp/rag_ctx_pause_until"))


def _maybe_wait_for_pause() -> None:
    """Block if another worker has signaled a quota cooldown."""
    try:
        until = float(_PAUSE_FILE.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return
    now = time.time()
    if until > now:
        # Cap at 60s sanity bound — a runaway value shouldn't freeze us forever.
        time.sleep(min(until - now, 60.0))


def _signal_pause(seconds: float) -> None:
    """Tell every worker to back off for `seconds`. Last writer wins."""
    if seconds <= 0:
        return
    until = time.time() + min(seconds, 120.0)  # clamp absurd suggestions
    try:
        _PAUSE_FILE.write_text(str(until))
    except OSError:
        pass


_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)(ms|s)")


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Pull a server-suggested wait time from the error. Returns seconds or None.

    Recognized formats:
    - Gemini APIError: error.details[].retryInfo.retryDelay (e.g. "57.4s", "604ms")
    - OpenAI-compat (DeepSeek/OpenRouter) APIStatusError / RateLimitError:
      Retry-After header on `exc.response`. Value is either seconds (integer)
      or HTTP-date — we only handle the seconds case (OpenAI/DeepSeek use seconds).
    """
    # Gemini-style retryDelay embedded in the error message
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        val = float(m.group(1))
        return val / 1000.0 if m.group(2) == "ms" else val
    # OpenAI-compat: pull Retry-After from the underlying httpx Response
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = None
        try:
            headers = getattr(response, "headers", None)
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            retry_after = None
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                return None
    return None


def _on_retry(retry_state: Any) -> None:  # noqa: ANN401
    """tenacity hook: when we retry, signal a cross-process pause if the API
    told us how long to wait (429/RESOURCE_EXHAUSTED includes retryDelay)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is None:
        return
    delay = _extract_retry_delay(exc)
    if delay is not None and delay > 0:
        _signal_pause(delay)


@lru_cache(maxsize=8)
def _read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def _build_system_prompt(spec: AudienceSpec) -> str:
    base = _read_prompt("base.md")
    glossary = _read_prompt(f"glossary_{spec['domain']}.md")
    examples = _read_prompt(f"examples_{spec['domain']}.md")
    return base.format(
        audience=spec["audience"],
        glossary=glossary.strip(),
        examples=examples.strip(),
    )


_CLIENT: genai.Client | None = None


def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        # Bump httpx connection pool to match parallel fan-out (4 files × 10
        # workers = up to 40 in-flight). Default 10 would queue silently and
        # cap throughput regardless of thread count.
        import httpx

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
        _CLIENT = genai.Client(
            api_key=config.GEMINI_API_KEY, http_options=http_options  # type: ignore[arg-type]
        )
    return _CLIENT


def _provider_for(model: str) -> str:
    """Return 'gemini', 'deepseek', or 'openrouter' based on model name prefix."""
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openrouter"


@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 5),
    before_sleep=_on_retry,
    reraise=True,
)
def _generate_openai_compat_ctx(model: str, user_prompt: str, system_prompt: str) -> str:
    """OpenAI-compat call for DeepSeek / OpenRouter contextualization (no caching).

    Wrapped with tenacity retry: 429 / 5xx / connection errors back off
    exponentially, and ``_on_retry`` parses ``Retry-After`` to coordinate
    cross-process pauses via ``/tmp/rag_ctx_pause_until``.
    """
    provider = _provider_for(model)
    if provider == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        client = _openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
    else:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        client = _openai.OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    _maybe_wait_for_pause()
    with _INFLIGHT:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
    return (response.choices[0].message.content or "").strip()


@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 5),
    before_sleep=_on_retry,
    reraise=True,
)
def _generate_deepseek_with_usage(
    model: str, user_prompt: str, system_prompt: str
) -> tuple[str, dict[str, Any]]:
    """DeepSeek call returning (content, full_usage_dict).

    DeepSeek auto-caches identical prefixes server-side (~50× discount per
    docs). When the same (system_prompt + doc) prefix repeats across chunk
    calls, the second+ requests get most of their input billed as cached.

    Returned usage dict has billing-relevant fields:
      cache_hit_tokens, cache_miss_tokens, completion_tokens, reasoning_tokens
    """
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    client = _openai.OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1",
    )
    _maybe_wait_for_pause()
    with _INFLIGHT:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
    content = (response.choices[0].message.content or "").strip()
    u = getattr(response, "usage", None)
    if u is None:
        return content, {"cache_hit_tokens": 0, "cache_miss_tokens": 0,
                         "completion_tokens": 0, "reasoning_tokens": 0}
    completion_details = getattr(u, "completion_tokens_details", None)
    return content, {
        "cache_hit_tokens": int(getattr(u, "prompt_cache_hit_tokens", 0) or 0),
        "cache_miss_tokens": int(getattr(u, "prompt_cache_miss_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        if completion_details else 0,
    }


def _contextualize_document_deepseek(
    doc_text: str,
    chunks: list[str],
    model: str,
    system_prompt: str,
) -> tuple[list[str], dict[str, Any]]:
    """DeepSeek-specific bulk contextualizer.

    Sends the full document inline as the user prompt prefix on every chunk
    call. DeepSeek's automatic prefix cache absorbs the (system_prompt + doc)
    portion — only the trailing chunk text varies between calls. The first
    call seeds the cache (full miss); subsequent calls report
    `prompt_cache_hit_tokens` for the cached prefix and bill at ~50× discount.

    Optimization: fire the first chunk synchronously to seed the server-side
    cache, then parallelize the rest. Otherwise concurrent first-wave calls
    race past the cache before any seeds it (verified empirically on 3-chunk
    smoke test: 0 cache hits with parallel-from-start).
    """
    totals = {
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }

    def _process_chunk(chunk_text: str) -> tuple[str, dict[str, Any]]:
        user_prompt = (
            f"<document>\n{doc_text}\n</document>\n\n"
            f"<chunk>\n{chunk_text}\n</chunk>"
        )
        return _generate_deepseek_with_usage(model, user_prompt, system_prompt)

    if not chunks:
        return [], {
            "used_cache": False,
            "fallback_used": False,
            "cached_input_tokens": 0,
            "n_chunks": 0,
            "estimated_usd_cost": 0.0,
        }

    # Seed the cache with one synchronous call.
    first_ctx, first_usage = _process_chunk(chunks[0])
    for k, v in first_usage.items():
        totals[k] += v

    contexts: list[str] = [first_ctx]
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            for ctx, usage in pool.map(_process_chunk, chunks[1:]):
                contexts.append(ctx)
                for k, v in usage.items():
                    totals[k] += v

    # DeepSeek V4 Flash pricing (USD per 1M tokens, May 2026):
    #   input cache miss: $0.14 / cache hit: $0.0028 / output: $0.28
    cost_usd = (
        totals["cache_miss_tokens"] * 0.14e-6
        + totals["cache_hit_tokens"] * 0.0028e-6
        + totals["completion_tokens"] * 0.28e-6
    )

    return contexts, {
        "used_cache": totals["cache_hit_tokens"] > 0,
        "fallback_used": False,
        "cached_input_tokens": totals["cache_hit_tokens"],
        "cache_miss_tokens": totals["cache_miss_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "reasoning_tokens": totals["reasoning_tokens"],
        "n_chunks": len(chunks),
        "estimated_usd_cost": round(cost_usd, 6),
    }


def contextualize_chunk(
    doc_text: str,
    chunk_text: str,
    notebook: str,
    *,
    model: str = CONTEXTUALIZE_MODEL,
) -> str:
    """Generate a chunk-context summary for `chunk_text` within `doc_text`.

    `notebook` selects audience + glossary + examples per partition.
    Returns the raw model output (caller validates length / content).
    """
    spec = for_notebook(notebook)
    system_prompt = _build_system_prompt(spec)

    user_prompt = (
        f"<document>\n{doc_text}\n</document>\n\n"
        f"<chunk>\n{chunk_text}\n</chunk>"
    )
    if _provider_for(model) == "gemini":
        response = _generate_with_retry(model, user_prompt, system_prompt)
        return (response.text or "").strip()
    return _generate_openai_compat_ctx(model, user_prompt, system_prompt)


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    stop=stop_after_attempt(15),
    reraise=True,
    before_sleep=_on_retry,
)
def _generate_with_retry(model: str, user_prompt: str, system_prompt: str) -> Any:  # noqa: ANN401
    _maybe_wait_for_pause()
    with _INFLIGHT:
        return _client().models.generate_content(
            model=model,
            contents=user_prompt,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                max_output_tokens=200,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )


def contextualize_document(
    doc_text: str,
    chunks: list[str],
    notebook: str,
    *,
    model: str = CONTEXTUALIZE_MODEL,
    doc_token_estimate: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Contextualize many chunks of one document with provider-specific caching.

    - Gemini: explicit `caches.create(...)` resource (90% input discount).
    - DeepSeek: automatic prefix caching server-side (~50× discount on
      cached prefix tokens — same system prompt + same document each call).
    - Other: inline, no caching.

    For Gemini, always deletes the cache in a finally block — orphaned caches
    accrue storage cost. DeepSeek caches are server-managed (TTL hours-days,
    no explicit lifecycle).

    Returns (contexts_in_chunk_order, stats_dict).
    """
    spec = for_notebook(notebook)
    system_prompt = _build_system_prompt(spec)
    provider = _provider_for(model)

    if provider == "deepseek":
        return _contextualize_document_deepseek(
            doc_text, chunks, model, system_prompt
        )

    use_cache = (
        provider == "gemini"
        and (doc_token_estimate or len(doc_text) // 4) >= CACHE_MIN_TOKENS
    )

    cache_name: str | None = None
    cached_input_tokens = 0
    fallback_used = False

    try:
        if use_cache:
            try:
                cache = _client().caches.create(
                    model=model,
                    config=CreateCachedContentConfig(
                        contents=[  # type: ignore[arg-type]
                            Content(
                                role="user",
                                parts=[Part(text=f"<document>\n{doc_text}\n</document>")],
                            )
                        ],
                        system_instruction=system_prompt,
                        ttl=f"{CACHE_TTL_SECONDS}s",
                    ),
                )
                cache_name = cache.name
            except APIError as e:
                # Below-minimum or transient — fall back to inline path.
                if getattr(e, "code", None) == 400:
                    fallback_used = True
                else:
                    raise

        # Capture cache_name for thread workers (immutable per call).
        cache_name_snapshot = cache_name

        def _process_chunk(chunk_text: str) -> tuple[str, int, bool]:
            """Returns (context, cached_tokens, used_fallback)."""
            if cache_name_snapshot is not None:
                try:
                    resp = _generate_with_cache(model, chunk_text, cache_name_snapshot)
                    meta = getattr(resp, "usage_metadata", None)
                    cached = (getattr(meta, "cached_content_token_count", 0) or 0) if meta else 0
                    return (resp.text or "").strip(), cached, False
                except APIError as e:
                    if getattr(e, "code", None) not in (403, 404):
                        raise
                    # fall through to inline for this chunk only
            resp = _generate_with_retry(
                model,
                f"<document>\n{doc_text}\n</document>\n\n<chunk>\n{chunk_text}\n</chunk>",
                system_prompt,
            )
            return (resp.text or "").strip(), 0, True

        contexts: list[str] = []
        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            for ctx, cached, used_fb in pool.map(_process_chunk, chunks):
                contexts.append(ctx)
                cached_input_tokens += cached
                if used_fb:
                    fallback_used = True

        return contexts, {
            "used_cache": cache_name is not None,
            "fallback_used": fallback_used,
            "cached_input_tokens": cached_input_tokens,
            "n_chunks": len(chunks),
        }

    finally:
        if cache_name is not None:
            try:
                _client().caches.delete(name=cache_name)
            except APIError:
                pass


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=4, max=120) + wait_random(0, 5),
    stop=stop_after_attempt(15),
    reraise=True,
    before_sleep=_on_retry,
)
def _generate_with_cache(model: str, chunk_text: str, cache_name: str) -> Any:  # noqa: ANN401
    _maybe_wait_for_pause()
    with _INFLIGHT:
        return _client().models.generate_content(
            model=model,
            contents=f"<chunk>\n{chunk_text}\n</chunk>",
            config=GenerateContentConfig(
                cached_content=cache_name,
                temperature=0.0,
                max_output_tokens=200,
                thinking_config=ThinkingConfig(thinking_budget=0),
            ),
        )
