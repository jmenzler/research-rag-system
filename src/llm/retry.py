"""Centralised retry policy for transient LLM errors.

Single source of truth for what counts as "transient" across providers
(Gemini APIError, OpenAI/DeepSeek/OpenRouter, network) and the tenacity
decorator that wraps a callable with exponential backoff + jitter.

Used by ``src.llm.client.LLMClient`` for the normal generate path, and by
``src.evaluate`` to wrap the raw google-genai client that RAGAS' instructor
adapter calls directly (bypassing LLMClient).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.llm.messages import LLMError

# Number of distinct attempts (initial call + 7 retries). 8 attempts at
# 2-60s exponential backoff covers ~5 minutes of total wait — enough to
# ride out the kind of Gemini Flash 503 spike that killed the 2026-05-04
# N=10 RAGAS run.
RETRY_ATTEMPTS = 8
WAIT_MIN_SECONDS = 2
WAIT_MAX_SECONDS = 60
WAIT_JITTER_SECONDS = 5

F = TypeVar("F", bound=Callable[..., Any])


def is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying.

    Recognised:

    - ``LLMError`` with HTTP code in {408, 429, 500, 502, 503, 504} or a
      Google-style nested ``status`` of RESOURCE_EXHAUSTED / UNAVAILABLE /
      DEADLINE_EXCEEDED / INTERNAL.
    - OpenAI SDK transients (RateLimitError, APITimeoutError,
      APIConnectionError, InternalServerError) and ``APIStatusError`` with
      a retryable HTTP code.
    - ``google.genai.errors.APIError`` with the same retryable codes /
      statuses.
    - Bare ``TimeoutError`` / ``ConnectionError``.

    Non-retryable: 4xx other than 408/429, schema/validation errors, etc.
    """
    if isinstance(exc, LLMError):
        code = exc.code
        if code in (408, 429, 500, 502, 503, 504):
            return True
        orig = exc.original
        status = (getattr(orig, "status", "") or "").upper()
        if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"):
            return True

    try:
        from openai import (  # noqa: PLC0415
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        _retry_types = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        if isinstance(exc, _retry_types):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code in (408, 429, 500, 502, 503, 504):
                return True
    except ImportError:
        pass

    try:
        from google.genai.errors import APIError  # noqa: PLC0415

        if isinstance(exc, APIError):
            code = getattr(exc, "code", None)
            if code in (408, 429, 500, 502, 503, 504):
                return True
            status = (getattr(exc, "status", "") or "").upper()
            if status in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"):
                return True
    except ImportError:
        pass

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    return False


def transient_retry() -> Callable[[F], F]:
    """Return the canonical tenacity decorator for transient LLM errors.

    Configuration (shared across the codebase, change here only):
      - 8 attempts total
      - exponential backoff with min=2s, max=60s
      - random jitter 0-5s on top
      - re-raises the last exception on exhaustion

    Usage::

        @transient_retry()
        def call_llm() -> str:
            return client.models.generate_content(...).text
    """
    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=WAIT_MIN_SECONDS, max=WAIT_MAX_SECONDS)
        + wait_random(0, WAIT_JITTER_SECONDS),
        reraise=True,
    )
