"""Acceptance tests for src/chunking/summarize.py — table summarizer.

These MUST fail before the summarizer is implemented (TDD gate)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Sample tables for testing
# ---------------------------------------------------------------------------

_SAMPLE_CAPTION = "Table 1: Entity bank composition."

_SAMPLE_MD = """\
| Entity Type | Count | Percentage |
| --- | --- | --- |
| Person | 1,234 | 45.2% |
| Organization | 987 | 36.1% |
| Location | 512 | 18.7% |"""

_SAMPLE_BREADCRUMB = "Paper Title > Section 3 > Results"


def _make_cache_path() -> Path:
    """Return a temporary path for .tables_summary_cache.json."""
    return Path(tempfile.mkdtemp()) / ".tables_summary_cache.json"


# ---------------------------------------------------------------------------
# Functional tests (no API calls)
# ---------------------------------------------------------------------------


def test_summarize_table_function_exists() -> None:
    """summarize_table is importable from src.chunking.summarize."""
    from src.chunking.summarize import summarize_table

    assert callable(summarize_table)


def test_summarize_table_cache_hit_no_api_call() -> None:
    """When cache has a hit, no LLM call is made and cached value is returned."""
    from src.chunking.summarize import summarize_table

    cache_path = _make_cache_path()
    expected = "This is a cached summary."

    body_hash = hashlib.sha256((_SAMPLE_CAPTION + "\n\n" + _SAMPLE_MD).encode()).hexdigest()
    cache_path.write_text(json.dumps({body_hash: expected}))

    # Even with a broken client, cache hit should work without any API call.
    result = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        cache_path=cache_path,
        # No client — should hit cache before trying to build one.
    )
    assert result == expected


def test_summarize_table_llm_error_returns_empty() -> None:
    """On LLMError, summarize_table returns empty string (no crash)."""
    from src.chunking.summarize import summarize_table

    fake_client = mock.MagicMock()
    fake_client.generate.side_effect = __import__(
        "src.llm.messages", fromlist=["LLMError"]
    ).LLMError(provider="test", model="test", code=500, original=RuntimeError("fake"))

    result = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        client=fake_client,
    )
    assert result == ""


def test_summarize_table_cache_miss_writes_on_success() -> None:
    """After a successful summarization, the cache is updated."""
    from src.chunking.summarize import summarize_table
    from src.llm.messages import LLMResponse

    cache_path = _make_cache_path()
    assert not cache_path.exists()

    expected_summary = "Table 1: Entity bank composition lists entity types with counts."

    fake_client = mock.MagicMock()
    fake_client.generate.return_value = LLMResponse(
        text=expected_summary,
        input_tokens=100,
        output_tokens=30,
        model="test-model",
        latency_ms=50.0,
    )

    result = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        client=fake_client,
        cache_path=cache_path,
    )
    assert result == expected_summary

    # Cache file written
    assert cache_path.exists()
    cache = json.loads(cache_path.read_text())
    body_hash = hashlib.sha256((_SAMPLE_CAPTION + "\n\n" + _SAMPLE_MD).encode()).hexdigest()
    assert body_hash in cache
    assert cache[body_hash] == expected_summary


def test_summarize_table_calls_with_correct_params() -> None:
    """summarize_table passes the right prompt params to LLMClient.generate."""
    from src.chunking.summarize import summarize_table

    fake_client = mock.MagicMock()
    fake_client.generate.return_value = __import__(
        "src.llm.messages", fromlist=["LLMResponse"]
    ).LLMResponse(
        text="summary",
        input_tokens=10,
        output_tokens=5,
        model="x",
        latency_ms=1.0,
    )

    summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        client=fake_client,
    )

    call_kwargs = fake_client.generate.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_tokens"] == 400
    assert call_kwargs["json_mode"] is False

    # The prompt should contain the caption, breadcrumb, and markdown body.
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    content = messages[0].content
    assert _SAMPLE_CAPTION in content
    assert _SAMPLE_BREADCRUMB in content
    assert _SAMPLE_MD in content


# ---------------------------------------------------------------------------
# Smoke — real API (requires API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "not __import__('os').getenv('GEMINI_API_KEY')",
    reason="Requires GEMINI_API_KEY in .env",
)
def test_summarize_table_real_gemini() -> None:
    """Real summarization via Gemini produces a reasonable summary string."""
    from src.chunking.summarize import summarize_table

    result = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        model="gemini-2.5-flash",
    )
    assert isinstance(result, str)
    assert len(result) > 20  # should be a real paragraph


@pytest.mark.skipif(
    "not __import__('os').getenv('GEMINI_API_KEY')",
    reason="Requires GEMINI_API_KEY in .env",
)
def test_summarize_table_cache_replay() -> None:
    """Cache replay: second call with same inputs returns cached result."""
    from src.chunking.summarize import summarize_table

    cache_path = _make_cache_path()

    result1 = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        model="gemini-2.5-flash",
        cache_path=cache_path,
    )
    assert len(result1) > 20

    # Second call: should hit cache, return identically.
    result2 = summarize_table(
        caption=_SAMPLE_CAPTION,
        markdown_body=_SAMPLE_MD,
        breadcrumb=_SAMPLE_BREADCRUMB,
        model="gemini-2.5-flash",
        cache_path=cache_path,
    )
    assert result2 == result1

    # Verify cache was written
    cache = json.loads(cache_path.read_text())
    body_hash = hashlib.sha256((_SAMPLE_CAPTION + "\n\n" + _SAMPLE_MD).encode()).hexdigest()
    assert body_hash in cache
