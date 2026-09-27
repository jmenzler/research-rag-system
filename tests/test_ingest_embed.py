"""Tests for ``src.ingest.embed`` — text/image embedding clients.

Boundaries pinned here:

  * ``_is_rate_limit_error`` — only Gemini ClientError with code=429 is a
    rate-limit; other ClientError codes / non-ClientError exceptions are not.
  * ``_embed_text_batch`` — provider dispatch on EMBEDDING_PROVIDER prefix.
    Wrong dispatch silently routes embedding traffic to the wrong endpoint.
  * ``_embed_text_batch_openrouter`` — happy path + the three foot-guns:
    out-of-order results (sort defensively by index), base64 string
    response (some providers ignore encoding_format, must fail loud),
    dim mismatch (silently storing wrong-dim vectors corrupts the index).

The OpenRouter HTTP client is mocked at the singleton-getter, so we
exercise the real response-parsing code path (no network).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embed import (
    _embed_text_batch,
    _embed_text_batch_openrouter,
    _is_rate_limit_error,
)

# ---------------------------------------------------------------------------
# _is_rate_limit_error — narrow predicate
# ---------------------------------------------------------------------------


def _make_client_error(code: int) -> Exception:
    """Construct a Gemini ClientError with a given code attribute.

    The real ClientError class needs structured args, so we stub it by
    instantiating a bare object whose .code reproduces the predicate's input.
    """
    from google.genai.errors import ClientError

    err = ClientError.__new__(ClientError)
    err.code = code
    return err


@pytest.mark.parametrize(
    "label,exc,expected",
    [
        ("rate_limit_429", _make_client_error(429), True),
        ("server_error_500", _make_client_error(500), False),
        ("bad_request_400", _make_client_error(400), False),
        ("not_a_client_error", RuntimeError("transport"), False),
        ("network_error", ConnectionError("dns"), False),
    ],
)
def test_is_rate_limit_error(label: str, exc: Exception, expected: bool) -> None:
    assert _is_rate_limit_error(exc) is expected, f"{label}: wrong predicate result"


# ---------------------------------------------------------------------------
# _embed_text_batch — provider dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,provider,expected_callee",
    [
        ("gemini_native", "gemini:gemini-embedding-2", "_embed_text_batch_gemini"),
        ("openrouter_qwen", "openrouter:qwen/qwen3-embedding-8b", "_embed_text_batch_openrouter"),
        (
            "openrouter_perplexity",
            "openrouter:perplexity/pplx-embed-v1-4b",
            "_embed_text_batch_openrouter",
        ),
    ],
)
def test_embed_text_batch_dispatches_on_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    provider: str,
    expected_callee: str,
) -> None:
    """Wrong dispatch silently routes traffic to the wrong endpoint."""
    monkeypatch.setattr("src.config.EMBEDDING_PROVIDER", provider)

    fake_client = MagicMock()  # genai.Client placeholder
    expected_vectors = [[1.0, 2.0]]

    with patch(f"src.ingest.embed.{expected_callee}", return_value=expected_vectors) as callee:
        result = _embed_text_batch(fake_client, ["text"])

    callee.assert_called_once()
    assert result == expected_vectors, f"{label}: dispatch returned wrong payload"


def test_embed_text_batch_unknown_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown provider prefix must fail loud, not silently no-op."""
    monkeypatch.setattr("src.config.EMBEDDING_PROVIDER", "azure:text-embedding-3-large")

    with pytest.raises(RuntimeError, match="Unhandled EMBEDDING_PROVIDER"):
        _embed_text_batch(MagicMock(), ["text"])


# ---------------------------------------------------------------------------
# _embed_text_batch_openrouter — happy path + 3 foot-guns
# ---------------------------------------------------------------------------


def _fake_openrouter_response(items: list[dict[str, Any]]) -> MagicMock:
    """Build a httpx.Response stub that returns {data: items} on .json()."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": items}
    return resp


@pytest.fixture
def fake_openrouter(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Patch the singleton getter so _embed_text_batch_openrouter sees a mock client.

    Yields the mock client so individual tests can configure post() return
    values per scenario.
    """
    fake_client = MagicMock()
    monkeypatch.setattr("src.ingest.embed._get_openrouter_client", lambda: fake_client)
    yield fake_client


@pytest.fixture
def openrouter_dim_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin EMBED_DIM=4096 to match the production Qwen3-8b setting."""
    monkeypatch.setattr("src.config.EMBED_DIM", 4096)
    monkeypatch.setattr("src.config.EMBED_MODEL", "qwen/qwen3-embedding-8b")


def test_openrouter_happy_path_returns_vectors_in_input_order(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Vectors come back in the same order as input texts."""
    fake_openrouter.post.return_value = _fake_openrouter_response(
        [
            {"index": 0, "embedding": [0.1] * 4096},
            {"index": 1, "embedding": [0.2] * 4096},
            {"index": 2, "embedding": [0.3] * 4096},
        ]
    )

    result = _embed_text_batch_openrouter(["a", "b", "c"])

    assert len(result) == 3
    assert result[0][0] == pytest.approx(0.1)
    assert result[1][0] == pytest.approx(0.2)
    assert result[2][0] == pytest.approx(0.3)


def test_openrouter_resorts_out_of_order_responses(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Provider returns data in arbitrary order — must sort by .index."""
    fake_openrouter.post.return_value = _fake_openrouter_response(
        [
            # Deliberately scrambled: index 2 first, then 0, then 1.
            {"index": 2, "embedding": [0.3] * 4096},
            {"index": 0, "embedding": [0.1] * 4096},
            {"index": 1, "embedding": [0.2] * 4096},
        ]
    )

    result = _embed_text_batch_openrouter(["a", "b", "c"])

    # Output must be in input order (a→0.1, b→0.2, c→0.3), not response order.
    assert result[0][0] == pytest.approx(0.1)
    assert result[1][0] == pytest.approx(0.2)
    assert result[2][0] == pytest.approx(0.3)


def test_openrouter_base64_response_fails_loud(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Some providers return base64 even when 'float' was requested.

    Silent acceptance would store junk floats. Must raise instead so we
    notice and switch providers / change encoding_format.
    """
    fake_openrouter.post.return_value = _fake_openrouter_response(
        [
            {"index": 0, "embedding": "AAAA=base64data="},
        ]
    )

    with pytest.raises(RuntimeError, match="base64"):
        _embed_text_batch_openrouter(["a"])


def test_openrouter_dim_mismatch_fails_loud(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Wrong dimensionality from provider would silently corrupt the index."""
    fake_openrouter.post.return_value = _fake_openrouter_response(
        [
            {"index": 0, "embedding": [0.0] * 768},  # wrong dim — config wants 4096
        ]
    )

    with pytest.raises(RuntimeError, match="dim="):
        _embed_text_batch_openrouter(["a"])


def test_openrouter_response_count_mismatch_fails_loud(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Provider returns fewer vectors than texts (truncated batch). Must raise."""
    fake_openrouter.post.return_value = _fake_openrouter_response(
        [
            {"index": 0, "embedding": [0.1] * 4096},
            # Missing index 1 — sent 2 texts, only 1 came back.
        ]
    )

    with pytest.raises(RuntimeError, match="2 inputs|2 embeddings"):
        _embed_text_batch_openrouter(["a", "b"])


def test_openrouter_missing_data_field_fails_loud(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """Malformed response (no 'data' list) must fail loud, not silently return []."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"error": {"message": "rate limit"}}  # no 'data'
    fake_openrouter.post.return_value = resp

    with pytest.raises(RuntimeError, match="missing 'data'"):
        _embed_text_batch_openrouter(["a"])


def test_openrouter_batches_at_50(
    fake_openrouter: MagicMock,
    openrouter_dim_4096: None,
) -> None:
    """120 texts → 3 calls (50 + 50 + 20)."""

    # Per-call response with the right number of items based on the request.
    def post_side_effect(_path: str, json: dict[str, Any]) -> MagicMock:
        n = len(json["input"])
        return _fake_openrouter_response(
            [{"index": i, "embedding": [0.0] * 4096} for i in range(n)]
        )

    fake_openrouter.post.side_effect = post_side_effect

    result = _embed_text_batch_openrouter(["x"] * 120)

    assert fake_openrouter.post.call_count == 3, (
        f"expected 3 batches (50+50+20), got {fake_openrouter.post.call_count}"
    )
    assert len(result) == 120
