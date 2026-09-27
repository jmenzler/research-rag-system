"""Unit tests for ``_build_doc_prefix`` and the meta-aware ``chunk_sidecar``.

The Tier-1 prefix is composed once per document from ``meta.json`` and woven
into every chunk's embed text. Format follows the RAG literature's root-to-leaf
recommendation: ``Title (Cite) → Breadcrumb\\n\\nbody``.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken

from src.chunking import _build_doc_prefix, chunk_sidecar

_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# _build_doc_prefix
# ---------------------------------------------------------------------------


def test_doc_prefix_full_metadata() -> None:
    """Title + short_cite produces ``Title (Cite)``."""
    meta = {
        "title": "Risk Factors Impact on the P&L",
        "metadata": {"short_cite": "Gougas 2020", "year": 2020, "confidence": "high"},
    }
    assert _build_doc_prefix(meta) == "Risk Factors Impact on the P&L (Gougas 2020)"


def test_doc_prefix_year_only_when_no_cite() -> None:
    """Title + year (no short_cite) falls back to ``Title (Year)``."""
    meta = {"title": "Foo Paper", "metadata": {"year": 2024}}
    assert _build_doc_prefix(meta) == "Foo Paper (2024)"


def test_doc_prefix_title_only_when_no_cite_or_year() -> None:
    """Bare title produces ``Title`` (no parenthesis)."""
    meta = {"title": "Foo Paper", "metadata": {}}
    assert _build_doc_prefix(meta) == "Foo Paper"


def test_doc_prefix_url_title_substitutes_host() -> None:
    """URL-titled web sources swap to ``host`` (96% population)."""
    meta = {
        "title": "https://medium.com/@kryptonlabs/vpin-the-coolest",
        "host": "medium.com",
        "metadata": {"short_cite": "Krypton 2024", "confidence": "high"},
    }
    assert _build_doc_prefix(meta) == "medium.com (Krypton 2024)"


def test_doc_prefix_drops_short_cite_when_confidence_below_high() -> None:
    """Medium/low/none confidence cites are parser garbage; reject them.

    Empirical examples from the corpus: ``"Strategies et al."`` (blog title
    parsed as authors), ``"straightforward & underlying"`` (random article
    words). Year falls through if present.
    """
    meta = {
        "title": "What Is Gamma Scalping?",
        "metadata": {
            "short_cite": "Strategies et al.",  # parser garbage
            "year": None,
            "confidence": "medium",
        },
    }
    assert _build_doc_prefix(meta) == "What Is Gamma Scalping?"

    # year alone still allowed when present (year extractor is high-precision)
    meta_with_year = {
        "title": "Foo",
        "metadata": {
            "short_cite": "Garbage et al.",
            "year": 2024,
            "confidence": "medium",
        },
    }
    assert _build_doc_prefix(meta_with_year) == "Foo (2024)"


def test_doc_prefix_returns_empty_when_unusable() -> None:
    """No title and no host => empty string (caller falls back)."""
    assert _build_doc_prefix(None) == ""
    assert _build_doc_prefix({}) == ""
    assert _build_doc_prefix({"title": "", "metadata": {}}) == ""
    assert _build_doc_prefix({"title": "https://x.com/foo", "host": ""}) == ""


def test_doc_prefix_truncates_long_title() -> None:
    """Titles >60 chars truncate at the last word boundary with ellipsis."""
    long_title = (
        "An Empirical Study of Variance Swap Markets and Their "
        "Convexity Properties During Crisis Periods"
    )
    meta = {"title": long_title, "metadata": {"short_cite": "Smith 2024", "confidence": "high"}}
    out = _build_doc_prefix(meta)
    assert out.endswith("(Smith 2024)")
    title_part = out[: -len(" (Smith 2024)")]
    assert len(title_part) <= 64  # 60 chars + " …"
    assert "…" in title_part
    assert " " not in title_part[-2:]  # ellipsis is on a word boundary


# ---------------------------------------------------------------------------
# chunk_sidecar end-to-end with meta
# ---------------------------------------------------------------------------


def _write_minimal_sidecar(path: Path) -> None:
    """Write a content_list.json long enough to survive merge thresholds."""
    body = (
        "Variance swaps offer direct exposure to realised volatility. "
        "They are quoted in volatility terms and pay the difference between "
        "realised and strike variance. The contract is convex in vol, "
        "creating positive vol-of-vol exposure. Replication uses a static "
        "portfolio of out-of-the-money options weighted by 1/K^2. Hedging "
        "the variance swap requires daily delta updates against the underlying. "
    ) * 8  # Repeat so the section has ~200+ tokens, exceeding MERGE_UNDER_TOKENS=150
    elements = [
        {"type": "text", "text": "Variance Swap Notes", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Introduction", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": body, "page_idx": 0},
    ]
    path.write_text(json.dumps(elements))


def test_chunk_sidecar_with_meta_threads_doc_prefix(tmp_path: Path) -> None:
    """When meta is supplied, every chunk's text begins with the doc-prefix."""
    sidecar = tmp_path / "content_list.json"
    _write_minimal_sidecar(sidecar)
    meta = {
        "title": "Variance Swap Mechanics",
        "metadata": {"short_cite": "Demeterfi 1999", "confidence": "high"},
    }
    _title, chunks = chunk_sidecar(sidecar, _ENC, meta=meta)
    assert chunks
    for c in chunks:
        # Root-to-leaf: doc-prefix → breadcrumb, no surrounding brackets
        assert c.text.startswith("Variance Swap Mechanics (Demeterfi 1999) → ")
        # Breadcrumb still present, bracket-free
        assert c.breadcrumb in c.text
        # Body still present
        assert c.raw_text in c.text


def test_chunk_sidecar_dedupes_when_breadcrumb_matches_title(tmp_path: Path) -> None:
    """When MinerU emits a single L1 heading, walker uses the title as
    BOTH doc title AND the preamble breadcrumb. The doc-prefix must skip
    the redundant breadcrumb segment to avoid embedding the title twice.
    """
    sidecar = tmp_path / "content_list.json"
    body = "Volatility analysis paragraph. " * 30
    elements = [
        {"type": "text", "text": "Variance Swap Notes", "text_level": 1, "page_idx": 0},
        # No L2 heading — the body lives in a preamble whose breadcrumb
        # the walker sets to the title.
        {"type": "text", "text": body, "page_idx": 0},
    ]
    sidecar.write_text(json.dumps(elements))
    meta = {
        "title": "Variance Swap Notes",
        "metadata": {"short_cite": "Demeterfi 1999", "confidence": "high"},
    }
    _title, chunks = chunk_sidecar(sidecar, _ENC, meta=meta)
    assert chunks
    # Doc-prefix is still there; breadcrumb segment is NOT (would be redundant)
    for c in chunks:
        assert c.text.startswith("Variance Swap Notes (Demeterfi 1999)")
        # Title should appear exactly once in the prefix line
        first_line = c.text.split("\n")[0]
        assert first_line.count("Variance Swap Notes") == 1
        # No `→ Variance Swap Notes` redundancy
        assert " → Variance Swap Notes" not in first_line


def test_chunk_sidecar_without_meta_preserves_legacy_shape(tmp_path: Path) -> None:
    """meta=None → chunks emit with the bracketed breadcrumb-only prefix
    (legacy shape preserved for backward compatibility)."""
    sidecar = tmp_path / "content_list.json"
    _write_minimal_sidecar(sidecar)
    _title, chunks = chunk_sidecar(sidecar, _ENC)
    assert chunks
    # Legacy: breadcrumb wrapped in brackets, no doc-prefix
    for c in chunks:
        assert c.text.startswith(f"[{c.breadcrumb}]\n\n")
