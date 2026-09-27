"""Acceptance tests for tables_summary.json → chunker wiring.

These MUST fail before the wiring is implemented (TDD gate).

The contract under test: when a sibling ``tables_summary.json`` provides a
summary for a given table (matched by ``body_hash``), the chunker emits
small-to-big — one parent chunk holding the full table for sqlite lookup
plus one child chunk holding the embedding-friendly summary. Tables without
a summary keep the existing atomic-when-fits / row-banded fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
import tiktoken

from src.chunking.models import Section

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")

# Small table — fits a single chunk in the existing path.
_SMALL_CAPTION = "Table 1: Tiny composition table."
_SMALL_HTML = (
    "<table>"
    "<tr><td>Type</td><td>Count</td></tr>"
    "<tr><td>Person</td><td>10</td></tr>"
    "<tr><td>Org</td><td>5</td></tr>"
    "</table>"
)

# Large table — exceeds CHUNK_TEXT_BUDGET_CHARS so the legacy path row-bands it,
# and the wiring path replaces with summary+parent.
_BIG_CAPTION = "Table 2: Big detail table."


def _make_big_html(num_rows: int = 600) -> str:
    """Generate a wide+tall HTML table guaranteed to exceed any reasonable budget.

    600 rows × ~80 chars each ≈ 48 KB of body — comfortably above the
    Milvus VARCHAR field budget so the chunker is forced down the
    row-banded fallback (in absence of a summary).
    """
    parts = ["<table><tr><td>Symbol</td><td>Description (long)</td><td>Value</td></tr>"]
    for i in range(num_rows):
        parts.append(
            f"<tr><td>SYM_{i:04d}</td>"
            f"<td>Description-of-row-{i}-with-extra-padding-words</td>"
            f"<td>{i * 1.234:.4f}</td></tr>"
        )
    parts.append("</table>")
    return "".join(parts)


def _body_hash_for(caption: str, html: str) -> str:
    """Mirror the hashing the summarizer writes into tables_summary.json."""
    from src.chunking.tables import html_table_to_markdown

    md = html_table_to_markdown(html)
    return hashlib.sha256(f"{caption}\n\n{md}".encode()).hexdigest()


def _section_with(elt: dict) -> Section:
    return Section(role="results", breadcrumb="Paper > Section 4", page_idx=3, elements=[elt])


# ---------------------------------------------------------------------------
# extract_table_caption
# ---------------------------------------------------------------------------


def test_extract_table_caption_string() -> None:
    from src.chunking.tables import extract_table_caption

    assert extract_table_caption({"table_caption": "Table 1: Foo"}) == "Table 1: Foo"


def test_extract_table_caption_list_joins() -> None:
    from src.chunking.tables import extract_table_caption

    assert (
        extract_table_caption({"table_caption": ["Table 2:", "Composition", "by type"]})
        == "Table 2: Composition by type"
    )


def test_extract_table_caption_list_drops_falsy() -> None:
    from src.chunking.tables import extract_table_caption

    elt = {"table_caption": ["Table 3:", "", None, "Body"]}
    assert extract_table_caption(elt) == "Table 3: Body"


def test_extract_table_caption_falls_back_to_text() -> None:
    """No table_caption field — fall through to elt.text (matches summarizer logic)."""
    from src.chunking.tables import extract_table_caption

    elt = {"text": "Table 5 caption"}
    assert extract_table_caption(elt) == "Table 5 caption"


def test_extract_table_caption_empty_when_neither() -> None:
    from src.chunking.tables import extract_table_caption

    assert extract_table_caption({}) == ""
    assert extract_table_caption({"table_caption": ""}) == ""


def test_body_hash_caption_list_matches_string(tmp_path: Path) -> None:
    """Caption-as-list and caption-as-string with same final text must hash identically.

    This is the regression that prevents summarizer/chunker drift: the
    summarizer's caption resolution is the canonical one, and the chunker
    must reproduce it bit-for-bit before computing body_hash for lookup.
    """
    from src.chunking.tables import extract_table_caption, html_table_to_markdown

    html = _SMALL_HTML
    md = html_table_to_markdown(html)

    caption_str = "Table 1: Tiny composition table."
    caption_list_elt = {"table_caption": ["Table 1: Tiny", "composition table."]}
    caption_str_elt = {"table_caption": caption_str}

    h_list = hashlib.sha256(
        f"{extract_table_caption(caption_list_elt)}\n\n{md}".encode()
    ).hexdigest()
    h_str = hashlib.sha256(f"{extract_table_caption(caption_str_elt)}\n\n{md}".encode()).hexdigest()
    assert h_list == h_str


# ---------------------------------------------------------------------------
# load_table_summaries
# ---------------------------------------------------------------------------


def test_load_table_summaries_missing_file_returns_empty(tmp_path: Path) -> None:
    from src.chunking.walker import load_table_summaries

    sidecar = tmp_path / "content_list.json"
    sidecar.write_text("[]")
    assert load_table_summaries(sidecar) == {}


def test_load_table_summaries_valid_returns_dict(tmp_path: Path) -> None:
    from src.chunking.walker import load_table_summaries

    sidecar = tmp_path / "content_list.json"
    sidecar.write_text("[]")
    summary_path = tmp_path / "tables_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "page_idx": 3,
                        "caption": "Table 7",
                        "summary": "summary-A",
                        "body_hash": "abc123",
                    },
                    {
                        "page_idx": 5,
                        "caption": "Table 9",
                        "summary": "summary-B",
                        "body_hash": "def456",
                    },
                ]
            }
        )
    )
    assert load_table_summaries(sidecar) == {"abc123": "summary-A", "def456": "summary-B"}


def test_load_table_summaries_corrupt_file_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from src.chunking.walker import load_table_summaries

    sidecar = tmp_path / "content_list.json"
    sidecar.write_text("[]")
    (tmp_path / "tables_summary.json").write_text("not json {{{")
    with caplog.at_level(logging.WARNING):
        result = load_table_summaries(sidecar)
    assert result == {}


# ---------------------------------------------------------------------------
# emit_table_chunks regression (no summary path unchanged)
# ---------------------------------------------------------------------------


def test_emit_table_chunks_no_summary_small_table_atomic() -> None:
    """Small table without summary still produces 1 atomic chunk (regression)."""
    from src.chunking.chunker import emit_table_chunks

    elt = {
        "type": "table",
        "table_caption": _SMALL_CAPTION,
        "table_body": _SMALL_HTML,
        "page_idx": 3,
    }
    chunks, next_idx = emit_table_chunks(elt, _section_with(elt), 0, _ENC)
    assert next_idx == 1
    assert len(chunks) == 1
    assert chunks[0].child_idx is None
    assert chunks[0].modality == "table"
    assert _SMALL_CAPTION in chunks[0].raw_text


def test_emit_table_chunks_no_summary_big_table_row_banded() -> None:
    """Big table without summary still goes through row-banding (regression)."""
    from src.chunking.chunker import emit_table_chunks

    elt = {
        "type": "table",
        "table_caption": _BIG_CAPTION,
        "table_body": _make_big_html(),
        "page_idx": 5,
    }
    chunks, next_idx = emit_table_chunks(elt, _section_with(elt), 0, _ENC)
    assert len(chunks) >= 2  # row-banded into multiple parts
    assert next_idx == len(chunks)
    # Bands carry the "(part i/N)" marker
    assert any("part 1/" in c.raw_text for c in chunks)


# ---------------------------------------------------------------------------
# emit_table_chunks small-to-big when summary supplied
# ---------------------------------------------------------------------------


def test_emit_table_chunks_with_summary_emits_parent_plus_child() -> None:
    """Summary present → exactly 2 ProtoChunks: parent (full) + child (summary)."""
    from src.chunking.chunker import emit_table_chunks

    summary = "This table compares 600 trading symbols with their descriptions and decimal values."
    elt = {
        "type": "table",
        "table_caption": _BIG_CAPTION,
        "table_body": _make_big_html(),
        "page_idx": 5,
    }
    chunks, next_idx = emit_table_chunks(elt, _section_with(elt), 7, _ENC, summary=summary)
    assert next_idx == 8  # one parent_idx consumed
    assert len(chunks) == 2

    parent = next(c for c in chunks if c.child_idx is None)
    child = next(c for c in chunks if c.child_idx is not None)

    assert parent.parent_idx == 7
    assert child.parent_idx == 7
    assert child.child_idx == 0

    # Parent holds the full table; child holds the summary surrogate.
    assert _BIG_CAPTION in parent.raw_text
    assert "SYM_0001" in parent.raw_text  # row content present
    assert child.raw_text == summary
    assert "SYM_0001" not in child.text  # summary embed must NOT contain raw rows
    assert summary in child.text

    # Both share modality="table".
    assert parent.modality == "table"
    assert child.modality == "table"

    # Both carry the breadcrumb prefix.
    assert "[Paper > Section 4]" in parent.text
    assert "[Paper > Section 4]" in child.text


def test_emit_table_chunks_empty_summary_falls_through(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Empty/whitespace summary → fall back to existing path, no warning."""
    from src.chunking.chunker import emit_table_chunks

    elt = {
        "type": "table",
        "table_caption": _BIG_CAPTION,
        "table_body": _make_big_html(),
        "page_idx": 5,
    }
    chunks_with_empty, _ = emit_table_chunks(elt, _section_with(elt), 0, _ENC, summary="")
    chunks_baseline, _ = emit_table_chunks(elt, _section_with(elt), 0, _ENC)
    # Empty summary string is equivalent to no summary at all.
    assert len(chunks_with_empty) == len(chunks_baseline)
    assert all(c.child_idx is None for c in chunks_with_empty)


def test_emit_table_chunks_oversize_summary_falls_back_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pathologically long summary → warn, fall back to row-banded path."""
    from src.chunking.chunker import emit_table_chunks
    from src.config import CHUNK_TEXT_BUDGET_CHARS

    bad_summary = "X" * (CHUNK_TEXT_BUDGET_CHARS + 100)  # exceeds char budget
    elt = {
        "type": "table",
        "table_caption": _BIG_CAPTION,
        "table_body": _make_big_html(),
        "page_idx": 5,
    }
    with caplog.at_level(logging.WARNING):
        chunks, _ = emit_table_chunks(elt, _section_with(elt), 0, _ENC, summary=bad_summary)
    # Fell back to row-banding (multiple chunks, all parent-only).
    assert len(chunks) >= 2
    assert all(c.child_idx is None for c in chunks)
    assert any(
        "oversize" in r.message.lower() or "summary" in r.message.lower() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# chunk_sidecar end-to-end
# ---------------------------------------------------------------------------


def test_chunk_sidecar_threads_summaries_through(tmp_path: Path) -> None:
    """chunk_sidecar reads sibling tables_summary.json and emits parent+child for matched tables."""
    from src.chunking import chunk_sidecar

    big_html = _make_big_html()
    summary = "Compares 600 trading symbols with descriptions and decimal values."

    # Build a synthetic sidecar: title + section + small table + big table.
    elements = [
        {"type": "text", "text_level": 1, "text": "Test Paper", "page_idx": 0},
        {"type": "text", "text_level": 2, "text": "4 Results", "page_idx": 3},
        {"type": "text", "text": "Some prose introducing the results.", "page_idx": 3},
        {
            "type": "table",
            "table_caption": _SMALL_CAPTION,
            "table_body": _SMALL_HTML,
            "page_idx": 3,
        },
        {
            "type": "table",
            "table_caption": _BIG_CAPTION,
            "table_body": big_html,
            "page_idx": 5,
        },
    ]
    sidecar = tmp_path / "content_list.json"
    sidecar.write_text(json.dumps(elements))

    # Summary file targets ONLY the big table.
    big_hash = _body_hash_for(_BIG_CAPTION, big_html)
    (tmp_path / "tables_summary.json").write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "page_idx": 5,
                        "caption": _BIG_CAPTION,
                        "summary": summary,
                        "body_hash": big_hash,
                    }
                ]
            }
        )
    )

    title, chunks = chunk_sidecar(sidecar, _ENC)
    assert title == "Test Paper"

    table_chunks = [c for c in chunks if c.modality == "table"]
    # Expected:
    #   - small table → 1 atomic chunk (child_idx=None)
    #   - big table   → 1 parent + 1 child (paired by parent_idx)
    assert len(table_chunks) == 3

    # Group by parent_idx to verify pairing.
    by_parent: dict[int, list] = {}
    for c in table_chunks:
        by_parent.setdefault(c.parent_idx, []).append(c)

    pairs = [v for v in by_parent.values() if len(v) == 2]
    atomics = [v for v in by_parent.values() if len(v) == 1]
    assert len(pairs) == 1, "big table should yield one parent+child pair"
    assert len(atomics) == 1, "small table should remain atomic"

    # Pair contents check.
    pair = pairs[0]
    parent = next(c for c in pair if c.child_idx is None)
    child = next(c for c in pair if c.child_idx is not None)
    assert "SYM_0001" in parent.raw_text
    assert child.raw_text == summary

    # No row-band markers should appear for the summarized table.
    assert not any("part 1/" in c.raw_text for c in pair)


def test_chunk_sidecar_no_summary_file_unchanged_behavior(tmp_path: Path) -> None:
    """Without a summary sidecar, chunk_sidecar behavior is unchanged."""
    from src.chunking import chunk_sidecar

    big_html = _make_big_html()
    elements = [
        {"type": "text", "text_level": 1, "text": "Test Paper", "page_idx": 0},
        {"type": "text", "text_level": 2, "text": "4 Results", "page_idx": 3},
        {"type": "table", "table_caption": _BIG_CAPTION, "table_body": big_html, "page_idx": 5},
    ]
    sidecar = tmp_path / "content_list.json"
    sidecar.write_text(json.dumps(elements))
    # NO tables_summary.json sibling.

    _title, chunks = chunk_sidecar(sidecar, _ENC)
    table_chunks = [c for c in chunks if c.modality == "table"]
    # Big table → row-banded (multiple atomic chunks, all child_idx=None).
    assert len(table_chunks) >= 2
    assert all(c.child_idx is None for c in table_chunks)
    assert any("part 1/" in c.raw_text for c in table_chunks)
