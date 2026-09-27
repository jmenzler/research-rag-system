"""Tests for ``src.ingest.chunker`` — the pure-functional chunking layer.

Boundaries pinned here:

  * ``_split_into_parent_chunks`` — paragraph-then-sentence packing under a
    token budget.
  * ``_split_into_child_chunks`` — word-level packing with optional overlap.
  * ``_strip_tail_sections`` — two-tier (HARD markdown / SOFT bare-text)
    reference-trimming with a 20% soft floor.
  * ``_load_context_sidecar`` / ``_load_source_meta`` — defensive sidecar
    readers that must NEVER raise on missing/malformed input (the ingest
    pipeline degrades gracefully; raising would abort whole-doc ingest).
  * ``_build_hierarchical_chunks`` — non-structural fallback path.
  * ``_cap_child_text`` — Milvus VARCHAR(4096) byte cap.

All tests use the real ``cl100k_base`` tiktoken encoder via the shared
``enc`` fixture. Token budgets are small so the assertions remain readable
without losing the production behavior.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.ingest.chunker import (
    _build_hierarchical_chunks,
    _cap_child_text,
    _count_tokens,
    _get_encoder,
    _load_context_sidecar,
    _load_source_meta,
    _split_into_child_chunks,
    _split_into_parent_chunks,
    _strip_tail_sections,
)

if TYPE_CHECKING:
    import tiktoken


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def enc() -> tiktoken.Encoding:
    """Real cl100k_base encoder. Module-scoped — the encoder load is hot."""
    return _get_encoder()


@pytest.fixture
def tmp_source(tmp_path: Path) -> Iterator[Path]:
    """A throwaway source file path. Caller writes the body it wants."""
    yield tmp_path / "source.pdf"


# ---------------------------------------------------------------------------
# _split_into_parent_chunks
# ---------------------------------------------------------------------------


_PARENT_TOKENS = 80  # small budget keeps test text short and assertions tight


@pytest.mark.parametrize(
    "label,text,expected_n",
    [
        # Single short paragraph fits in one parent.
        ("single_short", "alpha beta gamma.", 1),
        # Two short paragraphs that together fit in one parent.
        ("two_fit_together", "alpha beta gamma.\n\ndelta epsilon zeta.", 1),
        # Two paragraphs each big enough to force a split between them.
        # Each ~100 tokens, budget 80 → must split.
        (
            "two_force_split",
            ("word " * 100).strip() + "\n\n" + ("token " * 100).strip(),
            2,
        ),
        # Empty input — degenerate but must not crash; falls back to [text].
        ("empty", "", 1),
    ],
)
def test_split_into_parent_chunks_count(
    enc: tiktoken.Encoding,
    label: str,
    text: str,
    expected_n: int,
) -> None:
    chunks = _split_into_parent_chunks(text, enc, _PARENT_TOKENS)
    assert len(chunks) == expected_n, f"{label}: expected {expected_n} got {len(chunks)}"


def test_split_into_parent_chunks_paragraph_too_large_falls_back_to_sentences(
    enc: tiktoken.Encoding,
) -> None:
    """A single paragraph that exceeds the budget triggers sentence-level split.

    No `\\n\\n` in the text → only ONE paragraph from the first split. If the
    sentence-fallback is broken (e.g. removed), the function would return that
    one giant paragraph as a single chunk, blowing the budget by ~20×.
    """
    sentence = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    # ~16 sentences × ~16 tok ≈ 250 tok. Single paragraph, no \n\n.
    big_para = ". ".join([sentence] * 16) + "."
    assert _count_tokens(big_para, enc) > _PARENT_TOKENS * 2, "test text not big enough"

    chunks = _split_into_parent_chunks(big_para, enc, _PARENT_TOKENS)

    # Must produce more than one chunk (sentence-fallback was triggered).
    assert len(chunks) >= 2, (
        f"expected sentence-fallback to split big para into >=2 chunks, got {len(chunks)}"
    )
    # The biggest chunk must NOT be wildly over budget. The packer can briefly
    # exceed by one sentence (it appends-then-flushes), so 1.5× is the
    # operational ceiling.
    for c in chunks:
        assert _count_tokens(c, enc) <= _PARENT_TOKENS * 1.5, (
            f"chunk overflow: {_count_tokens(c, enc)} tokens vs budget {_PARENT_TOKENS}"
        )


# ---------------------------------------------------------------------------
# _split_into_child_chunks
# ---------------------------------------------------------------------------


_CHILD_TOKENS = 50


@pytest.mark.parametrize(
    "label,word_count,expected_min_chunks",
    [
        ("under_budget", 20, 1),  # 20 words ≪ 50 tok → one chunk
        ("at_budget", 50, 1),  # roughly fits in one chunk (whitespace tokens vary)
        ("over_budget", 200, 2),  # forces at least one split
        ("way_over", 1000, 5),  # bulk text → many chunks
    ],
)
def test_split_into_child_chunks_count(
    enc: tiktoken.Encoding,
    label: str,
    word_count: int,
    expected_min_chunks: int,
) -> None:
    text = " ".join(["word"] * word_count)
    chunks = _split_into_child_chunks(text, enc, _CHILD_TOKENS)
    assert len(chunks) >= expected_min_chunks, (
        f"{label}: got {len(chunks)} chunks, expected >= {expected_min_chunks}"
    )


def test_split_into_child_chunks_overlap_carries_words_across_boundary(
    enc: tiktoken.Encoding,
) -> None:
    """With overlap_tokens > 0, each successive chunk shares words with its predecessor.

    The carry-over semantics prepend the previous chunk's tail to the next
    chunk's head — so the head of chunk N must contain words that appeared
    near the tail of chunk N-1. Compare against the no-overlap baseline:
    same input, different chunk[1] head, with the difference being words
    that came from chunk[0]'s tail.
    """
    text = " ".join(f"w{i:03d}" for i in range(60))
    no_overlap = _split_into_child_chunks(text, enc, _CHILD_TOKENS, overlap_tokens=0)
    with_overlap = _split_into_child_chunks(text, enc, _CHILD_TOKENS, overlap_tokens=15)

    assert len(no_overlap) >= 2 and len(with_overlap) >= 2, (
        "need at least 2 chunks to observe overlap"
    )
    # First chunk identical — no carry-in possible.
    assert with_overlap[0] == no_overlap[0]

    # Chunk 1's head with overlap must start EARLIER in the sequence than
    # chunk 1's head without overlap. That earlier start IS the carryover.
    no_overlap_first_word = no_overlap[1].split()[0]
    with_overlap_first_word = with_overlap[1].split()[0]
    assert with_overlap_first_word < no_overlap_first_word, (
        f"with_overlap chunk[1] starts at {with_overlap_first_word!r}, "
        f"no_overlap starts at {no_overlap_first_word!r} — "
        "carry-over should make with_overlap start EARLIER"
    )

    # The carried words must come from chunk[0]'s tail.
    chunk0_words = set(no_overlap[0].split())
    overlap_head_words = set(with_overlap[1].split()[:5])
    shared = chunk0_words & overlap_head_words
    assert shared, "no shared words between chunk[0] and overlap chunk[1] head"


def test_split_into_child_chunks_empty_returns_singleton(enc: tiktoken.Encoding) -> None:
    """Defensive: empty input must not yield zero chunks."""
    assert _split_into_child_chunks("", enc, _CHILD_TOKENS) == [""]


# ---------------------------------------------------------------------------
# _strip_tail_sections
# ---------------------------------------------------------------------------


_BODY = "Body paragraph one. " * 20  # ~400 chars of body so soft-floor matters


@pytest.mark.parametrize(
    "label,text,should_strip,marker_in_dropped",
    [
        # HARD markdown heading — stripped regardless of position.
        ("hard_top", "## References\nfoo bar baz", True, "References"),
        ("hard_after_body", _BODY + "\n## References\ncite 1\ncite 2", True, "References"),
        # SOFT bare heading — stripped only when past 20% of doc.
        (
            "soft_late",
            _BODY + "\nReferences\ncite 1\ncite 2",
            True,
            "References",
        ),
        (
            "soft_too_early",
            "References\n" + _BODY,  # heading at 0% — must NOT strip
            False,
            None,
        ),
        # No tail at all.
        ("no_heading", _BODY, False, None),
        # Hard markdown form for "Acknowledgments".
        ("hard_acks", _BODY + "\n### Acknowledgments\nthanks", True, "Acknowledgments"),
    ],
)
def test_strip_tail_sections(
    label: str,
    text: str,
    should_strip: bool,
    marker_in_dropped: str | None,
) -> None:
    kept, dropped = _strip_tail_sections(text)
    if should_strip:
        assert dropped is not None, f"{label}: expected a tail to be stripped"
        assert marker_in_dropped is not None
        assert marker_in_dropped.lower() in dropped.lower(), (
            f"{label}: dropped section ({dropped!r}) doesn't contain {marker_in_dropped!r}"
        )
        # Body must be preserved.
        assert "Body paragraph" in kept or kept == "" or "foo bar" in kept, (
            f"{label}: kept text lost the body"
        )
    else:
        assert dropped is None, f"{label}: expected no strip, got {dropped!r}"
        assert kept == text, f"{label}: kept != input even though no strip"


def test_strip_tail_sections_empty_input_safe() -> None:
    """Degenerate input must not raise."""
    kept, dropped = _strip_tail_sections("")
    assert kept == ""
    assert dropped is None


# ---------------------------------------------------------------------------
# _load_context_sidecar — defensive reader, must never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,sidecar_body,expected",
    [
        ("missing_file", None, {}),
        ("empty_chunks_array", '{"chunks": []}', {}),
        ("malformed_json", "{not valid json", {}),
        (
            "valid_two_entries",
            json.dumps(
                {
                    "chunks": [
                        {"parent_idx": 0, "child_idx": 0, "context": "A"},
                        {"parent_idx": 1, "child_idx": 2, "context": "B"},
                    ]
                }
            ),
            {(0, 0): "A", (1, 2): "B"},
        ),
    ],
)
def test_load_context_sidecar(
    tmp_source: Path,
    label: str,
    sidecar_body: str | None,
    expected: dict[tuple[int, int], str],
) -> None:
    if sidecar_body is not None:
        # Sidecar lives at <source.pdf>.ctx.json — the helper appends to suffix.
        (tmp_source.parent / (tmp_source.name + ".ctx.json")).write_text(sidecar_body)
    result = _load_context_sidecar(str(tmp_source))
    assert result == expected, f"{label}: got {result!r}"


# ---------------------------------------------------------------------------
# _load_source_meta — defensive reader, must never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,meta_body,expected_keys",
    [
        ("missing_file", None, None),
        ("empty_dict", "{}", set()),
        ("malformed_json", "{not valid", None),
        # Per-spec: top-level must be a dict; arrays/strings return None.
        ("not_a_dict", '["a", "b"]', None),
        (
            "valid_with_metadata_block",
            json.dumps({"metadata": {"short_cite": "Foo 2024"}}),
            {"metadata"},
        ),
    ],
)
def test_load_source_meta(
    tmp_path: Path,
    label: str,
    meta_body: str | None,
    expected_keys: set[str] | None,
) -> None:
    if meta_body is not None:
        (tmp_path / "meta.json").write_text(meta_body)
    result = _load_source_meta(tmp_path)
    if expected_keys is None:
        assert result is None, f"{label}: expected None, got {result!r}"
    else:
        assert result is not None, f"{label}: expected dict, got None"
        assert set(result.keys()) >= expected_keys, f"{label}: missing keys"


# ---------------------------------------------------------------------------
# _build_hierarchical_chunks — non-structural fallback path
# ---------------------------------------------------------------------------


def test_build_hierarchical_chunks_emits_parents_and_children(
    enc: tiktoken.Encoding,
) -> None:
    """Smoke test: real text → real ParentChunk + ChildChunk structures.

    We don't assert exact counts (depend on PARENT_TOKENS / CHILD_TOKENS config),
    just that the contract holds: at least one parent, every child references
    a parent, every chunk has the right metadata threading through.
    """
    text = "\n\n".join([("word " * 50).strip() for _ in range(5)])
    parents, children = _build_hierarchical_chunks(
        text=text,
        source_file="/tmp/test/source.pdf",
        notebook="trading",
        modality="text",
        page_number=0,
        enc=enc,
    )
    assert len(parents) >= 1
    assert len(children) >= 1
    parent_ids = {p.id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids, "orphan child"
        assert c.notebook == "trading"
        assert c.modality == "text"
        assert c.source_file == "/tmp/test/source.pdf"


def test_build_hierarchical_chunks_threads_ctx_sidecar_when_present(
    tmp_source: Path,
    enc: tiktoken.Encoding,
) -> None:
    """When a .ctx.json sidecar exists, child embed text gets the context prefix."""
    text = "\n\n".join([("word " * 50).strip() for _ in range(3)])
    sidecar = tmp_source.parent / (tmp_source.name + ".ctx.json")
    sidecar.write_text(
        json.dumps(
            {
                "chunks": [
                    {"parent_idx": 0, "child_idx": 0, "context": "CTX-MARKER-X"},
                ]
            }
        )
    )
    _, children = _build_hierarchical_chunks(
        text=text,
        source_file=str(tmp_source),
        notebook="trading",
        modality="text",
        page_number=0,
        enc=enc,
    )
    # At least the (0, 0) child got the marker; non-matching children get the
    # plain `[doc_id]\n\n<text>` form.
    assert any("CTX-MARKER-X" in c.text for c in children), "ctx-prefix not threaded into any child"


# ---------------------------------------------------------------------------
# _cap_child_text — Milvus VARCHAR(4096) byte cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,text_len,should_truncate",
    [
        ("under_limit", 1000, False),
        ("at_limit", 4096, False),  # exactly at cap is kept
        ("just_over", 4097, True),
        ("way_over", 50000, True),
    ],
)
def test_cap_child_text(label: str, text_len: int, should_truncate: bool) -> None:
    text = "x" * text_len
    capped = _cap_child_text(text, "test-id")
    if should_truncate:
        assert len(capped) <= 4096, f"{label}: cap broken ({len(capped)} > 4096)"
        assert capped.endswith(" …"), f"{label}: missing ellipsis marker"
    else:
        assert capped == text, f"{label}: untruncated text was modified"
