"""Tests for citation parsing: end-to-end mapping of [N] in answers to Citation objects.

Why these tests exist
---------------------
Mapping inline ``[N]`` citations back to the right source has bitten us at
least three times:

1. ``_clean_source_name`` originally returned ``Path(...).name`` which collapses
   every post-flatten chunk to ``source.pdf`` / ``nlm.txt`` / ``pdf.txt``. The LLM
   would then write ``[1] source.pdf`` and the parser would happily store
   ``source.pdf`` as the citation key — but multiple chunks all share that leaf,
   so the citation pointed nowhere.

2. ``_parse_citations`` blindly trusted the LLM's Sources block (line 364:
   ``source_by_index[int(m.group(1))] = m.group(3).strip()``). When the LLM
   abbreviated or hallucinated, the parser threw away the prompt's authoritative
   ``index → slug`` map.

3. ``page_number`` was hard-coded to 0 in Pass 1 (lines 386, 390), so structured
   citations never carried the actual chunk page even though the LLM saw it via
   ``page="N"`` in the prompt.

The tests below pin each contract point individually. Each test names which
of the 7 contract requirements it covers (1-7) — losing any of them silently
is the failure mode that bit us.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import ChildChunk, ParentChunk, RetrievedChunk
from src.query.generate import (
    _cited_indices,
    _clean_source_name,
    _parse_citations,
    _render_sources_block,
    _slug_for,
    _strip_llm_sources_block,
)


def _retrieved(
    *,
    parent_id: str,
    source_file: str,
    page: int = 0,
    modality: str = "text",
    text: str = "x",
) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for citation tests."""
    parent = ParentChunk(
        id=parent_id,
        text=text,
        source_file=source_file,
        notebook="trading",
        modality=modality,
        page_number=page,
    )
    child = ChildChunk(
        id=f"{parent_id}_c0",
        parent_id=parent_id,
        text=text,
        source_file=source_file,
        notebook="trading",
        modality=modality,
        page_number=page,
    )
    return RetrievedChunk(
        child=child,
        parent=parent,
        dense_score=0.0,
        sparse_score=0.0,
        rerank_score=1.0,
    )


# ---------------------------------------------------------------------------
# Contract 1: every [N] in the answer maps to a unique source — leaf-name
# collisions across chunks are not allowed to confuse the citation key.
# ---------------------------------------------------------------------------


def test_clean_source_name_returns_slug_for_post_flatten_layout() -> None:
    """Post-flatten paths give back the slug directory, not the generic leaf."""
    assert (
        _clean_source_name(
            "sources/trading/flowhft_imitation_learning_via_flow_matching/source.pdf"
        )
        == "flowhft_imitation_learning_via_flow_matching"
    )
    assert (
        _clean_source_name("sources/trading/avellaneda_stoikov_market_making/nlm.txt")
        == "avellaneda_stoikov_market_making"
    )
    assert _clean_source_name("sources/notes/research_notes/pdf.txt") == "research_notes"


def test_clean_source_name_disambiguates_multiple_source_pdfs() -> None:
    """Two chunks from different papers must produce different keys.

    This is the bug that broke the original Avellaneda-Stoikov query — every
    chunk came back as ``source.pdf``, so [1] and [4] were indistinguishable.
    """
    a = _clean_source_name("sources/trading/paper_one/source.pdf")
    b = _clean_source_name("sources/trading/paper_two/source.pdf")
    assert a != b
    assert a == "paper_one"
    assert b == "paper_two"


def test_clean_source_name_non_canonical_returns_leaf() -> None:
    """For paths whose leaf is not source.pdf/pdf.txt/nlm.txt, return the leaf verbatim."""
    assert _clean_source_name("sources/x/some_paper.txt") == "some_paper.txt"
    assert _clean_source_name("ad_hoc/paper.pdf") == "paper.pdf"


# ---------------------------------------------------------------------------
# Contracts 2 + 3 + 4: structured citations carry the slug + page + modality of
# the cited chunk, regardless of how the LLM abbreviated the Sources block.
# ---------------------------------------------------------------------------


def test_citations_use_chunk_slug_not_llm_abbreviation() -> None:
    """LLM-emitted Sources lines do not override the prompt's index→slug map.

    The fix: when a Sources-block line names something that doesn't match any
    known chunk slug, the parser falls back to the index assigned at prompt-
    build time (i.e. ``retrieved[N-1]``).
    """
    retrieved = [
        _retrieved(parent_id="p1", source_file="sources/trading/flowhft_paper/source.pdf", page=11),
        _retrieved(parent_id="p2", source_file="sources/trading/ultra_low_latency/nlm.txt", page=0),
    ]
    answer = (
        "Under the AS model the spread is δ = γσ²τ + (1/γ)ln(1+γ/k) [1].\n"
        "The reservation price shifts with inventory [2].\n"
        "Sources:\n"
        "[1] source.pdf page 11\n"  # LLM abbreviation — must NOT become the key
        "[2] nlm.txt (some hallucinated description)\n"
    )
    cites = _parse_citations(answer, retrieved)
    keys = [c.source_file for c in cites]
    assert "flowhft_paper" in keys, f"expected flowhft_paper slug, got {keys}"
    assert "ultra_low_latency" in keys, f"expected ultra_low_latency slug, got {keys}"
    # Confirm neither LLM-abbreviated form leaked through.
    assert "source.pdf page 11" not in keys
    assert not any("hallucinated" in k for k in keys)


def test_citations_carry_chunk_page_number() -> None:
    """page_number on the structured Citation reflects the cited chunk's page.

    Was hard-coded to 0 in the numeric Pass 1 — silently broke every cite.
    """
    retrieved = [
        _retrieved(parent_id="p1", source_file="sources/trading/foo/source.pdf", page=11),
        _retrieved(parent_id="p2", source_file="sources/trading/bar/source.pdf", page=42),
    ]
    answer = "Foo [1]. Bar [2].\nSources:\n[1] foo\n[2] bar\n"
    cites = _parse_citations(answer, retrieved)
    by_slug = {c.source_file: c for c in cites}
    assert by_slug["foo"].page_number == 11
    assert by_slug["bar"].page_number == 42


def test_citations_carry_chunk_modality() -> None:
    """modality on the structured Citation reflects the cited chunk's modality."""
    retrieved = [
        _retrieved(
            parent_id="p1",
            source_file="sources/trading/diagram/source.pdf",
            modality="image",
        ),
        _retrieved(
            parent_id="p2",
            source_file="sources/trading/paper/source.pdf",
            modality="text",
        ),
    ]
    answer = "Diagram shows X [1]. Text says Y [2].\nSources:\n[1] diagram\n[2] paper\n"
    cites = _parse_citations(answer, retrieved)
    by_slug = {c.source_file: c for c in cites}
    assert by_slug["diagram"].modality == "image"
    assert by_slug["paper"].modality == "text"


# ---------------------------------------------------------------------------
# Contract 5: only chunks the LLM actually cited appear in citations[].
# ---------------------------------------------------------------------------


def test_uncited_chunks_do_not_leak_into_citations() -> None:
    """Five chunks retrieved, only [1] and [3] cited — citations has 2 entries."""
    retrieved = [
        _retrieved(parent_id=f"p{i}", source_file=f"sources/trading/paper{i}/source.pdf", page=i)
        for i in range(1, 6)
    ]
    answer = "First claim [1]. Third claim [3].\nSources:\n[1] paper1\n[3] paper3\n"
    cites = _parse_citations(answer, retrieved)
    assert len(cites) == 2
    keys = {c.source_file for c in cites}
    assert keys == {"paper1", "paper3"}


# ---------------------------------------------------------------------------
# Contract 6: duplicate references collapse.
# ---------------------------------------------------------------------------


def test_duplicate_inline_citations_dedup() -> None:
    """[1] cited 3 times → exactly one citation entry."""
    retrieved = [
        _retrieved(parent_id="p1", source_file="sources/trading/paper1/source.pdf", page=11),
    ]
    answer = "Claim A [1]. Claim B [1]. Claim C [1].\nSources:\n[1] paper1\n"
    cites = _parse_citations(answer, retrieved)
    assert len(cites) == 1
    assert cites[0].source_file == "paper1"
    assert cites[0].page_number == 11


def test_multi_index_block_dedups_with_other_blocks() -> None:
    """[1, 2] then [2, 3] → exactly 3 citations (paper1, paper2, paper3), order preserved."""
    retrieved = [
        _retrieved(parent_id=f"p{i}", source_file=f"sources/trading/paper{i}/source.pdf", page=i)
        for i in range(1, 4)
    ]
    answer = "Foo [1, 2]. Bar [2, 3].\nSources:\n[1] paper1\n[2] paper2\n[3] paper3\n"
    cites = _parse_citations(answer, retrieved)
    keys_in_order = [c.source_file for c in cites]
    assert keys_in_order == ["paper1", "paper2", "paper3"]


# ---------------------------------------------------------------------------
# Contract 7: LLM-invented Sources block doesn't poison the mapping.
#
# This is the strongest test — feeds the parser the literal answer-text we
# saw on the real Avellaneda-Stoikov run and asserts the structured output is
# correct despite the LLM emitting prose-like Sources lines.
# ---------------------------------------------------------------------------


def test_real_avellaneda_response_recovers_correct_citations() -> None:
    """End-to-end: feed the actual prose the LLM emitted; check structured citations.

    On the real run before the fix, the answer's Sources block looked like:
        [1] source.pdf page 11
        [2] nlm.txt (discussion of reservation price...)
        [3] nlm.txt (question about bid/ask bounds)
        [5] nlm.txt (spread calculation and quote placement)
    and citations[] came back with
        source_file='source.pdf page 11', page_number=0
    instead of
        source_file='flowhft_imitation_learning_via_flow_matching_policy_for_opti', page=11.
    """
    retrieved = [
        _retrieved(
            parent_id="p1",
            source_file="sources/trading/flowhft_imitation_learning_via_flow_matching_policy_for_opti/source.pdf",
            page=11,
        ),
        _retrieved(
            parent_id="p2",
            source_file="sources/trading/ultra_low_latency_high_frequency_market_making_a_comprehensi/nlm.txt",
            page=0,
        ),
        _retrieved(
            parent_id="p3",
            source_file="sources/trading/avellaneda_stoikov_market_making_model_quantitative_finance_/nlm.txt",
            page=0,
        ),
        _retrieved(
            parent_id="p4",
            source_file="sources/trading/optimal_high_frequency_market_making_stanford_university/source.pdf",
            page=2,
        ),
        _retrieved(
            parent_id="p5",
            source_file="sources/trading/ultra_low_latency_high_frequency_market_making_a_comprehensi/nlm.txt",
            page=0,
        ),
    ]
    answer = (
        "Under the Avellaneda–Stoikov model the optimal spread is "
        "δ = γσ²τ + (1/γ)ln(1+γ/k) [1].\n"
        "The reservation price shifts with inventory [2, 5].\n"
        "It comes from the HJB equation [1, 4].\n"
        "Sources:\n"
        "[1] source.pdf page 11\n"
        "[2] nlm.txt (discussion of reservation price and spread determination)\n"
        "[4] source.pdf\n"
        "[5] nlm.txt (spread calculation and quote placement)\n"
    )
    cites = _parse_citations(answer, retrieved)
    keys_in_order = [c.source_file for c in cites]
    # [1] then [2, 5] then [1, 4] → first-seen order preserved.
    # [1] cited twice → dedup to one.
    # [2] and [5] both point at chunks with slug=ultra_low_latency, page=0, modality=text
    # → dedup key (slug,page,mod) collapses to one entry. [4] is its own slug.
    assert keys_in_order == [
        "flowhft_imitation_learning_via_flow_matching_policy_for_opti",  # [1]
        "ultra_low_latency_high_frequency_market_making_a_comprehensi",  # [2]/[5] merged
        "optimal_high_frequency_market_making_stanford_university",  # [4]
    ]
    # Pages threaded through correctly.
    by_idx_first = next(c for c in cites if c.source_file.startswith("flowhft"))
    assert by_idx_first.page_number == 11
    by_idx_four = next(c for c in cites if c.source_file.startswith("optimal_high"))
    assert by_idx_four.page_number == 2


# ---------------------------------------------------------------------------
# Server-rendered Sources block — the trailing block is built by code from
# meta.json keyed by which [N]s appear in prose. The LLM is told NOT to emit
# one; if it does anyway we strip it before rendering ours.
# ---------------------------------------------------------------------------


def test_strip_llm_sources_block_removes_trailing_block() -> None:
    """A model that ignored the prompt and emitted Sources: gets stripped clean."""
    answer = (
        "The optimal spread is δ = γσ²τ/2 + ln(1+γ/k)/γ [1].\n"
        "The reservation price shifts with inventory [2].\n"
        "Sources:\n"
        "[1] some_paper\n"
        "[2] another_paper\n"
    )
    body = _strip_llm_sources_block(answer)
    assert body.endswith("inventory [2].")
    assert "Sources:" not in body
    assert "some_paper" not in body


def test_strip_llm_sources_block_handles_references_alias() -> None:
    """Some models emit `References:` instead of `Sources:`. Strip both."""
    answer = "Body cite [1].\nReferences:\n[1] X\n"
    body = _strip_llm_sources_block(answer)
    assert body == "Body cite [1]."


def test_strip_llm_sources_block_no_op_when_absent() -> None:
    """If the LLM behaved and didn't emit one, the body is unchanged."""
    answer = "Body cite [1] and [2, 3]."
    assert _strip_llm_sources_block(answer) == answer


def test_cited_indices_first_appearance_order_dedups() -> None:
    """Indices returned in first-appearance order, no duplicates."""
    body = "Foo [1]. Bar [3, 2]. Baz [1, 2]."
    assert _cited_indices(body) == [1, 3, 2]


def test_cited_indices_ignores_array_subscripts() -> None:
    """`array[3]` is not a citation — must be preceded by non-identifier char.

    Without this guard the parser would treat ``matrix[1, 2]`` and
    ``arr[3]`` in a math/code passage as citations and pull in the wrong
    chunks. The strict regex requires whitespace/punct/start before ``[``.
    """
    # `array[3]` should NOT be picked up; `[3]` after a space SHOULD be.
    body = "Initialize array[3] with zeros, then verify the result [3]."
    assert _cited_indices(body) == [3]
    # The array reference alone is not a citation.
    body_only_array = "Use array[3] to store the values."
    assert _cited_indices(body_only_array) == []


def test_render_sources_block_uses_short_cite_and_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When meta.json has both short_cite and title, render `[N] cite — "title" (p. P)`.

    Uses an in-tmp meta.json + monkeypatched _SOURCES_DIR so the test doesn't
    depend on the real corpus on disk.
    """
    import json as _json

    # Create one slug dir with full metadata.
    slug = "li2025_flowhft"
    slug_dir = tmp_path / "trading" / slug
    slug_dir.mkdir(parents=True)
    (slug_dir / "meta.json").write_text(
        _json.dumps(
            {
                "title": "FlowHFT: Imitation Learning via Flow Matching",
                "metadata": {"short_cite": "Li et al. 2025"},
            }
        )
    )
    import sys

    gen_mod = sys.modules["src.query.generate"]
    monkeypatch.setattr(gen_mod, "_SOURCES_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "_META_CACHE", {})

    retrieved = [
        _retrieved(
            parent_id="p1",
            source_file=f"sources/trading/{slug}/source.pdf",
            page=11,
        ),
    ]
    block = _render_sources_block([1], retrieved)
    assert block.startswith("Sources:")
    assert '[1] Li et al. 2025 — "FlowHFT: Imitation Learning via Flow Matching" (p. 11)' in block


def test_render_sources_block_falls_back_to_slug_when_meta_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No meta.json on disk → fall back to the slug as the human label.

    Page suffix still applies when the chunk has a real page number.
    """
    import sys

    gen_mod = sys.modules["src.query.generate"]
    monkeypatch.setattr(gen_mod, "_SOURCES_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "_META_CACHE", {})

    retrieved = [
        _retrieved(
            parent_id="p1",
            source_file="sources/trading/no_meta_paper/source.pdf",
            page=7,
        ),
    ]
    block = _render_sources_block([1], retrieved)
    assert block == "Sources:\n[1] no_meta_paper (p. 7)"


def test_render_sources_block_omits_page_for_zero_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """page=0 (NLM artifact) and modality=image both omit the page suffix."""
    import sys

    gen_mod = sys.modules["src.query.generate"]
    monkeypatch.setattr(gen_mod, "_SOURCES_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "_META_CACHE", {})

    retrieved = [
        _retrieved(
            parent_id="p1",
            source_file="sources/trading/nlm_only/nlm.txt",
            page=0,
        ),
        _retrieved(
            parent_id="p2",
            source_file="sources/trading/diagram/source.pdf",
            page=3,
            modality="image",
        ),
    ]
    block = _render_sources_block([1, 2], retrieved)
    assert "[1] nlm_only" in block
    # No page suffix on either entry.
    for line in block.splitlines()[1:]:
        assert "(p." not in line


def test_render_sources_block_handles_bad_index() -> None:
    """LLM cites [99] when only 5 chunks were retrieved — show, but flag fallback."""
    retrieved = [
        _retrieved(parent_id="p1", source_file="sources/trading/x/source.pdf"),
    ]
    block = _render_sources_block([1, 99], retrieved)
    assert "[99] (no document at this index)" in block


def test_render_sources_block_preserves_cite_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Block lines emitted in the same order indices appear in cited_indices."""
    import sys

    gen_mod = sys.modules["src.query.generate"]
    monkeypatch.setattr(gen_mod, "_SOURCES_DIR", tmp_path)
    monkeypatch.setattr(gen_mod, "_META_CACHE", {})

    retrieved = [
        _retrieved(parent_id=f"p{i}", source_file=f"sources/trading/paper{i}/source.pdf")
        for i in range(1, 4)
    ]
    block = _render_sources_block([3, 1], retrieved)
    lines = block.splitlines()
    assert lines[1].startswith("[3] paper3")
    assert lines[2].startswith("[1] paper1")


# ---------------------------------------------------------------------------
# _slug_for + _META_CACHE — the path-resolution layer that bit us at the
# layout flatten. Lock down its contract so future flattens don't regress.
# ---------------------------------------------------------------------------


def test_slug_for_post_flatten_returns_parent_dir() -> None:
    """The slug under post-flatten is the parent directory, not the leaf stem."""
    assert _slug_for("sources/trading/flowhft_paper/source.pdf") == "flowhft_paper"
    assert _slug_for("sources/trading/some_doc/nlm.txt") == "some_doc"
    assert _slug_for("sources/notes/notes/pdf.txt") == "notes"


def test_slug_for_non_canonical_returns_leaf_stem() -> None:
    """For non-canonical leaves, _slug_for returns the leaf stem verbatim."""
    assert _slug_for("sources/x/some_paper.txt") == "some_paper"
    assert _slug_for("ad_hoc/paper.pdf") == "paper"
