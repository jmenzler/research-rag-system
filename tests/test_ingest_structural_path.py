"""Unit tests for the ProtoChunk → ParentChunk + ChildChunk converter.

Pure functional tests — no Milvus, no embedding, no fs. Asserts the
contract between the structural chunker (`src.chunking`) and the production
ingest path (`src.ingest.chunker._proto_chunks_to_models`).
"""

from __future__ import annotations

from src.chunking.models import ProtoChunk
from src.ingest.chunker import _proto_chunks_to_models

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proto(
    *,
    parent_idx: int,
    child_idx: int | None,
    raw_text: str,
    text: str | None = None,
    modality: str = "text",
    breadcrumb: str = "Paper > Section",
    page_idx: int = 1,
    role: str = "body",
) -> ProtoChunk:
    """Compact factory for ProtoChunks in tests."""
    return ProtoChunk(
        parent_idx=parent_idx,
        child_idx=child_idx,
        role=role,
        breadcrumb=breadcrumb,
        page_idx=page_idx,
        modality=modality,
        text=text if text is not None else f"[{breadcrumb}]\n\n{raw_text}",
        raw_text=raw_text,
        n_tokens=len(raw_text.split()),
    )


_SOURCE = "sources/spread_capture/050__paper/ingested.md"
_NOTEBOOK = "spread_capture"
_DOC_MODALITY = "pdf"


# ---------------------------------------------------------------------------
# 1. Text-section grouping
# ---------------------------------------------------------------------------


def test_text_section_grouping() -> None:
    """3 ProtoChunks share parent_idx=0 with child_idx ∈ {0,1,2}.

    Expect 1 parent (raw_texts joined with '\\n\\n'), 3 children all linked
    to the same parent_id.
    """
    protos = [
        _proto(parent_idx=0, child_idx=0, raw_text="alpha"),
        _proto(parent_idx=0, child_idx=1, raw_text="beta"),
        _proto(parent_idx=0, child_idx=2, raw_text="gamma"),
    ]

    parents, children = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(parents) == 1
    assert len(children) == 3
    assert parents[0].text == "alpha\n\nbeta\n\ngamma"
    assert parents[0].modality == "text"
    assert parents[0].notebook == _NOTEBOOK
    assert parents[0].source_file == _SOURCE

    parent_id = parents[0].id
    for c in children:
        assert c.parent_id == parent_id
        assert c.source_file == _SOURCE
        assert c.notebook == _NOTEBOOK

    # Children carry the chunker's already-prepared embed text (with breadcrumb).
    assert children[0].text == protos[0].text
    assert children[1].text == protos[1].text
    assert children[2].text == protos[2].text


# ---------------------------------------------------------------------------
# 2. Summarized table preserves parent (full) + child (summary)
# ---------------------------------------------------------------------------


def test_summarized_table_preserved() -> None:
    """A table that came with a tables_summary.json entry yields 1 parent
    (full markdown table from the child_idx=None ProtoChunk) and 1 child
    (LLM summary from the child_idx=0 ProtoChunk).
    """
    full_table_md = "| Symbol | Value |\n|---|---|\n| SOL | 100 |"
    summary_text = "[Paper > Section]\n\nLLM-generated summary of the table."
    protos = [
        _proto(
            parent_idx=5,
            child_idx=None,
            raw_text=full_table_md,
            text=full_table_md,
            modality="table",
            page_idx=4,
        ),
        _proto(
            parent_idx=5,
            child_idx=0,
            raw_text="LLM-generated summary of the table.",
            text=summary_text,
            modality="table",
            page_idx=4,
        ),
    ]

    parents, children = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].text == full_table_md
    assert parents[0].modality == "table"
    assert parents[0].page_number == 4

    assert children[0].text == summary_text
    assert children[0].modality == "table"
    assert children[0].parent_id == parents[0].id
    assert children[0].page_number == 4


# ---------------------------------------------------------------------------
# 3. Atomic table — single ProtoChunk produces parent + synthetic child
# ---------------------------------------------------------------------------


def test_atomic_table_synthetic_child() -> None:
    """An atomic table (no summary) is one ProtoChunk with child_idx=None.

    Without a synthetic child, Milvus has no row to index. Helper must emit
    a child whose text equals the parent ProtoChunk's text.
    """
    table_md = "| A | B |\n|---|---|\n| 1 | 2 |"
    proto = _proto(
        parent_idx=2,
        child_idx=None,
        raw_text=table_md,
        text=f"[Paper > Section]\n\n{table_md}",
        modality="table",
        page_idx=7,
    )

    parents, children = _proto_chunks_to_models(
        [proto], source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].modality == "table"
    assert parents[0].text == table_md
    assert parents[0].page_number == 7
    assert children[0].modality == "table"
    assert children[0].text == proto.text
    assert children[0].parent_id == parents[0].id
    assert children[0].page_number == 7


# ---------------------------------------------------------------------------
# 4. Row-banded fallback — 3 separate parent_idx, each atomic
# ---------------------------------------------------------------------------


def test_row_banded_table() -> None:
    """A row-banded oversized table emits 3 ProtoChunks, each with
    distinct parent_idx and child_idx=None. Helper must produce 3 parents
    and 3 synthetic children, one pair per band.
    """
    protos = [
        _proto(parent_idx=10, child_idx=None, raw_text="band 1", modality="table"),
        _proto(parent_idx=11, child_idx=None, raw_text="band 2", modality="table"),
        _proto(parent_idx=12, child_idx=None, raw_text="band 3", modality="table"),
    ]

    parents, children = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(parents) == 3
    assert len(children) == 3

    # Each child must link to its corresponding parent.
    parent_ids = {p.id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids
        assert c.modality == "table"

    # Parent IDs are unique (deterministic chunk_id keyed off parent_idx).
    assert len({p.id for p in parents}) == 3


# ---------------------------------------------------------------------------
# 5. Determinism — same input → same IDs
# ---------------------------------------------------------------------------


def test_chunk_id_determinism() -> None:
    """Two identical calls must produce identical parent and child IDs."""
    protos = [
        _proto(parent_idx=0, child_idx=0, raw_text="alpha"),
        _proto(parent_idx=0, child_idx=1, raw_text="beta"),
        _proto(parent_idx=1, child_idx=None, raw_text="table", modality="table"),
    ]

    p1, c1 = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )
    p2, c2 = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert [p.id for p in p1] == [p.id for p in p2]
    assert [c.id for c in c1] == [c.id for c in c2]


# ---------------------------------------------------------------------------
# 6. Mixed paper — text + summarized tables + atomic table
# ---------------------------------------------------------------------------


def test_mixed_text_and_table_in_one_paper() -> None:
    """Realistic mix:

    * 5 text sections, each with 3 children (parent_idx 0..4)
    * 2 summarized tables, each with 1 None + 1 child_idx=0 (parent_idx 5, 6)
    * 1 atomic table (parent_idx 7, child_idx=None)

    Expect 8 parents and 5*3 + 2*1 + 1 = 18 children with intact linkage.
    """
    protos: list[ProtoChunk] = []

    # 5 text sections × 3 children each
    for p_idx in range(5):
        for c_idx in range(3):
            protos.append(
                _proto(
                    parent_idx=p_idx,
                    child_idx=c_idx,
                    raw_text=f"text-{p_idx}-{c_idx}",
                    page_idx=p_idx,
                )
            )

    # 2 summarized tables — parent_idx 5 and 6
    for p_idx in (5, 6):
        protos.append(
            _proto(
                parent_idx=p_idx,
                child_idx=None,
                raw_text=f"full-table-{p_idx}",
                modality="table",
                page_idx=p_idx,
            )
        )
        protos.append(
            _proto(
                parent_idx=p_idx,
                child_idx=0,
                raw_text=f"summary-{p_idx}",
                modality="table",
                page_idx=p_idx,
            )
        )

    # 1 atomic table
    protos.append(
        _proto(
            parent_idx=7,
            child_idx=None,
            raw_text="atomic-table",
            modality="table",
            page_idx=7,
        )
    )

    parents, children = _proto_chunks_to_models(
        protos, source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(parents) == 8
    assert len(children) == 5 * 3 + 2 * 1 + 1  # 18

    # Every child must reference an existing parent.
    parent_ids = {p.id for p in parents}
    for c in children:
        assert c.parent_id in parent_ids

    # Modality counts: 5 text parents, 3 table parents.
    assert sum(1 for p in parents if p.modality == "text") == 5
    assert sum(1 for p in parents if p.modality == "table") == 3


# ---------------------------------------------------------------------------
# 7. Milvus VARCHAR safety — children >4096 chars are truncated with ellipsis
# ---------------------------------------------------------------------------


def test_child_text_capped_at_milvus_limit() -> None:
    """Milvus VARCHAR field limit is 4096 chars. Children whose embed text
    exceeds this must be truncated to 4090 chars + ' …' so insert never
    fails. Parent (sqlite) is uncapped — full text preserved there.
    """
    long_text = "x" * 5000
    proto = _proto(
        parent_idx=0,
        child_idx=0,
        raw_text="parent body",
        text=long_text,
    )

    parents, children = _proto_chunks_to_models(
        [proto], source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )

    assert len(children) == 1
    assert len(children[0].text) <= 4096
    assert children[0].text.endswith(" …")
    # Parent unaffected by char cap (sqlite has no limit)
    assert parents[0].text == "parent body"


def test_child_text_under_limit_is_unchanged() -> None:
    """Children at or under 4096 chars must pass through verbatim."""
    text = "x" * 4096
    proto = _proto(parent_idx=0, child_idx=0, raw_text="p", text=text)
    _, children = _proto_chunks_to_models(
        [proto], source_file=_SOURCE, notebook=_NOTEBOOK, modality=_DOC_MODALITY
    )
    assert children[0].text == text  # exact pass-through, no ellipsis added
