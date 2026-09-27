"""Server-side citation validation pass (CHAT-05, D-09, CRIT-7).

Walks every [N] / [N, M] marker in the LLM answer; intersects with retrieved
chunks; emits ``resolved: false`` for markers that exceed the retrieved set.
Citations are pointers only — ``child_id`` + ``parent_id`` are references into
parents.sqlite, never copies of chunk text.

Reuses ``_NUMERIC_BLOCK_RE`` from src/query/generate.py:93 (read-only import)
to guarantee a single source of truth for the marker regex (with its
negative-lookbehind that skips word-internal brackets like ``array[3]``).
"""
from __future__ import annotations

from typing import TypedDict

from src.models import RetrievedChunk
from src.query.generate import _NUMERIC_BLOCK_RE  # noqa: PLC2701  # read-only import


class CitationDict(TypedDict):
    """Shape of a single validated citation entry.

    Matches the ``citations`` table column shape from 0002_chats.sql so the
    SSE handler in Plan 03 can pass these dicts straight to
    ``chats_store.attach_citations`` without remapping.
    """

    marker: int
    child_id: str | None
    parent_id: str | None
    paper_id: str | None
    score_dense: float | None
    score_sparse: float | None
    score_rerank: float | None
    resolved: bool


def validate_citations(
    answer: str,
    retrieved: list[RetrievedChunk],
) -> list[CitationDict]:
    """Build the ``citations[]`` SSE payload from the LLM answer + retrieved chunks.

    For every ``[N]`` or ``[N, M, ...]`` block in the answer:
      * If ``1 <= N <= len(retrieved)``: emit with scores and ``resolved=True``.
      * Else: emit with all pointer fields ``None`` and ``resolved=False`` —
        NEVER drop (CRIT-7 / D-09: hallucinated markers must surface as
        struck-through ``[?]`` pills in the UI, not silently disappear).

    Dedupes markers (first-occurrence wins); preserves answer-order; uses
    ``_NUMERIC_BLOCK_RE``'s negative-lookbehind to skip word-internal brackets
    like ``arr[3]``.
    """
    out: list[CitationDict] = []
    seen: set[int] = set()
    for block in _NUMERIC_BLOCK_RE.finditer(answer):
        for raw_tok in block.group(1).split(","):
            tok = raw_tok.strip()
            if not tok.isdigit():
                continue
            n = int(tok)
            if n in seen:
                continue
            seen.add(n)
            if 1 <= n <= len(retrieved):
                rc = retrieved[n - 1]
                out.append(CitationDict(
                    marker=n,
                    child_id=rc.child.id,
                    parent_id=rc.child.parent_id,
                    paper_id=rc.parent.source_file,
                    score_dense=rc.dense_score,
                    score_sparse=rc.sparse_score,
                    score_rerank=rc.rerank_score,
                    resolved=True,
                ))
            else:
                out.append(CitationDict(
                    marker=n,
                    child_id=None,
                    parent_id=None,
                    paper_id=None,
                    score_dense=None,
                    score_sparse=None,
                    score_rerank=None,
                    resolved=False,
                ))
    return out
