"""Structural chunker for MinerU `content_list.json` sidecars.

The chunker exploits typed elements (headings, tables, equations) to build
hierarchical chunks bounded by section headings, with IMRaD-conditional sizing
and atomic table chunks.

Public API:

    from src.chunking import chunk_sidecar, ProtoChunk
    title, chunks = chunk_sidecar(Path("path/to/content_list.json"), enc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tiktoken

from .chunker import _count_tokens, _split_words_with_overlap, section_to_chunks
from .classifier import DROP_ROLES, classify_section_role, looks_like_toc
from .models import ProtoChunk, Section
from .profiles import MERGE_UNDER_TOKENS, SECTION_PROFILES, SectionProfile
from .render import content_list_to_markdown
from .tables import extract_table_caption
from .walker import load_sidecar, load_table_summaries, walk_into_sections

__all__ = [
    "DROP_ROLES",
    "MERGE_UNDER_TOKENS",
    "ProtoChunk",
    "SECTION_PROFILES",
    "Section",
    "SectionProfile",
    "_build_doc_prefix",
    "chunk_sidecar",
    "classify_section_role",
    "content_list_to_markdown",
    "extract_table_caption",
    "load_table_summaries",
]

# Maximum doc-prefix length before truncating. Per RAG literature, the
# combined prefix should stay under 50-100 tokens to avoid diluting the
# chunk's semantic signal. We cap title to 60 chars (~10-12 tokens), then
# allow short_cite + year on top — total ~20-30 tokens worst case.
_TITLE_MAX_CHARS = 60


def _build_doc_prefix(meta: dict[str, Any] | None) -> str:
    """Build a Tier-1 document-level prefix from `meta.json`.

    Format follows the root-to-leaf convention from the RAG literature:
    ``Title (Cite Year)`` — a single line, no surrounding brackets, ready to
    be joined with the section breadcrumb via `` → ``. When the title is a
    URL (web sources where parser fell back), substitutes ``metadata.host``.

    Field selection rules (presence-based, no quality gating):
      * ``title`` (always included; truncated when >60 chars)
      * ``short_cite`` (parenthetical; e.g. "Gougas 2020"), or ``year``
        alone if no short_cite

    Returns ``""`` when meta has no usable signal (caller emits
    breadcrumb-only as fallback).
    """
    if not meta:
        return ""

    md = meta.get("metadata") or {}
    raw_title = (meta.get("title") or "").strip()
    confidence = md.get("confidence") or "none"
    # Quality gate: only trust ``short_cite`` when the metadata extractor was
    # confident. ``medium`` confidence cites are often parser garbage like
    # ``"Strategies et al."`` from blog titles or ``"straightforward &
    # underlying"`` from picking random words, which would otherwise pollute
    # the embedding space. Year is always allowed (it's high-precision).
    short_cite = (md.get("short_cite") or "").strip() if confidence == "high" else ""
    year = md.get("year")
    host = (meta.get("host") or "").strip()

    # URL-titled web sources: title is the URL itself, swap to host.
    title: str
    if raw_title.startswith(("http://", "https://", "www.")):
        title = host or ""
    else:
        title = raw_title

    if not title:
        return ""

    # Truncate long titles on a word boundary.
    if len(title) > _TITLE_MAX_CHARS:
        words = title[: _TITLE_MAX_CHARS].rsplit(" ", 1)[0]
        title = f"{words}…" if words else title[: _TITLE_MAX_CHARS] + "…"

    if short_cite:
        return f"{title} ({short_cite})"
    if year:
        return f"{title} ({year})"
    return title


def _merge_short_parents(
    chunks: list[ProtoChunk],
    enc: tiktoken.Encoding,
    merge_under: int,
) -> list[ProtoChunk]:
    """Merge consecutive undersized text parents.

    Tables are never merged — they flush the buffer and pass through unchanged.
    """
    if not chunks:
        return []

    profile = SECTION_PROFILES["body"]
    result: list[ProtoChunk] = []
    buffer: list[ProtoChunk] = []
    buffer_tok = 0
    parent_idx = 0

    def _flush_buffer() -> None:
        nonlocal parent_idx
        if not buffer:
            return
        if len(buffer) == 1:
            c = buffer[0]
            # Re-number parent_idx
            result.append(ProtoChunk(
                parent_idx=parent_idx, child_idx=c.child_idx,
                role=c.role, breadcrumb=c.breadcrumb, page_idx=c.page_idx,
                modality=c.modality, text=c.text, raw_text=c.raw_text,
                n_tokens=c.n_tokens,
            ))
            parent_idx += 1
        else:
            merged_text = "\n\n".join(c.raw_text for c in buffer)
            merged_tok = _count_tokens(merged_text, enc)
            breadcrumb = buffer[0].breadcrumb
            page_idx = buffer[0].page_idx
            breadcrumb_prefix = f"[{breadcrumb}]\n\n"

            if merged_tok <= profile.parent_tokens + merge_under:
                # Emit as single parent, split into children
                child_texts = _split_words_with_overlap(
                    merged_text, enc, profile.child_tokens, profile.child_overlap,
                )
                for ci, ctext in enumerate(child_texts):
                    if _count_tokens(ctext, enc) < MERGE_UNDER_TOKENS and len(child_texts) > 1:
                        continue
                    embed = breadcrumb_prefix + ctext
                    result.append(ProtoChunk(
                        parent_idx=parent_idx, child_idx=ci,
                        role=buffer[0].role, breadcrumb=breadcrumb,
                        page_idx=page_idx, modality="text",
                        text=embed, raw_text=ctext,
                        n_tokens=_count_tokens(embed, enc),
                    ))
                parent_idx += 1
            else:
                # Oversize after merge — re-split into multiple parents
                child_texts = _split_words_with_overlap(
                    merged_text, enc, profile.parent_tokens, 0,
                )
                for parent_text in child_texts:
                    if not parent_text.strip():
                        continue
                    sub_children = _split_words_with_overlap(
                        parent_text, enc, profile.child_tokens, profile.child_overlap,
                    )
                    for ci, ctext in enumerate(sub_children):
                        embed = breadcrumb_prefix + ctext
                        result.append(ProtoChunk(
                            parent_idx=parent_idx, child_idx=ci,
                            role=buffer[0].role, breadcrumb=breadcrumb,
                            page_idx=page_idx, modality="text",
                            text=embed, raw_text=ctext,
                            n_tokens=_count_tokens(embed, enc),
                        ))
                    parent_idx += 1
        buffer.clear()
        nonlocal buffer_tok
        buffer_tok = 0

    # Track original-to-renumbered parent_idx for table chunks,
    # so parent+child pairs keep the same parent_idx.
    table_pidx_map: dict[int, int] = {}

    for c in chunks:
        if c.modality == "table":
            _flush_buffer()
            # Preserve parent+child pairing — remap original parent_idx
            # to the next available slot, reusing the mapping for children.
            orig = c.parent_idx
            if orig not in table_pidx_map:
                table_pidx_map[orig] = parent_idx
                parent_idx += 1
            result.append(ProtoChunk(
                parent_idx=table_pidx_map[orig], child_idx=c.child_idx,
                role=c.role, breadcrumb=c.breadcrumb, page_idx=c.page_idx,
                modality="table", text=c.text, raw_text=c.raw_text,
                n_tokens=c.n_tokens,
            ))
            continue

        c_tok = _count_tokens(c.raw_text, enc)
        if c_tok < merge_under:
            buffer.append(c)
            buffer_tok += c_tok
        else:
            _flush_buffer()
            result.append(ProtoChunk(
                parent_idx=parent_idx, child_idx=c.child_idx,
                role=c.role, breadcrumb=c.breadcrumb, page_idx=c.page_idx,
                modality=c.modality, text=c.text, raw_text=c.raw_text,
                n_tokens=c.n_tokens,
            ))
            parent_idx += 1

    _flush_buffer()
    return result


def _apply_doc_prefix(
    chunks: list[ProtoChunk], doc_prefix: str, enc: tiktoken.Encoding,
) -> list[ProtoChunk]:
    """Rewrite each chunk's `text` to thread the doc-prefix in front of the
    breadcrumb. The chunker writes ``text = "[{breadcrumb}]\\n\\n{raw_text}"``;
    we replace that with ``"{doc_prefix} → {breadcrumb}\\n\\n{raw_text}"``.

    When ``doc_prefix`` is empty, chunks are returned unchanged. Recomputes
    n_tokens to reflect the new text length.
    """
    if not doc_prefix:
        return chunks
    out: list[ProtoChunk] = []
    # The doc-prefix opens with the document's title (possibly truncated).
    # When MinerU emits a single L1 heading, the walker uses that title as
    # the breadcrumb for preamble content too — leading to a redundant
    # ``Title (Cite) → Title-as-breadcrumb`` sequence. Detect that overlap
    # by comparing on the title-prefix portion (everything up to ``" ("``,
    # which separates title from cite/year), accounting for ellipsis-truncation.
    doc_title_part = doc_prefix.split(" (", 1)[0].rstrip("…").rstrip()
    for c in chunks:
        # The chunker invariant: text starts with f"[{breadcrumb}]\n\n".
        old_prefix = f"[{c.breadcrumb}]\n\n"
        if c.text.startswith(old_prefix):
            body = c.text[len(old_prefix):]
            # Dedup when breadcrumb is (a substring of) the title.
            crumb = c.breadcrumb.strip()
            is_redundant = bool(doc_title_part) and (
                crumb == doc_title_part
                or crumb.startswith(doc_title_part)
                or doc_title_part.startswith(crumb)
            )
            if is_redundant:
                new_text = f"{doc_prefix}\n\n{body}"
            else:
                new_text = f"{doc_prefix} → {crumb}\n\n{body}"
        else:
            # Defensive: chunker invariant violated; fall back to wrapping.
            new_text = f"{doc_prefix}\n\n{c.text}"
        out.append(
            ProtoChunk(
                parent_idx=c.parent_idx,
                child_idx=c.child_idx,
                role=c.role,
                breadcrumb=c.breadcrumb,
                page_idx=c.page_idx,
                modality=c.modality,
                text=new_text,
                raw_text=c.raw_text,
                n_tokens=_count_tokens(new_text, enc),
            )
        )
    return out


def chunk_sidecar(
    path: Path,
    enc: tiktoken.Encoding,
    *,
    meta: dict[str, Any] | None = None,
) -> tuple[str, list[ProtoChunk]]:
    """Chunk a single content_list.json sidecar. Returns (title, chunks).

    If a sibling ``tables_summary.json`` exists, oversize tables flagged in
    that file are emitted as parent (full table) + child (LLM summary) pairs;
    tables not in the summary file fall back to the row-banded atomic path.

    When ``meta`` is provided (typically the parsed ``meta.json`` next to the
    sidecar), a Tier-1 document-level prefix is woven into each chunk's
    ``text`` field per the RAG literature's recommendation. The final shape:
    ``"{Title} ({Cite}) → {Breadcrumb}\\n\\n{body}"``. When meta is missing
    or unusable, chunks emit with breadcrumb only (legacy fallback).
    """
    elements = load_sidecar(path)
    title, sections = walk_into_sections(elements)
    table_summaries = load_table_summaries(path)
    kept = [s for s in sections if s.role not in DROP_ROLES]
    chunks: list[ProtoChunk] = []
    parent_idx = 0
    for sec in kept:
        if not sec.elements:
            continue
        # Structural ToC drop — runs after heading-role drop so missed
        # ``Outline``/foreign-language ToCs get caught by line-pattern.
        body_text = "\n".join(
            (e.get("text") or "").strip() for e in sec.elements
        )
        if looks_like_toc(body_text):
            continue
        sec_chunks, parent_idx = section_to_chunks(
            sec, parent_idx, enc, table_summaries=table_summaries
        )
        chunks.extend(sec_chunks)

    chunks = _merge_short_parents(chunks, enc, MERGE_UNDER_TOKENS)
    doc_prefix = _build_doc_prefix(meta)
    chunks = _apply_doc_prefix(chunks, doc_prefix, enc)
    return title, chunks
