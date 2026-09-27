"""Section → ProtoChunk conversion.

- Tables → atomic chunks (caption + markdown body), oversize tables row-banded.
- Text/list/code/equation → accumulate into parent buffer; close on overflow.
- Each parent → child split with section-conditional overlap.
- Equations are atomic (no bisection mid-formula).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import tiktoken

from .models import ProtoChunk, Section
from .profiles import (
    CHILD_MIN_TOKENS,
    MERGE_UNDER_TOKENS,
    SECTION_PROFILES,
    TEXT_FIELD_MAX_CHARS,
)
from .tables import (
    extract_chart_caption,
    extract_table_caption,
    html_table_to_markdown,
    split_table_md_by_rows,
)

logger = logging.getLogger(__name__)


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text or ""))


def _split_words_with_overlap(
    text: str,
    enc: tiktoken.Encoding,
    max_tok: int,
    overlap_tok: int,
    max_chars: int | None = None,
) -> list[str]:
    """Word-level splitter with token-bounded overlap.

    Closes the current chunk when EITHER the token budget OR the char budget
    (when ``max_chars`` is set) is exceeded. The char gate exists because
    pathological inputs — TOCs with dot-leaders, ASCII figures — can pack many
    chars into few tokens and overflow Milvus' VARCHAR(text) field.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    counts: list[int] = []
    cur_tok = 0
    cur_chars = 0
    for w in words:
        wt = _count_tokens(w, enc)
        added_chars = len(w) + (1 if current else 0)
        token_overflow = cur_tok + wt > max_tok
        char_overflow = max_chars is not None and cur_chars + added_chars > max_chars
        if (token_overflow or char_overflow) and current:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carry_tok = 0
            if overlap_tok > 0:
                for cw, ct in zip(reversed(current), reversed(counts)):
                    if carry_tok + ct > overlap_tok:
                        break
                    carry.insert(0, cw)
                    carry_tok += ct
            current = list(carry)
            counts = [_count_tokens(cw, enc) for cw in carry]
            cur_tok = carry_tok
            cur_chars = len(" ".join(carry)) if carry else 0
        current.append(w)
        counts.append(wt)
        cur_tok += wt
        cur_chars += len(w) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _element_to_text(elt: dict[str, Any]) -> str:
    """Render a non-table element back to plain text for the parent buffer."""
    return (elt.get("text") or "").strip()


def emit_table_chunks(
    elt: dict[str, Any],
    section: Section,
    parent_idx_start: int,
    enc: tiktoken.Encoding,
    *,
    summary: str | None = None,
) -> tuple[list[ProtoChunk], int]:
    """Emit one or more atomic chunks per table.

    When ``summary`` is provided (typically loaded from a sibling
    ``tables_summary.json``), emit small-to-big: one parent chunk holding the
    full caption + markdown table (for the sqlite parent store, retrieved by
    ``parent_chunk_id`` at generation time) plus one child chunk holding the
    short LLM summary (the embedding-friendly retrieval surrogate). Tables
    without a summary keep the legacy atomic-when-fits / row-banded path.
    """
    caption = extract_table_caption(elt)
    md = html_table_to_markdown(elt.get("table_body", ""))
    if not md.strip() or "|" not in md:
        return [], parent_idx_start

    page = elt.get("page_idx", section.page_idx)
    breadcrumb_prefix = f"[{section.breadcrumb}]\n\n"

    # --- Small-to-big path when a summary is available -----------------------
    if summary and summary.strip():
        s = summary.strip()
        child_text_chars = len(breadcrumb_prefix) + len(s)
        if child_text_chars > TEXT_FIELD_MAX_CHARS:
            logger.warning(
                "emit_table_chunks: oversize summary (%d chars) for table on page %d; "
                "falling back to row-banded path",
                child_text_chars,
                page,
            )
        else:
            parent_body = f"{caption}\n\n{md}".strip() if caption else md
            parent_chunks = [
                ProtoChunk(
                    parent_idx=parent_idx_start,
                    child_idx=None,
                    role="table",
                    breadcrumb=section.breadcrumb,
                    page_idx=page,
                    modality="table",
                    text=breadcrumb_prefix + parent_body,
                    raw_text=parent_body,
                    n_tokens=_count_tokens(parent_body, enc),
                ),
                ProtoChunk(
                    parent_idx=parent_idx_start,
                    child_idx=0,
                    role="table",
                    breadcrumb=section.breadcrumb,
                    page_idx=page,
                    modality="table",
                    text=breadcrumb_prefix + s,
                    raw_text=s,
                    n_tokens=_count_tokens(breadcrumb_prefix + s, enc),
                ),
            ]
            return parent_chunks, parent_idx_start + 1

    # Single-band path: full caption + body fits the budget.
    if caption and len(breadcrumb_prefix) + len(caption) + 2 + len(md) <= TEXT_FIELD_MAX_CHARS:
        body = f"{caption}\n\n{md}".strip()
        embed = breadcrumb_prefix + body
        return [
            ProtoChunk(
                parent_idx=parent_idx_start,
                child_idx=None,
                role="table",
                breadcrumb=section.breadcrumb,
                page_idx=page,
                modality="table",
                text=embed,
                raw_text=body,
                n_tokens=_count_tokens(embed, enc),
            )
        ], parent_idx_start + 1
    if not caption and len(breadcrumb_prefix) + len(md) <= TEXT_FIELD_MAX_CHARS:
        body = md
        embed = breadcrumb_prefix + body
        return [
            ProtoChunk(
                parent_idx=parent_idx_start,
                child_idx=None,
                role="table",
                breadcrumb=section.breadcrumb,
                page_idx=page,
                modality="table",
                text=embed,
                raw_text=body,
                n_tokens=_count_tokens(embed, enc),
            )
        ], parent_idx_start + 1

    # Multi-band path: caption only on band 1; bands 2..N use a short ref
    # marker. Budget for the table-rows portion of each band is sized for the
    # most constrained case (band 1) so all bands fit uniformly.
    short_caption_ref = (
        f"(continued — see {caption.split(':', 1)[0].strip()} above)"
        if caption else "(continued)"
    )
    band1_overhead = len(breadcrumb_prefix) + (len(caption) + 2 if caption else 0) + 16  # part i/N
    band_n_overhead = len(breadcrumb_prefix) + len(short_caption_ref) + 2 + 16
    band_budget = TEXT_FIELD_MAX_CHARS - max(band1_overhead, band_n_overhead)
    band_budget = max(band_budget, 200)  # floor: degraded is better than stuck

    bands = split_table_md_by_rows(md, band_budget)
    chunks: list[ProtoChunk] = []
    parent_idx = parent_idx_start
    n_bands = len(bands)
    for i, band in enumerate(bands):
        if i == 0:
            head = f"{caption} (part 1/{n_bands})" if caption else f"(part 1/{n_bands})"
        else:
            head = f"{short_caption_ref} (part {i+1}/{n_bands})"
        body = f"{head}\n\n{band}".strip()
        embed = breadcrumb_prefix + body
        chunks.append(
            ProtoChunk(
                parent_idx=parent_idx,
                child_idx=None,
                role="table",
                breadcrumb=section.breadcrumb,
                page_idx=page,
                modality="table",
                text=embed,
                raw_text=body,
                n_tokens=_count_tokens(embed, enc),
            )
        )
        parent_idx += 1
    return chunks, parent_idx


def _table_body_hash(elt: dict[str, Any]) -> str:
    """Mirror summarize._body_hash so chunker lookups match the summary index."""
    caption = extract_table_caption(elt)
    md = html_table_to_markdown(elt.get("table_body", ""))
    return hashlib.sha256(f"{caption}\n\n{md}".encode()).hexdigest()


def section_to_chunks(
    section: Section,
    parent_idx_start: int,
    enc: tiktoken.Encoding,
    *,
    table_summaries: dict[str, str] | None = None,
) -> tuple[list[ProtoChunk], int]:
    """Convert one section into chunks. Returns (chunks, next_parent_idx).

    ``table_summaries`` maps body_hash → LLM-generated summary; tables matched
    here go through the small-to-big path in ``emit_table_chunks``. Default
    ``None`` preserves the prototype-script call site.
    """
    profile = SECTION_PROFILES.get(section.role, SECTION_PROFILES["body"])
    chunks: list[ProtoChunk] = []
    parent_idx = parent_idx_start

    buffer_parts: list[str] = []
    buffer_pages: list[int] = []
    buffer_tok = 0

    breadcrumb_prefix = f"[{section.breadcrumb}]\n\n"
    # Char budget for the *raw* child text (post-prepend it must fit the field).
    child_char_budget = TEXT_FIELD_MAX_CHARS - len(breadcrumb_prefix)

    def flush_buffer() -> None:
        nonlocal parent_idx, buffer_parts, buffer_pages, buffer_tok
        if not buffer_parts:
            return
        parent_text = "\n\n".join(buffer_parts)
        first_page = buffer_pages[0] if buffer_pages else section.page_idx
        child_texts = _split_words_with_overlap(
            parent_text,
            enc,
            profile.child_tokens,
            profile.child_overlap,
            max_chars=child_char_budget,
        )
        # Child merge: fold trailing undersized child into previous child
        # if it fits within child_tokens + MERGE_UNDER_TOKENS.
        if len(child_texts) > 1:
            last_tok = _count_tokens(child_texts[-1], enc)
            if last_tok < MERGE_UNDER_TOKENS:
                prev_tok = _count_tokens(child_texts[-2], enc)
                if prev_tok + last_tok <= profile.child_tokens + MERGE_UNDER_TOKENS:
                    child_texts[-2] = child_texts[-2] + " " + child_texts[-1]
                    child_texts.pop()

        emitted = 0
        for ctext in child_texts:
            if _count_tokens(ctext, enc) < CHILD_MIN_TOKENS:
                continue
            embed = breadcrumb_prefix + ctext
            chunks.append(
                ProtoChunk(
                    parent_idx=parent_idx,
                    child_idx=emitted,
                    role=section.role,
                    breadcrumb=section.breadcrumb,
                    page_idx=first_page,
                    modality="text",
                    text=embed,
                    raw_text=ctext,
                    n_tokens=_count_tokens(embed, enc),
                )
            )
            emitted += 1
        if emitted:
            parent_idx += 1
        buffer_parts.clear()
        buffer_pages.clear()
        buffer_tok = 0

    for elt in section.elements:
        etype = elt.get("type", "")
        page = elt.get("page_idx", section.page_idx)

        if etype == "table":
            flush_buffer()
            summary = None
            if table_summaries:
                summary = table_summaries.get(_table_body_hash(elt))
            t_chunks, parent_idx = emit_table_chunks(
                elt, section, parent_idx, enc, summary=summary
            )
            chunks.extend(t_chunks)
            continue

        if etype == "chart":
            flush_buffer()
            caption = extract_chart_caption(elt)
            if caption:
                embed = breadcrumb_prefix + caption
                chunks.append(
                    ProtoChunk(
                        parent_idx=parent_idx,
                        child_idx=None,
                        role="table",
                        breadcrumb=section.breadcrumb,
                        page_idx=elt.get("page_idx", section.page_idx),
                        modality="table",
                        text=embed,
                        raw_text=caption,
                        n_tokens=_count_tokens(embed, enc),
                    )
                )
                parent_idx += 1
            continue

        text = _element_to_text(elt)
        if not text:
            continue
        t_tok = _count_tokens(text, enc)

        if etype == "equation":
            # Atomic — close current parent first if equation overflows budget.
            if buffer_tok + t_tok > profile.parent_tokens and buffer_parts:
                flush_buffer()
            buffer_parts.append(text)
            buffer_pages.append(page)
            buffer_tok += t_tok
            continue

        if buffer_tok + t_tok > profile.parent_tokens and buffer_parts:
            flush_buffer()
        buffer_parts.append(text)
        buffer_pages.append(page)
        buffer_tok += t_tok

    flush_buffer()
    return chunks, parent_idx
