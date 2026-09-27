"""Walker: turn a flat content_list.json element list into structured `Section`s.

A Section is bounded by L2 heading boundaries — Unstructured.io's `by_title`
rule. Pre-heading content goes into a leading 'preamble' section. The walker
also filters element-type noise (page numbers, footnotes, recurring headers).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .classifier import classify_section_role
from .models import Section
from .tables import extract_chart_caption

logger = logging.getLogger(__name__)

# Element types that contribute substantive text to a section's body.
TEXT_TYPES = {"text", "list", "code", "equation"}

# Element types that are noise — never produce chunks, never split sections.
NOISE_TYPES = {"page_number", "page_footnote", "aside_text", "header", "image", "footer"}


def load_table_summaries(sidecar_path: Path) -> dict[str, str]:
    """Load the sibling ``tables_summary.json`` for a content_list sidecar.

    Returns ``{body_hash: summary_text}`` indexed for direct lookup by the
    chunker. Returns an empty dict if the file is missing or malformed —
    a corrupt summary file should never crash the chunker; it just falls
    back to the row-banded path.
    """
    summary_path = sidecar_path.parent / "tables_summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("load_table_summaries: failed to read %s: %s", summary_path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("load_table_summaries: %s top-level is not a dict", summary_path)
        return {}
    summaries = data.get("summaries", [])
    result: dict[str, str] = {}
    for entry in summaries:
        if not isinstance(entry, dict):
            continue
        bh = entry.get("body_hash")
        s = entry.get("summary")
        if isinstance(bh, str) and isinstance(s, str) and s:
            result[bh] = s
    return result


def load_sidecar(path: Path) -> list[dict[str, Any]]:
    """Load content_list.json. Defensive against double-encoded variants
    produced by an older parser bug (see commit ad5802b).
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, str):  # pre-fix double-encoded
        data = json.loads(data)
    if not isinstance(data, list):
        raise ValueError(f"expected list at top of {path}, got {type(data).__name__}")
    return data


def walk_into_sections(elements: list[dict[str, Any]]) -> tuple[str, list[Section]]:
    """Return (paper_title, sections).

    Title = first L1 heading (text_level=1), if any. Sections begin at each L2
    heading; pre-heading content goes into a leading 'body' section labeled
    'preamble'. The heading stack tracks breadcrumb path.
    """
    title = ""
    heading_stack: list[tuple[int, str]] = []   # (text_level, text)
    sections: list[Section] = []

    def _open_section(role: str, breadcrumb: str, page_idx: int) -> Section:
        s = Section(role=role, breadcrumb=breadcrumb, page_idx=page_idx)
        sections.append(s)
        return s

    current = _open_section("body", title or "(preamble)", page_idx=0)

    for elt in elements:
        etype = elt.get("type", "")
        text_level = elt.get("text_level")
        text = (elt.get("text") or "").strip()

        # Title (first L1 only — extra L1s in the corpus are usually section banners)
        if text_level == 1 and not title:
            title = text
            current.breadcrumb = title
            heading_stack = [(1, title)]
            continue

        # Section heading (L2) — close current, start new
        if text_level == 2:
            heading_stack = [h for h in heading_stack if h[0] < 2]
            heading_stack.append((2, text))
            role = classify_section_role(text)
            breadcrumb = " > ".join(t for _, t in heading_stack)
            current = _open_section(role, breadcrumb, page_idx=elt.get("page_idx", 0))
            continue

        # L3+ heading — track in breadcrumb but don't split sections at L3.
        if text_level == 3:
            heading_stack = [h for h in heading_stack if h[0] < 3]
            heading_stack.append((3, text))
            current.breadcrumb = " > ".join(t for _, t in heading_stack)
            continue

        if etype in NOISE_TYPES:
            continue

        if etype == "chart":
            if extract_chart_caption(elt):
                current.elements.append(elt)
            continue

        if etype in TEXT_TYPES or etype == "table":
            current.elements.append(elt)

    return title, sections
