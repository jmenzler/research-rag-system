"""Structured extraction: web HTML → MinerU-shape content_list.json elements.

Uses trafilatura's TEI-XML output as the intermediate representation, then
walks the XML tree to produce typed elements matching the MinerU content_list.json
schema consumed by the structural chunker.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

_CAPTION_RE = re.compile(r"^Table\s+\d", re.IGNORECASE)

# Structural TEI elements we recurse into without emitting
_CONTAINER_TAGS = frozenset({"text", "body", "div", "TEI", "teiHeader", "front", "back"})

# Tags whose children we walk but don't emit the parent
_PASSTHROUGH_TAGS = frozenset({"figure", "note", "ref"})


def _local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _flatten_text(elem: ET.Element) -> str:
    text = "".join(elem.itertext())
    return " ".join(text.split())


def _tei_table_to_html(table_elt: ET.Element) -> str:
    rows_html: list[str] = []
    for row in table_elt:
        if _local_tag(row) != "row":
            continue
        cells_html: list[str] = []
        for cell in row:
            if _local_tag(cell) not in ("cell",):
                continue
            text = _flatten_text(cell)
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cells_html.append(f"<td>{text}</td>")
        if cells_html:
            rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
    if not rows_html:
        return ""
    return f"<table>{''.join(rows_html)}</table>"


def _walk_tei(
    element: ET.Element,
    elements: list[dict[str, Any]],
    h1_seen: bool,
) -> bool:
    tag = _local_tag(element)

    if tag in _CONTAINER_TAGS:
        for child in element:
            h1_seen = _walk_tei(child, elements, h1_seen)
        return h1_seen

    if tag in _PASSTHROUGH_TAGS:
        return h1_seen

    if tag == "head":
        rend = element.get("rend", "")
        text = _flatten_text(element)
        if not text:
            return h1_seen
        if rend == "h1":
            level = 1 if not h1_seen else 2
            h1_seen = True
        elif rend == "h2":
            level = 2
        elif rend == "h3":
            level = 3
        else:
            elements.append({"type": "text", "text": text})
            return h1_seen
        elt: dict[str, Any] = {"type": "text", "text_level": level, "text": text}
        elements.append(elt)
        return h1_seen

    if tag == "p":
        text = _flatten_text(element)
        if text:
            elements.append({"type": "text", "text": text})
        return h1_seen

    if tag == "list":
        items: list[str] = []
        for child in element:
            if _local_tag(child) == "item":
                item_text = _flatten_text(child)
                if item_text:
                    items.append(f"- {item_text}")
        if items:
            elements.append({"type": "list", "text": "\n".join(items)})
        return h1_seen

    if tag == "code":
        text = _flatten_text(element)
        if text:
            elements.append({"type": "code", "text": text})
        return h1_seen

    if tag == "table":
        table_html = _tei_table_to_html(element)
        if table_html:
            elements.append({
                "type": "table",
                "table_body": table_html,
                "table_caption": "",
            })
        return h1_seen

    if tag == "quote":
        text = _flatten_text(element)
        if text:
            elements.append({"type": "text", "text": text})
        return h1_seen

    # Unknown tags — recurse into children
    for child in element:
        h1_seen = _walk_tei(child, elements, h1_seen)
    return h1_seen


def _resolve_captions(elements: list[dict[str, Any]]) -> None:
    for i in range(1, len(elements)):
        if elements[i]["type"] != "table":
            continue
        prev = elements[i - 1]
        if prev["type"] == "text" and _CAPTION_RE.match(prev.get("text", "")):
            elements[i]["table_caption"] = prev["text"]


def extract_structured(
    html: str,
    *,
    page_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Web HTML → MinerU-shape content_list.json elements, or None on failure.

    Internal flow:
    1. ``trafilatura.extract(..., output_format='xml')`` → TEI-XML
    2. Parse + walk depth-first, emitting typed elements
    3. Resolve table captions from preceding "Table N: …" paragraphs
    4. Synthesize title element from ``page_metadata`` if no L1 heading found
    """
    import trafilatura  # noqa: PLC0415

    xml_str = trafilatura.extract(
        html,
        output_format="xml",
        include_tables=True,
        include_formatting=True,
        include_comments=False,
    )
    if not xml_str or not xml_str.strip():
        return None

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    elements: list[dict[str, Any]] = []
    _walk_tei(root, elements, h1_seen=False)

    # If no L1 heading, synthesize from page_metadata
    if not any(e.get("text_level") == 1 for e in elements):
        title = (page_metadata or {}).get("title", "")
        title = title.strip() if isinstance(title, str) else ""
        if title:
            elements.insert(0, {
                "type": "text",
                "text_level": 1,
                "text": title,
                "page_idx": 0,
            })

    _resolve_captions(elements)

    if not elements:
        return None
    return elements
