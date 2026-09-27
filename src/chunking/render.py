"""Render content_list.json elements back to a flat markdown string.

Used by the staging pipeline to produce ``web.txt``-equivalent input for
the legacy markdown chunker. When ingest is later refactored to call
``chunk_sidecar()`` directly on the JSON, this rendering step disappears.
"""

from __future__ import annotations

from typing import Any

from .tables import html_table_to_markdown


def content_list_to_markdown(elements: list[dict[str, Any]]) -> str:
    """Render content_list.json elements back to a flat markdown string."""
    parts: list[str] = []

    for elt in elements:
        etype = elt.get("type", "")

        if etype == "text":
            level = elt.get("text_level")
            text = elt.get("text", "")
            if level and isinstance(level, int):
                parts.append(f"{'#' * level} {text}")
            else:
                parts.append(text)

        elif etype == "list":
            parts.append(elt.get("text", ""))

        elif etype == "code":
            text = elt.get("text", "")
            parts.append(f"```\n{text}\n```")

        elif etype == "table":
            table_html = elt.get("table_body", "")
            md_table = html_table_to_markdown(table_html)
            caption = elt.get("table_caption", "")
            if caption:
                parts.append(f"**{caption}**\n\n{md_table}")
            else:
                parts.append(md_table)

        elif etype == "equation":
            parts.append(elt.get("text", ""))

        else:
            # Unknown types: emit text if present
            text = elt.get("text", "")
            if text:
                parts.append(text)

    return "\n\n".join(p for p in parts if p)
