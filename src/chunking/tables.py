"""Convert table HTML to Markdown, splitting oversized tables into row bands."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any


def extract_table_caption(elt: dict[str, Any]) -> str:
    """Resolve a MinerU table element's caption to a single string.

    MinerU emits ``table_caption`` as either str or list[str]; some extracts
    omit the field entirely and put the caption text in ``elt['text']``. The
    summarizer's ``body_hash`` is computed against the resolved string, so
    the chunker MUST mirror this resolution exactly to produce matching keys.
    """
    raw = elt.get("table_caption")
    if isinstance(raw, list):
        joined = " ".join(str(c) for c in raw if c).strip()
        if joined:
            return joined
    elif isinstance(raw, str):
        if raw.strip():
            return raw.strip()
    return (elt.get("text") or "").strip()


def extract_chart_caption(elt: dict[str, Any]) -> str:
    """Resolve a MinerU chart element's caption to a single string.

    MinerU emits ``chart_caption`` as list[str] or str. Charts with multi-word
    captions (e.g. "Figure 3: Channelized 2D landscapes. (A-C) ...") get kept
    as caption-only chunks; subfigure labels like "A", "B" are dropped by the
    caller via a 30-char threshold.

    Returns empty string when the caption is missing or trivial.
    """
    raw = elt.get("chart_caption")
    if isinstance(raw, list):
        joined = " ".join(str(c) for c in raw if c).strip()
        if len(joined) > 30:
            return joined
    elif isinstance(raw, str):
        if len(raw.strip()) > 30:
            return raw.strip()
    return ""


class _TableMDExtractor(HTMLParser):
    """Convert a flat <table><tr><td> into a GFM markdown table.

    Assumes simple structure produced by MinerU TableFormer (no nesting,
    no thead/tbody, td/th cells only). Multi-line cells are joined with spaces.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell_buf: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell_buf = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            cell = " ".join("".join(self._cell_buf).split()).strip()
            self._row.append(cell)
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buf.append(data)


def html_table_to_markdown(html: str) -> str:
    """Convert MinerU table HTML to GFM markdown. Falls back to raw on parse error."""
    if not html or "<table" not in html.lower():
        return html or ""
    p = _TableMDExtractor()
    try:
        p.feed(html)
    except Exception:
        return html
    rows = [r for r in p.rows if any(c for c in r)]
    if not rows:
        return html
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _band_with_header(header: str, sep: str, rows: list[str]) -> str:
    """Join rows with the header (and separator if present)."""
    if sep:
        return "\n".join([header, sep, *rows])
    return "\n".join([header, *rows])


def _split_oversize_row(
    row: str, header: str, sep: str, max_chars: int
) -> list[str]:
    """Split a single markdown row that exceeds ``max_chars`` into multiple
    bands by chopping the cell text on word boundaries.

    Each band still carries header + sep. The split is purely defensive — we
    sacrifice cell-row integrity in exchange for fitting the storage budget.
    """
    base_overhead = len(header) + len(sep) + 2
    row_budget = max(200, max_chars - base_overhead)
    bands: list[str] = []
    words = row.split(" ")
    cur: list[str] = []
    cur_len = 0
    for w in words:
        added = len(w) + (1 if cur else 0)
        if cur and cur_len + added > row_budget:
            bands.append(_band_with_header(header, sep, [" ".join(cur)]))
            cur = []
            cur_len = 0
        cur.append(w)
        cur_len += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        bands.append(_band_with_header(header, sep, [" ".join(cur)]))
    return bands


def split_table_md_by_rows(md: str, max_chars: int) -> list[str]:
    """Split a markdown table into row-banded sub-tables that each fit ``max_chars``.

    Keeps header + separator on every band so each piece stands alone. When a
    single row by itself exceeds ``max_chars`` (e.g. a regulatory description
    cell with a paragraph of prose), the row is further bisected on word
    boundaries — the alternative is silently overflowing the storage field.
    Returns a single-element list if the table already fits.
    """
    lines = md.splitlines()
    if len(lines) < 2:
        return [md]
    if lines[1].strip().startswith("| ---"):
        header, sep, *rows = lines
    else:
        header, sep, rows = lines[0], "", lines[1:]
    parts: list[str] = []
    current_rows: list[str] = []
    base_overhead = len(header) + len(sep) + 2  # newlines after header[+sep]
    current_len = base_overhead

    def _flush() -> None:
        nonlocal current_rows, current_len
        if current_rows:
            parts.append(_band_with_header(header, sep, current_rows))
        current_rows = []
        current_len = base_overhead

    for r in rows:
        # Single oversize row — flush whatever we have, then split this row.
        if base_overhead + len(r) > max_chars:
            _flush()
            parts.extend(_split_oversize_row(r, header, sep, max_chars))
            continue

        if current_rows and current_len + len(r) + 1 > max_chars:
            _flush()
        current_rows.append(r)
        current_len += len(r) + 1

    _flush()
    return parts or [md]
