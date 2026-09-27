"""Synthesize content_list.json sidecars from nlm.txt for NotebookLM-only docs.

The ingest pipeline routes through the structural chunker only when a
content_list.json sidecar is present. NotebookLM-scraped docs (no source.pdf)
never go through MinerU, so they ship without one and are silently skipped.

This script walks ``sources/<notebook>/<doc>/`` looking for dirs that have
``nlm.txt`` but neither ``content_list.json`` nor ``structured/content_list.json``.
For each:

1. **Garbage filter** — skip docs that are <1KB or whose first 2KB matches a
   known scraper-error pattern (Cloudflare 1020, ResearchGate access denied,
   404 / 429, captcha challenges, etc.). Quarantined docs go to
   ``sources/_quarantine_nlm_garbage/`` (per-doc ``.json`` sidecar plus a
   ``manifest.jsonl`` line) — convention shared with ``_quarantine_poisoned``
   and ``_quarantine_dupes`` (see ``src/cleanup/apply_cleanup_manifest.py``).
2. **GFM-rich markdown extraction** — regex parser detects ATX headings
   (``#``..``######``), GFM pipe-tables, ordered/unordered lists, and block math.
   Everything else becomes a plain ``text`` item.
3. **Idempotent write** — if ``content_list.json`` already exists, the doc is
   left alone. Re-runs are no-ops.

Item shape mirrors the MinerU schema consumed by ``src/chunking/walker.py``:

- ``{"type": "text", "text": "...", "text_level": 1-3, "page_idx": 0}``
- ``{"type": "table", "table_body": "<HTML>", "table_caption": "...", "page_idx": 0}``
- ``{"type": "list", "text": "newline-joined items", "page_idx": 0}``
- ``{"type": "equation", "text": "<LaTeX>", "page_idx": 0}``

Image/figure items are intentionally not emitted — the chunker drops them
(see ``walker.NOISE_TYPES``).

CLI::

    python -m scripts.nlm_to_content_list                  # live
    python -m scripts.nlm_to_content_list --dry-run        # plan only
    python -m scripts.nlm_to_content_list --root sources   # alt root
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Garbage detection
# ---------------------------------------------------------------------------

_GARBAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.IGNORECASE | re.MULTILINE), name)
    for p, name in [
        (r"access\s+denied", "access_denied"),
        (r"\b(cloudflare|cf-ray|ray\s*id)\b", "cloudflare"),
        (r"error\s+(1020|1015|1010|403|404|503|429)", "error_code"),
        (r"site\s+owner.*restrictions", "owner_block"),
        (r"are\s+you\s+a\s+(robot|human)", "captcha"),
        (r"verify\s+(you|that)\s+(are|you)", "captcha2"),
        (r"this\s+(page|site|content)\s+(is\s+)?(not|no\s+longer)\s+available",
         "unavailable"),
        (r"temporarily\s+unavailable", "temp_unavail"),
        (r"^\s*you\s+do\s+not\s+have\s+access", "no_access"),
        (r"checking\s+your\s+browser", "ddos_check"),
        (r"just\s+a\s+moment\.\.\.", "just_moment"),
        # Audit-discovered gaps:
        (r"page\s+not\s+found", "page_not_found"),
        (r"oops[!,.]?\s+(this\s+page|we\s+couldn't|the\s+page)", "oops_404"),
        (r"incapsula\s+incident\s+id", "incapsula"),
        (r"\bopenresty\b", "openresty_block"),
        (r"verification\s+required", "verification_required"),
        (r"verifying\.\.\.", "cf_verifying"),
        (r"\baltcha\b", "altcha_captcha"),
        (r"request\s+unsuccessful", "request_unsuccessful"),
        (r"the\s+page\s+you('re|\s+are)\s+looking\s+for", "page_lookup"),
        (r"sorry[!,.\s]+the\s+page", "sorry_page"),
    ]
)

# Density check: docs that are mostly URLs or single-line garbage are low-signal
# even if large. NotebookLM dumps of platform pages can be 400KB of image URLs
# and UUIDs with zero real text. The parser strips these noise lines while
# parsing, so the threshold here is conservative — only quarantine when
# real-content lines are insufficient even after URL/UUID stripping.
_LOW_SIGNAL_REMAINING_BYTES = 800  # quarantine if <800B remains after stripping
_URL_LINE_RE = re.compile(
    r"^(https?://\S*|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\s*$"
)


def _strip_noise_lines(text: str) -> str:
    """Drop URL-only and UUID-only lines. Return the cleaned text."""
    return "\n".join(
        line for line in text.splitlines()
        if not _URL_LINE_RE.match(line.strip())
    )

_MIN_DOC_BYTES = 1000
_GARBAGE_HEAD_BYTES = 2000
_QUARANTINE_DIRNAME = "_quarantine_nlm_garbage"


def detect_garbage(text: str) -> str | None:
    """Return the matched garbage reason, or ``None`` if the doc looks valid.

    Three checks in order:
    1. Size — too small to carry useful content.
    2. Pattern match — known scraper-error pages (Cloudflare, 404, captcha).
    3. Post-strip size — after stripping URL-only and UUID-only lines, is there
       still enough content? NotebookLM platform-page dumps can be 100KB of
       image URLs with no actual text.
    """
    if len(text) < _MIN_DOC_BYTES:
        return "too_small"
    head = text[:_GARBAGE_HEAD_BYTES]
    for pat, name in _GARBAGE_PATTERNS:
        if pat.search(head):
            return name
    cleaned = _strip_noise_lines(text)
    if len(cleaned.strip()) < _LOW_SIGNAL_REMAINING_BYTES:
        return "low_signal_after_strip"
    return None


# ---------------------------------------------------------------------------
# Markdown → content_list.json item conversion
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+(.+)$")
_BLOCK_MATH_OPEN = re.compile(r"^\s*\$\$\s*$|^\s*\\\[\s*$")
_BLOCK_MATH_CLOSE = re.compile(r"^\s*\$\$\s*$|^\s*\\\]\s*$")


@dataclass
class Item:
    """A typed content_list.json item, ready to be JSON-serialized."""

    type: str
    page_idx: int = 0
    text: str = ""
    text_level: int | None = None
    table_body: str = ""
    table_caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "page_idx": self.page_idx}
        if self.type == "table":
            d["table_body"] = self.table_body
            d["table_caption"] = self.table_caption
        else:
            d["text"] = self.text
            if self.text_level is not None:
                d["text_level"] = self.text_level
        return d


def _gfm_row_cells(line: str) -> list[str]:
    """Split a GFM table row on pipes, dropping the leading/trailing empty cells."""
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def _gfm_table_to_html(rows: list[list[str]]) -> str:
    """Render a list of GFM rows as a simple HTML table.

    First row is treated as <th>; subsequent rows as <td>. Matches the schema
    that ``html_table_to_markdown`` (used by the structural chunker) round-trips.
    """
    if not rows:
        return ""
    out = ["<table>"]
    out.append("<tr>" + "".join(f"<th>{_escape(c)}</th>" for c in rows[0]) + "</tr>")
    for row in rows[1:]:
        out.append("<tr>" + "".join(f"<td>{_escape(c)}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _escape(s: str) -> str:
    """Minimal HTML escape for table cell content."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _next_nonblank(lines: list[str], i: int) -> int:
    """Index of next non-blank line ≥ i, or len(lines) if none."""
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def parse_nlm(text: str) -> list[Item]:
    """Convert markdown-ish text into a list of typed items.

    Tolerant of NotebookLM's habit of inserting blank lines between every line —
    table rows, list items, and consecutive paragraphs may all be blank-separated.

    Handles, in order of detection per line:
    1. GFM table block (header row + ``|---|`` separator + body rows, blank-tolerant)
    2. ATX heading (``#``..``######``)
    3. List block (consecutive ``-``/``*``/``+``/``1.`` lines, blank-tolerant)
    4. Block math (``$$``..``$$`` or ``\\[``..``\\]``)
    5. Plain paragraph (everything else, joined until next structural element)

    Strips URL-only and UUID-only lines (NotebookLM noise) before parsing.
    """
    lines = _strip_noise_lines(text).splitlines()
    items: list[Item] = []
    paragraph_buf: list[str] = []

    def flush_paragraph() -> None:
        joined = "\n".join(paragraph_buf).strip()
        if joined:
            items.append(Item(type="text", text=joined))
        paragraph_buf.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # 1. GFM table: header row, separator on next non-blank line
        if _TABLE_ROW_RE.match(line):
            sep_idx = _next_nonblank(lines, i + 1)
            if sep_idx < n and _TABLE_SEP_RE.match(lines[sep_idx]):
                flush_paragraph()
                header_cells = _gfm_row_cells(line)
                rows: list[list[str]] = [header_cells]
                j = _next_nonblank(lines, sep_idx + 1)
                while j < n and _TABLE_ROW_RE.match(lines[j]):
                    rows.append(_gfm_row_cells(lines[j]))
                    j = _next_nonblank(lines, j + 1)
                html = _gfm_table_to_html(rows)
                items.append(Item(type="table", table_body=html, table_caption=""))
                i = j
                continue

        # 2. ATX heading
        m = _HEADING_RE.match(line)
        if m:
            flush_paragraph()
            level = min(len(m.group(1)), 3)  # collapse 4-6 to 3 (chunker only uses 1-3)
            items.append(
                Item(type="text", text=m.group(2).strip(), text_level=level)
            )
            i += 1
            continue

        # 3. List block (blank-tolerant)
        if _LIST_RE.match(line):
            flush_paragraph()
            list_items: list[str] = []
            while i < n:
                k = _next_nonblank(lines, i)
                if k >= n:
                    break
                lm = _LIST_RE.match(lines[k])
                if not lm:
                    break
                list_items.append(lm.group(2).strip())
                i = k + 1
            items.append(Item(type="list", text="\n".join(list_items)))
            continue

        # 4. Block math
        if _BLOCK_MATH_OPEN.match(line):
            flush_paragraph()
            math_lines: list[str] = []
            i += 1
            while i < n and not _BLOCK_MATH_CLOSE.match(lines[i]):
                math_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            tex = "\n".join(math_lines).strip()
            if tex:
                items.append(Item(type="equation", text=tex))
            continue

        # 5. Blank line — paragraph break
        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        # Otherwise: accumulate into current paragraph
        paragraph_buf.append(line)
        i += 1

    flush_paragraph()
    return items


# ---------------------------------------------------------------------------
# Walker / dispatch
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """Outcome of analyzing one nlm-only doc dir."""

    doc_dir: Path
    nlm_path: Path
    size: int
    action: str  # "synthesize" | "quarantine" | "skip_existing"
    reason: str = ""
    items: list[Item] = field(default_factory=list)


def find_nlm_only_docs(root: Path) -> list[Path]:
    """Return doc directories that have nlm.txt but no content_list.json."""
    out: list[Path] = []
    for nb in sorted(root.iterdir()):
        if not nb.is_dir() or nb.name.startswith("_"):
            continue
        for d in sorted(nb.iterdir()):
            if not d.is_dir():
                continue
            if (d / "nlm.txt").exists() and not (
                (d / "content_list.json").exists()
                or (d / "structured" / "content_list.json").exists()
            ):
                out.append(d)
    return out


def plan_doc(doc_dir: Path) -> Plan:
    """Decide what to do with one doc."""
    nlm = doc_dir / "nlm.txt"
    text = nlm.read_text(errors="ignore")
    size = len(text)

    reason = detect_garbage(text)
    if reason is not None:
        return Plan(doc_dir=doc_dir, nlm_path=nlm, size=size,
                    action="quarantine", reason=reason)

    items = parse_nlm(text)
    if not items:
        return Plan(doc_dir=doc_dir, nlm_path=nlm, size=size,
                    action="quarantine", reason="empty_after_parse")

    return Plan(doc_dir=doc_dir, nlm_path=nlm, size=size,
                action="synthesize", items=items)


def synthesize_from_nlm(doc_dir: Path, root: Path) -> str:
    """Per-doc API for the fetch spider. Idempotent.

    Called after the HTML structured-sidecar path. Synthesizes a
    ``content_list.json`` from ``nlm.txt`` (or quarantines it) only when no
    sidecar exists yet — HTML extraction wins when present.

    Args:
        doc_dir: The source directory (e.g. ``sources/<nb>/<NNN__name>/``).
        root: Corpus root (where ``_quarantine_nlm_garbage/`` lives).

    Returns one of:
        - ``"skipped_no_nlm"``        — no nlm.txt to work with
        - ``"skipped_existing"``      — content_list.json already present
        - ``"synthesized"``           — wrote a fresh content_list.json
        - ``"quarantined:<reason>"``  — moved to _quarantine_nlm_garbage
    """
    nlm = doc_dir / "nlm.txt"
    if not nlm.exists():
        return "skipped_no_nlm"
    if (doc_dir / "content_list.json").exists() or (
        doc_dir / "structured" / "content_list.json"
    ).exists():
        return "skipped_existing"

    plan = plan_doc(doc_dir)
    if plan.action == "synthesize":
        write_synthesized(plan)
        return "synthesized"
    write_quarantine(plan, root)
    return f"quarantined:{plan.reason}"


def write_quarantine(plan: Plan, root: Path) -> None:
    """Emit per-doc sidecar + append to manifest.jsonl. Per quarantine convention."""
    qroot = root / _QUARANTINE_DIRNAME
    qroot.mkdir(exist_ok=True)
    rel = plan.doc_dir.relative_to(root)
    sidecar_path = qroot / f"{rel.as_posix().replace('/', '__')}.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "path": rel.as_posix(),
                "reason": plan.reason,
                "size": plan.size,
                "first_chars": plan.nlm_path.read_text(errors="ignore")[:240],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = qroot / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"path": rel.as_posix(), "reason": plan.reason, "size": plan.size},
                ensure_ascii=False,
            )
            + "\n"
        )


def write_synthesized(plan: Plan) -> None:
    """Write content_list.json next to nlm.txt."""
    out_path = plan.doc_dir / "content_list.json"
    payload = [item.to_dict() for item in plan.items]
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _classify_for_summary(items: list[Item]) -> str:
    """Bucket synthesized docs for the human-readable summary."""
    has_heading = any(it.type == "text" and it.text_level is not None for it in items)
    has_table = any(it.type == "table" for it in items)
    has_list = any(it.type == "list" for it in items)
    if has_heading and has_table:
        return "md_full"
    if has_heading or has_table or has_list:
        return "md_partial"
    return "plain"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.nlm_to_content_list",
        description="Synthesize content_list.json sidecars from NotebookLM nlm.txt.",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("sources"),
        help="Corpus root (default: sources)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only; print summary without writing files.",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    docs = find_nlm_only_docs(root)
    print(f"[nlm_to_content_list] {len(docs)} nlm-only docs scanned under {root}/")

    quarantine_reasons: dict[str, int] = {}
    bucket_counts: dict[str, int] = {"md_full": 0, "md_partial": 0, "plain": 0}
    plans: list[Plan] = []

    for d in docs:
        p = plan_doc(d)
        plans.append(p)
        if p.action == "quarantine":
            quarantine_reasons[p.reason] = quarantine_reasons.get(p.reason, 0) + 1
        elif p.action == "synthesize":
            bucket_counts[_classify_for_summary(p.items)] += 1

    n_synth = sum(1 for p in plans if p.action == "synthesize")
    n_quar = sum(1 for p in plans if p.action == "quarantine")

    print(f"  to synthesize: {n_synth}")
    for b in ("md_full", "md_partial", "plain"):
        print(f"    {b:<12} {bucket_counts[b]}")
    print(f"  to quarantine: {n_quar}")
    for r, n in sorted(quarantine_reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:<22} {n}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    qroot = root / _QUARANTINE_DIRNAME
    if qroot.exists():
        # Truncate the manifest so re-runs don't double-append.
        manifest = qroot / "manifest.jsonl"
        if manifest.exists():
            manifest.unlink()

    n_synth_written = 0
    n_quar_written = 0
    for p in plans:
        if p.action == "synthesize":
            write_synthesized(p)
            n_synth_written += 1
        elif p.action == "quarantine":
            write_quarantine(p, root)
            n_quar_written += 1

    print(
        f"\n[live] wrote {n_synth_written} content_list.json, "
        f"{n_quar_written} quarantine entries -> {qroot}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
