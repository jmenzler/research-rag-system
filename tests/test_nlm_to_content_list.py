"""Unit tests for the nlm.txt → content_list.json synthesizer.

Pins behaviors the structural ingest pipeline depends on:
- Garbage filter rejects too-small + scraper-error pages.
- ATX headings emit ``text_level``-tagged text items (chunker requires this).
- GFM pipe-tables emit ``{type: "table", table_body: "<HTML>", ...}``.
- Lists are concatenated into a single string (per chunker walker contract).
- Idempotency: re-running on a doc that already has content_list.json is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.nlm_to_content_list import (
    Item,
    detect_garbage,
    find_nlm_only_docs,
    main,
    parse_nlm,
    plan_doc,
    synthesize_from_nlm,
)

# ---------------------------------------------------------------------------
# Garbage detector
# ---------------------------------------------------------------------------


def test_too_small_is_garbage() -> None:
    assert detect_garbage("short") == "too_small"


def test_just_under_min_is_garbage() -> None:
    assert detect_garbage("a" * 999) == "too_small"


def test_at_min_passes_size_check() -> None:
    """Exactly 1KB of content should pass the size gate (no error pattern present)."""
    assert detect_garbage("Lorem ipsum " * 100) is None


def test_cloudflare_block_is_garbage() -> None:
    text = (
        "ResearchGate - Temporarily Unavailable\nAccess denied\nRay ID: abc1234\n" + "filler " * 200
    )
    reason = detect_garbage(text)
    assert reason in {"cloudflare", "access_denied", "temp_unavail"}


def test_404_page_is_garbage() -> None:
    text = "Page Not Found\nOops! We couldn't find the page that you're looking for.\n" + "x" * 1100
    assert detect_garbage(text) in {"page_not_found", "oops_404"}


def test_incapsula_block_is_garbage() -> None:
    text = "Request unsuccessful. Incapsula incident ID: 12345" + " filler" * 200
    assert detect_garbage(text) in {"incapsula", "request_unsuccessful"}


def test_429_with_openresty_is_garbage() -> None:
    text = "429 Too Many Requests\nopenresty\n" + "filler " * 200
    assert detect_garbage(text) in {"openresty_block", "error_code"}


def test_real_content_passes() -> None:
    text = (
        "# Real Article\n\nThis is the start of an actual research article "
        "discussing market microstructure and order flow toxicity. " * 30
    )
    assert detect_garbage(text) is None


def test_sorry_page_lookup_is_garbage() -> None:
    """MEXC-style 'Sorry! The page you're looking for cannot be found.'"""
    text = (
        "MEXC Exchange Trading Platform\n"
        "Sorry! The page you're looking for cannot be found. Back to Home\n"
        + "filler nav links "
        * 100
    )
    assert detect_garbage(text) in {"page_lookup", "sorry_page"}


def test_url_only_doc_is_garbage_after_strip() -> None:
    """NotebookLM platform-page dumps with no real content survive the size
    gate but should be quarantined when the URL/UUID lines are stripped.
    """
    text = (
        "\n".join(["https://lh3.googleusercontent.com/foo/bar/baz" + str(i) for i in range(40)])
        + "\n"
        + "9710e176-8dd9-497e-a998-339656f81812\n" * 20
    )
    assert detect_garbage(text) == "low_signal_after_strip"


def test_lecture_slides_with_real_text_among_urls_pass() -> None:
    """Real content (lecture slides) interleaved with thumbnail URLs should pass.

    Mirrors the systems_engineering lecture-slide pattern from the audit:
    half-URL, half-real-text. The strip-then-check approach keeps these.
    """
    real_lines = [
        "Lecture 4: Technical Processes Systems Engineering Wise 2025/26",
        "Prof. Dr.-Ing. Roman Dumitrescu",
        "Heinz Nixdorf Institute, Paderborn University",
        "Tutorial schedule and time plan of the module SE",
        "Lecture 0_Introduction 16. October",
        "Lecture 1_Intelligent Technical Systems 23. October",
        "Lecture 2_Systems Engineering 30. October",
    ] * 6
    url_lines = ["https://lh3.googleusercontent.com/notebooklm/abc" + str(i) for i in range(50)]
    interleaved = "\n".join(line for pair in zip(url_lines, real_lines) for line in pair)
    assert detect_garbage(interleaved) is None


def test_real_doc_with_some_urls_is_not_garbage() -> None:
    """Real articles often have a few cited URLs — those should pass."""
    text = (
        "# Real Research Article\n\n"
        "This article discusses market microstructure and order flow toxicity. "
        "See https://arxiv.org/abs/1234.5678 for the original paper. "
        "The methodology builds on prior work in the field. " * 30
    )
    assert detect_garbage(text) is None


def test_strip_noise_removes_googleusercontent_urls() -> None:
    from scripts.nlm_to_content_list import _strip_noise_lines

    text = (
        "https://lh3.googleusercontent.com/notebooklm/abc123\n"
        "Real text content line.\n"
        "9710e176-8dd9-497e-a998-339656f81812\n"
        "Another real line.\n"
    )
    cleaned = _strip_noise_lines(text)
    assert "googleusercontent" not in cleaned
    assert "9710e176" not in cleaned
    assert "Real text content line." in cleaned
    assert "Another real line." in cleaned


def test_javascript_disabled_alone_is_not_garbage() -> None:
    """Real ScienceDirect pages mention 'JavaScript is disabled' but are valid.

    The js_required pattern was removed during audit — verify this regression.
    """
    text = (
        "Douglas C. Montgomery, Introduction to Time Series Analysis, "
        "Wiley 2008. JavaScript is disabled. " + "Real abstract content. " * 50
    )
    assert detect_garbage(text) is None


# ---------------------------------------------------------------------------
# Markdown parser — heading
# ---------------------------------------------------------------------------


def test_atx_heading_h1() -> None:
    items = parse_nlm("# Title\n\nbody text here\n")
    assert items[0].type == "text"
    assert items[0].text_level == 1
    assert items[0].text == "Title"


def test_atx_heading_h2_h3() -> None:
    items = parse_nlm("## Heading 2\n\n### Heading 3\n\nbody\n")
    assert items[0].text_level == 2
    assert items[1].text_level == 3


def test_atx_heading_h4_collapses_to_3() -> None:
    """Chunker walker only uses text_level 1-3; deeper levels collapse."""
    items = parse_nlm("#### deep heading\nbody text " * 50)
    assert items[0].text_level == 3


def test_heading_then_paragraph_yields_two_items() -> None:
    items = parse_nlm("# Title\n\nFirst paragraph of body.\n")
    assert len(items) == 2
    assert items[0].text_level == 1
    assert items[1].type == "text"
    assert items[1].text_level is None
    assert "First paragraph" in items[1].text


# ---------------------------------------------------------------------------
# Markdown parser — GFM tables
# ---------------------------------------------------------------------------


def test_gfm_table_emits_table_item() -> None:
    text = (
        "Some intro paragraph.\n\n"
        "| Symbol | Meaning |\n"
        "|--------|---------|\n"
        "| γ | gamma |\n"
        "| σ | sigma |\n"
        "\nfollowing paragraph.\n"
    )
    items = parse_nlm(text)
    types = [it.type for it in items]
    assert "table" in types
    table = next(it for it in items if it.type == "table")
    assert "<table>" in table.table_body
    assert "<th>Symbol</th>" in table.table_body
    assert "<td>γ</td>" in table.table_body
    assert "<td>gamma</td>" in table.table_body


def test_gfm_table_required_keys_in_dict() -> None:
    """The chunker reads `table_body` and `table_caption` — both must be present."""
    text = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    items = parse_nlm(text)
    d = items[0].to_dict()
    assert d["type"] == "table"
    assert "table_body" in d
    assert "table_caption" in d
    assert "page_idx" in d


def test_gfm_table_blank_line_between_every_row() -> None:
    """NotebookLM inserts blank lines between every line — parser must tolerate this.

    Real corpus example: trading_literature_review/012__5_strategy_glossary/nlm.txt
    """
    text = (
        "## Greek Letters\n"
        "\n"
        "| Symbol | Name | Range |\n"
        "\n"
        "|--------|------|-------|\n"
        "\n"
        "| γ | gamma | 0.1-3.0 |\n"
        "\n"
        "| κ | kappa | 20-50 bps |\n"
    )
    items = parse_nlm(text)
    types = [it.type for it in items]
    assert "table" in types
    table = next(it for it in items if it.type == "table")
    assert "<th>Symbol</th>" in table.table_body
    assert "<td>γ</td>" in table.table_body
    assert "<td>κ</td>" in table.table_body


def test_list_blank_line_between_every_item() -> None:
    """Lists with blank lines between items still collapse to one list item."""
    text = "- alpha\n\n- beta\n\n- gamma\n"
    items = parse_nlm(text)
    assert len(items) == 1
    assert items[0].type == "list"
    assert items[0].text == "alpha\nbeta\ngamma"


def test_gfm_table_html_escapes_cell_content() -> None:
    text = "| col |\n|-----|\n| a < b & c |\n"
    items = parse_nlm(text)
    body = items[0].table_body
    assert "&lt;" in body
    assert "&amp;" in body
    assert "<b" not in body  # raw < shouldn't survive


# ---------------------------------------------------------------------------
# Markdown parser — lists
# ---------------------------------------------------------------------------


def test_list_block_is_single_item() -> None:
    """Walker contract: list items concatenated as a single string."""
    items = parse_nlm("- alpha\n- beta\n- gamma\n")
    assert len(items) == 1
    assert items[0].type == "list"
    assert items[0].text == "alpha\nbeta\ngamma"


def test_ordered_list_collapses_too() -> None:
    items = parse_nlm("1. first step\n2. second step\n3. third step\n")
    assert len(items) == 1
    assert items[0].type == "list"
    assert "first step" in items[0].text


def test_list_then_paragraph() -> None:
    items = parse_nlm("- one\n- two\n\nfollowing paragraph.\n")
    assert items[0].type == "list"
    assert items[1].type == "text"


# ---------------------------------------------------------------------------
# Markdown parser — block math
# ---------------------------------------------------------------------------


def test_block_math_dollar_form() -> None:
    items = parse_nlm("intro\n\n$$\nx = y + z\n$$\n\nafter\n")
    eqs = [it for it in items if it.type == "equation"]
    assert len(eqs) == 1
    assert "x = y + z" in eqs[0].text


# ---------------------------------------------------------------------------
# Markdown parser — plain text
# ---------------------------------------------------------------------------


def test_plain_paragraphs_blank_separated() -> None:
    items = parse_nlm("First para.\n\nSecond para.\n\nThird para.\n")
    assert len(items) == 3
    assert all(it.type == "text" for it in items)
    assert items[0].text == "First para."
    assert items[2].text == "Third para."


def test_no_structure_yields_text_item() -> None:
    items = parse_nlm("Just one big paragraph of unstructured text.\n")
    assert len(items) == 1
    assert items[0].type == "text"
    assert items[0].text_level is None


# ---------------------------------------------------------------------------
# Mixed content — everything together, in order
# ---------------------------------------------------------------------------


def test_mixed_content_preserves_order() -> None:
    text = (
        "# Doc Title\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Section A\n"
        "\n"
        "- bullet 1\n"
        "- bullet 2\n"
        "\n"
        "| Col | Val |\n"
        "|-----|-----|\n"
        "| x | 1 |\n"
        "\n"
        "Closing paragraph.\n"
    )
    items = parse_nlm(text)
    types = [(it.type, it.text_level) for it in items]
    assert types == [
        ("text", 1),  # # Doc Title
        ("text", None),  # Intro paragraph
        ("text", 2),  # ## Section A
        ("list", None),  # - bullet 1 / - bullet 2
        ("table", None),  # GFM table
        ("text", None),  # Closing paragraph
    ]


# ---------------------------------------------------------------------------
# plan_doc — dispatch decisions
# ---------------------------------------------------------------------------


def _make_doc(tmp_path: Path, text: str) -> Path:
    nb = tmp_path / "test_notebook"
    doc = nb / "001__test"
    doc.mkdir(parents=True)
    (doc / "nlm.txt").write_text(text)
    return doc


def test_plan_quarantines_garbage(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path, "Access denied\nRay ID: x" + " filler" * 200)
    p = plan_doc(doc)
    assert p.action == "quarantine"
    assert p.reason in {"access_denied", "cloudflare"}


def test_plan_synthesizes_real_content(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path, "# Title\n\nA real document body. " * 50)
    p = plan_doc(doc)
    assert p.action == "synthesize"
    assert any(it.type == "text" and it.text_level == 1 for it in p.items)


# ---------------------------------------------------------------------------
# Idempotency + walker
# ---------------------------------------------------------------------------


def test_find_nlm_only_skips_docs_with_existing_sidecar(tmp_path: Path) -> None:
    nb = tmp_path / "nb"
    a = nb / "doc_a"
    a.mkdir(parents=True)
    (a / "nlm.txt").write_text("real content " * 200)

    b = nb / "doc_b"
    b.mkdir(parents=True)
    (b / "nlm.txt").write_text("real content " * 200)
    (b / "content_list.json").write_text("[]")  # already has sidecar

    found = find_nlm_only_docs(tmp_path)
    assert a in found
    assert b not in found


def test_find_nlm_only_skips_quarantine_dirs(tmp_path: Path) -> None:
    """Walker must skip _quarantine_* directories at notebook level."""
    q = tmp_path / "_quarantine_nlm_garbage" / "fake_doc"
    q.mkdir(parents=True)
    (q / "nlm.txt").write_text("content " * 200)
    found = find_nlm_only_docs(tmp_path)
    assert all("_quarantine" not in str(p) for p in found)


def test_main_dry_run_writes_nothing(tmp_path: Path) -> None:
    """--dry-run must leave the corpus untouched."""
    doc = _make_doc(tmp_path, "# Title\n\nReal body content here for the synthesizer. " * 50)
    rc = main(["--root", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert not (doc / "content_list.json").exists()


def test_main_live_run_writes_content_list(tmp_path: Path) -> None:
    doc = _make_doc(tmp_path, "# Title\n\nReal body content here for the synthesizer. " * 50)
    rc = main(["--root", str(tmp_path)])
    assert rc == 0
    out = doc / "content_list.json"
    assert out.exists()
    items = json.loads(out.read_text())
    assert items[0]["type"] == "text"
    assert items[0]["text_level"] == 1


def test_main_live_run_writes_quarantine_manifest(tmp_path: Path) -> None:
    _make_doc(tmp_path, "Access denied\nRay ID: y" + " filler" * 200)
    rc = main(["--root", str(tmp_path)])
    assert rc == 0
    manifest = tmp_path / "_quarantine_nlm_garbage" / "manifest.jsonl"
    assert manifest.exists()
    line = json.loads(manifest.read_text().splitlines()[0])
    assert "path" in line
    assert "reason" in line


def test_main_idempotent_re_run(tmp_path: Path) -> None:
    """Re-running on a doc that already has content_list.json is a no-op."""
    doc = _make_doc(tmp_path, "# Title\n\nReal body content here for the synthesizer. " * 50)
    main(["--root", str(tmp_path)])
    assert (doc / "content_list.json").exists()
    # Modify the synthesized file — re-run should leave it alone
    (doc / "content_list.json").write_text('[{"type": "marker"}]')
    main(["--root", str(tmp_path)])
    items = json.loads((doc / "content_list.json").read_text())
    assert items == [{"type": "marker"}]


# ---------------------------------------------------------------------------
# Item.to_dict — schema conformance
# ---------------------------------------------------------------------------


def test_item_to_dict_text_with_level() -> None:
    d = Item(type="text", text="Hello", text_level=2).to_dict()
    assert d == {"type": "text", "page_idx": 0, "text": "Hello", "text_level": 2}


def test_item_to_dict_text_without_level_omits_field() -> None:
    """Plain text items should not carry text_level (chunker treats absence as body)."""
    d = Item(type="text", text="body").to_dict()
    assert "text_level" not in d


def test_item_to_dict_table_has_required_fields() -> None:
    d = Item(type="table", table_body="<table></table>", table_caption="").to_dict()
    assert d["type"] == "table"
    assert d["table_body"] == "<table></table>"
    assert d["table_caption"] == ""
    assert "text" not in d


# ---------------------------------------------------------------------------
# synthesize_from_nlm — public per-doc API consumed by the fetch spider
# ---------------------------------------------------------------------------


def test_synthesize_from_nlm_no_nlm_returns_skipped(tmp_path: Path) -> None:
    """Doc dir with no nlm.txt is a no-op (most fetches will hit this branch)."""
    doc = tmp_path / "nb" / "001"
    doc.mkdir(parents=True)
    assert synthesize_from_nlm(doc, tmp_path) == "skipped_no_nlm"
    assert not (doc / "content_list.json").exists()


def test_synthesize_from_nlm_existing_sidecar_returns_skipped(tmp_path: Path) -> None:
    """If HTML extraction already wrote a sidecar, the fallback must not overwrite."""
    doc = tmp_path / "nb" / "001"
    doc.mkdir(parents=True)
    (doc / "nlm.txt").write_text("real content " * 200)
    (doc / "content_list.json").write_text('[{"sentinel": true}]')

    assert synthesize_from_nlm(doc, tmp_path) == "skipped_existing"
    items = json.loads((doc / "content_list.json").read_text())
    assert items == [{"sentinel": True}]


def test_synthesize_from_nlm_writes_real_content(tmp_path: Path) -> None:
    doc = tmp_path / "nb" / "001"
    doc.mkdir(parents=True)
    (doc / "nlm.txt").write_text("# Real Title\n\nA real document body for testing. " * 50)
    assert synthesize_from_nlm(doc, tmp_path) == "synthesized"
    items = json.loads((doc / "content_list.json").read_text())
    assert items[0]["type"] == "text"
    assert items[0]["text_level"] == 1


def test_synthesize_from_nlm_quarantines_garbage(tmp_path: Path) -> None:
    doc = tmp_path / "nb" / "001"
    doc.mkdir(parents=True)
    (doc / "nlm.txt").write_text("Access denied\nRay ID: x" + " filler" * 200)
    result = synthesize_from_nlm(doc, tmp_path)
    assert result.startswith("quarantined:")
    assert not (doc / "content_list.json").exists()
    manifest = tmp_path / "_quarantine_nlm_garbage" / "manifest.jsonl"
    assert manifest.exists()


def test_synthesize_from_nlm_idempotent(tmp_path: Path) -> None:
    """Re-running on the same dir is safe: skipped_existing on the second call."""
    doc = tmp_path / "nb" / "001"
    doc.mkdir(parents=True)
    (doc / "nlm.txt").write_text("# Title\n\nReal content. " * 50)
    assert synthesize_from_nlm(doc, tmp_path) == "synthesized"
    assert synthesize_from_nlm(doc, tmp_path) == "skipped_existing"
