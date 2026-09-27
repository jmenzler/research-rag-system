"""Unit tests for src.fetch.metadata — restored extractor + new helpers + enrichment.

All tests use synthetic fixtures via ``tmp_path``; the real corpus (798 sources
under ``sources/*/*/meta.json``) is only touched during manual verification.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers for building synthetic doc dirs in tmp_path
# ---------------------------------------------------------------------------


def _make_doc_dir(tmp: Path, name: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_meta(doc_dir: Path, meta: dict) -> Path:
    p = doc_dir / "meta.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def _write_txt(doc_dir: Path, fname: str, text: str) -> Path:
    p = doc_dir / fname
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. extract_arxiv_id
# ---------------------------------------------------------------------------

# Will import after the module exists; use importlib to test existence first


def test_extract_arxiv_id_from_pdf_url() -> None:
    """https://arxiv.org/pdf/2403.12345v2 -> '2403.12345'"""
    from src.fetch.metadata import extract_arxiv_id

    assert extract_arxiv_id("https://arxiv.org/pdf/2403.12345v2") == "2403.12345"


def test_extract_arxiv_id_from_html_url() -> None:
    """https://arxiv.org/html/2403.12345 -> '2403.12345'"""
    from src.fetch.metadata import extract_arxiv_id

    assert extract_arxiv_id("https://arxiv.org/html/2403.12345") == "2403.12345"


def test_extract_arxiv_id_handles_5_digit() -> None:
    """2403.12345v1 and 2403.123456 (5-digit) both covered"""
    from src.fetch.metadata import extract_arxiv_id

    assert extract_arxiv_id("https://arxiv.org/abs/2403.12345v1") == "2403.12345"
    assert extract_arxiv_id("https://arxiv.org/abs/2403.123456") == "2403.123456"


def test_extract_arxiv_id_returns_none_on_non_arxiv() -> None:
    """Non-arxiv URL returns None"""
    from src.fetch.metadata import extract_arxiv_id

    assert extract_arxiv_id("https://example.com") is None
    assert extract_arxiv_id("https://ssrn.com/abstract=12345") is None
    assert extract_arxiv_id("") is None


# ---------------------------------------------------------------------------
# 2. extract_doi_from_pdf_text
# ---------------------------------------------------------------------------


def test_extract_doi_from_pdf_text() -> None:
    """Text containing 'doi.org/10.1234/abc.5678' -> that DOI"""
    from src.fetch.metadata import extract_doi_from_pdf_text

    text = "Some preamble text. See https://doi.org/10.1234/abc.5678 for details."
    result = extract_doi_from_pdf_text(text)
    assert result == "10.1234/abc.5678"


def test_extract_doi_handles_inline_form() -> None:
    """'DOI: 10.1234/abc.5678' -> that DOI"""
    from src.fetch.metadata import extract_doi_from_pdf_text

    text = "DOI: 10.1234/abc.5678"
    result = extract_doi_from_pdf_text(text)
    assert result == "10.1234/abc.5678"


def test_extract_doi_returns_none_on_no_match() -> None:
    """Random prose -> None"""
    from src.fetch.metadata import extract_doi_from_pdf_text

    assert extract_doi_from_pdf_text("This is just random text.") is None
    assert extract_doi_from_pdf_text("") is None


# ---------------------------------------------------------------------------
# 3. quality_counts
# ---------------------------------------------------------------------------


def test_quality_counts_reads_pdf_sidecar(tmp_path: Path) -> None:
    """Synthetic content_list.json with mixed types -> correct counts"""
    from src.fetch.metadata import quality_counts

    cl = tmp_path / "content_list.json"
    cl.write_text(
        json.dumps(
            [
                {"type": "text", "page_idx": 0, "text": "..."},
                {"type": "table", "page_idx": 1, "text": "..."},
                {"type": "table", "page_idx": 2, "text": "..."},
                {"type": "equation", "page_idx": 2, "text": "..."},
                {"type": "figure", "page_idx": 3, "text": "..."},
                {"type": "list", "page_idx": 4, "text": "..."},
                {"type": "list", "page_idx": 4, "text": "..."},
            ]
        ),
        encoding="utf-8",
    )

    qc = quality_counts(cl)
    assert qc["has_structured"] is True
    assert qc["n_pages"] == 5  # max page_idx=4  -> 5 pages
    assert qc["n_tables"] == 2
    assert qc["n_equations"] == 1
    assert qc["n_figures"] == 1
    assert qc["n_lists"] == 2


def test_quality_counts_handles_missing_file() -> None:
    """Nonexistent path -> {has_structured: False, ...zeros}"""
    from src.fetch.metadata import quality_counts

    qc = quality_counts(Path("/nonexistent/content_list.json"))
    assert qc["has_structured"] is False
    assert qc["n_pages"] == 0
    assert qc["n_tables"] == 0
    assert qc["n_equations"] == 0
    assert qc["n_figures"] == 0
    assert qc["n_lists"] == 0


def test_quality_counts_n_pages_uses_max_page_idx(tmp_path: Path) -> None:
    """Elements with page_idx in {0,1,2,4} -> n_pages=5"""
    from src.fetch.metadata import quality_counts

    cl = tmp_path / "content_list.json"
    cl.write_text(
        json.dumps(
            [
                {"type": "text", "page_idx": 0},
                {"type": "text", "page_idx": 1},
                {"type": "text", "page_idx": 2},
                {"type": "table", "page_idx": 4},
            ]
        ),
        encoding="utf-8",
    )

    qc = quality_counts(cl)
    assert qc["n_pages"] == 5


# ---------------------------------------------------------------------------
# 4. extract_metadata (restored rules)
# ---------------------------------------------------------------------------


def test_extract_metadata_arxiv_url_high_confidence(tmp_path: Path) -> None:
    """Real arxiv URL + plausible pdf.txt text -> confidence=high"""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "test_doc")
    pdf_text = (
        "# Deep Reinforcement Learning for Market Making\n\n"
        "John Smith, Jane Doe, and Robert Johnson\n\n"
        "Abstract\n\n"
        "We present a novel approach to market making using deep "
        "reinforcement learning, trained on simulated limit order book data.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {"url": "https://arxiv.org/abs/2403.12345"}
    result = extract_metadata(doc_dir, meta)

    assert result.confidence == "high"
    assert len(result.authors) >= 2
    assert result.year == 2024  # from arxiv URL year extraction
    assert result.short_cite is not None


def test_extract_metadata_login_screen_falls_through(tmp_path: Path) -> None:
    """pdf.txt that is a login page -> confidence='none'"""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "test_doc")
    pdf_text = (
        "Login to your account\n\n"
        "Username: ______\n"
        "Password: ______\n"
        "Keep me logged in\n"
        "Forgot password?\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {"url": "https://example.com/paper"}
    result = extract_metadata(doc_dir, meta)

    assert result.confidence == "none"
    assert result.authors == []
    assert result.year is None


def test_extract_metadata_particle_surnames(tmp_path: Path) -> None:
    """Author line with de/van/der particles -> correct short cite"""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "test_doc")
    pdf_text = (
        "# Some Paper\n\n"
        "Marcos Lopez de Prado\n\n"
        "Abstract: This paper develops an empirical framework for evaluating "
        "the out-of-sample performance of quantitative trading strategies.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {"url": "https://example.com"}
    result = extract_metadata(doc_dir, meta)

    assert len(result.authors) >= 1
    # The lastname extraction should handle "López de Prado"
    assert result.short_cite is not None
    assert "Prado" in result.short_cite


# ---------------------------------------------------------------------------
# 5. enrich_meta_json (in-place idempotent write)
# ---------------------------------------------------------------------------


def test_enrich_meta_json_idempotent(tmp_path: Path) -> None:
    """Run enrich_meta_json twice; second run produces no changes."""
    from src.fetch.metadata import enrich_meta_json

    doc_dir = _make_doc_dir(tmp_path, "test_doc")
    pdf_text = (
        "# Deep RL for Trading\n\n"
        "Alice Brown and Bob White\n\n"
        "Abstract\n\n"
        "We study RL-based trading strategies.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)
    meta_path = _write_meta(
        doc_dir,
        {
            "index": 1,
            "title": "Deep RL for Trading",
            "url": "https://arxiv.org/abs/2403.12345",
            "host": "arxiv.org",
            "tier": "T0_arxiv",
            "fetch": {},
        },
    )

    # First run
    changed = enrich_meta_json(meta_path)
    assert changed is True

    # Second run — idempotent, no diff
    changed2 = enrich_meta_json(meta_path)
    assert changed2 is False


def test_enrich_meta_json_overwrites_metadata_only(tmp_path: Path) -> None:
    """Pre-existing extracted_at / parser_versions preserved untouched."""
    from src.fetch.metadata import enrich_meta_json

    doc_dir = _make_doc_dir(tmp_path, "test_doc")
    pdf_text = "# A Great Paper\n\nCarol Davis\n\nAbstract text.\n"
    _write_txt(doc_dir, "pdf.txt", pdf_text)
    original_extracted_at = "2026-05-01T00:00:00+00:00"
    original_parser_versions = {"mineru": "1.0.0", "trafilatura": "9.9.9", "git_sha": "abc1234"}
    original_page_metadata = {"doi": "10.9999/foo.bar"}
    meta_path = _write_meta(
        doc_dir,
        {
            "index": 1,
            "title": "A Great Paper",
            "url": "https://example.com",
            "host": "example.com",
            "tier": "T5_web_blog",
            "fetch": {},
            "extracted_at": original_extracted_at,
            "parser_versions": original_parser_versions,
            "page_metadata": original_page_metadata,
        },
    )

    enrich_meta_json(meta_path)

    # Read back
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["extracted_at"] == original_extracted_at
    assert meta["parser_versions"] == original_parser_versions
    assert meta["page_metadata"] == original_page_metadata
    # metadata block should have been written
    assert "metadata" in meta
    assert "authors" in meta["metadata"]


# ---------------------------------------------------------------------------
# 6. _detect_parser_versions
# ---------------------------------------------------------------------------


def test_detect_parser_versions_returns_dict_with_known_keys() -> None:
    """Result has mineru/trafilatura/git_sha keys, all str."""
    from src.fetch.metadata import _detect_parser_versions

    versions = _detect_parser_versions()
    assert isinstance(versions, dict)
    for key in ("mineru", "trafilatura", "git_sha"):
        assert key in versions, f"missing key: {key}"
        assert isinstance(versions[key], str), f"{key} should be str"


# ---------------------------------------------------------------------------
# 7. Non-regression guard — must not silently downgrade confidence
# ---------------------------------------------------------------------------


def test_merge_with_guard_preserves_high_confidence_authors() -> None:
    """When old confidence is 'high' and new is 'medium', preserve old authors."""
    from src.fetch.metadata import _merge_with_guard

    old_md = {
        "authors": ["Smith, J.", "Jones, K."],
        "year": 2024,
        "short_cite": "Smith & Jones 2024",
        "confidence": "high",
    }
    new_md: dict[str, object] = {
        "authors": [],
        "year": None,
        "short_cite": None,
        "confidence": "medium",
        "arxiv_id": "2403.12345",
        "doi": None,
        "extraction_quality": {"n_pages": 12},
    }
    merged = _merge_with_guard(old_md, new_md)
    # Old high-confidence fields preserved
    assert merged["authors"] == ["Smith, J.", "Jones, K."]
    assert merged["year"] == 2024
    assert merged["short_cite"] == "Smith & Jones 2024"
    assert merged["confidence"] == "high"
    # New owned fields still applied
    assert merged["arxiv_id"] == "2403.12345"
    assert merged["extraction_quality"] == {"n_pages": 12}


def test_merge_with_guard_takes_new_when_new_is_higher() -> None:
    """When new confidence beats old, new fields win across the board."""
    from src.fetch.metadata import _merge_with_guard

    old_md = {
        "authors": [],
        "year": None,
        "short_cite": None,
        "confidence": "low",
    }
    new_md: dict[str, object] = {
        "authors": ["Liu, X."],
        "year": 2025,
        "short_cite": "Liu 2025",
        "confidence": "high",
        "arxiv_id": "2503.99999",
        "doi": None,
        "extraction_quality": {"n_pages": 5},
    }
    merged = _merge_with_guard(old_md, new_md)
    assert merged["authors"] == ["Liu, X."]
    assert merged["confidence"] == "high"


def test_merge_with_guard_handles_missing_old_authors() -> None:
    """Old authors=[] should not trigger the guard even with high confidence."""
    from src.fetch.metadata import _merge_with_guard

    old_md = {"authors": [], "confidence": "high"}
    new_md: dict[str, object] = {
        "authors": ["Smith, J."],
        "year": 2024,
        "short_cite": "Smith 2024",
        "confidence": "medium",
        "arxiv_id": None,
        "doi": None,
        "extraction_quality": {},
    }
    merged = _merge_with_guard(old_md, new_md)
    # Empty old.authors → no preservation → take the new authors
    assert merged["authors"] == ["Smith, J."]
    assert merged["confidence"] == "medium"


# ---------------------------------------------------------------------------
# 8. MinerU-style flattened author line (regression)
# ---------------------------------------------------------------------------


def test_extract_doi_from_meta_synthesizes_arxiv_doi(tmp_path: Path) -> None:
    """When no DOI is found in text but arxiv_id is set, derive 10.48550/arXiv.<id>."""
    from src.fetch.metadata import _extract_doi_from_meta

    d = _make_doc_dir(tmp_path, "arxiv_only")
    # No pdf/nlm/web with a DOI, no page_metadata.doi.
    _write_txt(d, "pdf.txt", "# Title\n\nSome paper without a DOI in body.\n")

    result = _extract_doi_from_meta(
        {"url": "https://arxiv.org/abs/2403.12345"},
        d,
        arxiv_id="2403.12345",
    )
    assert result == "10.48550/arXiv.2403.12345"


def test_extract_doi_from_meta_prefers_real_doi_over_arxiv_synth(tmp_path: Path) -> None:
    """A DOI found in pdf.txt outranks the arxiv-derived fallback."""
    from src.fetch.metadata import _extract_doi_from_meta

    d = _make_doc_dir(tmp_path, "arxiv_with_journal_doi")
    _write_txt(d, "pdf.txt", "Published in JFE.\nDOI: 10.1093/rfs/hhx084\n")
    result = _extract_doi_from_meta({"url": ""}, d, arxiv_id="2403.12345")
    assert result == "10.1093/rfs/hhx084"


def test_extract_doi_from_meta_reads_nlm_txt(tmp_path: Path) -> None:
    """Sources without pdf.txt must fall through to nlm.txt for DOI lookup."""
    from src.fetch.metadata import _extract_doi_from_meta

    d = _make_doc_dir(tmp_path, "nlm_only")
    # No pdf.txt; DOI lives in nlm.txt body.
    _write_txt(d, "nlm.txt", "Some intro\nhttps://doi.org/10.1234/example.5678\nrest of body\n")

    result = _extract_doi_from_meta({"url": ""}, d)
    assert result == "10.1234/example.5678"


def test_extract_authors_recovers_from_mineru_flattened_affiliation_line() -> None:
    """MinerU sometimes outputs `Author1*, Author2† 1Univ A, 2Univ B email@x.y` as one line.

    The inline-affil splitter must peel off the leading author segment and
    the iterated footnote stripper must drop the trailing superscript digits.
    """
    from src.fetch.metadata import _extract_authors, _format_short_cite

    text = (
        "# Resolving Latency in Market Making\n"
        "\n"
        "Junzhe Jiang1∗ , Chang Yang1∗ , Xinrun Wang2†, Zhiming Li3, "
        "Xiao Huang1, Bo Li1 1The Hong Kong Polytechnic University, "
        "2Singapore Management University, 3Nanyang Technological University "
        "author-one@example.com, author-two@example.org\n"
        "\n"
        "## Abstract\n"
    )
    authors = _extract_authors(text)
    assert authors == [
        "Junzhe Jiang",
        "Chang Yang",
        "Xinrun Wang",
        "Zhiming Li",
        "Xiao Huang",
        "Bo Li",
    ]
    assert _format_short_cite(authors, 2025) == "Jiang et al. 2025"


# ---------------------------------------------------------------------------
# 9. _harvest_page_metadata wiring + page_metadata.authors consumption
# ---------------------------------------------------------------------------


def test_harvest_page_metadata_extracts_citation_author() -> None:
    """citation_author <meta> tags become page_metadata['authors']."""
    from src.fetch.classify import _harvest_page_metadata  # noqa: PLC0415

    html = (
        "<html><head>"
        '<meta name="citation_author" content="John Smith">'
        '<meta name="citation_author" content="Jane Doe">'
        '<meta name="citation_title" content="A Paper About Trading">'
        "</head><body>...</body></html>"
    )
    result = _harvest_page_metadata(html)
    assert result["authors"] == ["John Smith", "Jane Doe"]
    assert result["title"] == "A Paper About Trading"


def test_extract_metadata_prefers_page_metadata_authors(tmp_path: Path) -> None:
    """page_metadata.authors override the text-scan extractor; confidence=high."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "page_meta_doc")
    # pdf.txt that the text scanner would extract a *different* author from
    # (proving the page_metadata path actually wins).
    pdf_text = (
        "# A Paper About Latency\n\n"
        "Wrong Author Listed Here\n\n"
        "Abstract\n\n"
        "This paragraph is long enough to clear the quality floor in the fail "
        "signature detector so we exercise the page_metadata branch cleanly.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {
        "url": "https://arxiv.org/abs/2403.12345",
        "page_metadata": {"authors": ["Alice Real", "Bob Real"]},
    }
    result = extract_metadata(doc_dir, meta)

    assert result.authors == ["Alice Real", "Bob Real"]
    assert result.year == 2024
    assert result.confidence == "high"
    assert result.short_cite is not None


def test_extract_metadata_ignores_invalid_page_metadata_authors(tmp_path: Path) -> None:
    """Invalid author metadata must fall back to the text scan."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "bad_page_meta_doc")
    pdf_text = (
        "# A Paper\n\n"
        "Carol Davis\n\n"
        "Abstract\n\n"
        "This paragraph is long enough to clear the quality floor in the fail "
        "signature detector so the text-scan branch runs cleanly.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {
        "url": "https://arxiv.org/abs/2403.12345",
        # email pattern fails _looks_like_author_line → fall back
        "page_metadata": {"authors": ["someone@example.com"]},
    }
    result = extract_metadata(doc_dir, meta)

    assert "Carol Davis" in result.authors
    # Year still extractable from URL even if authors fell back
    assert result.year == 2024


# ---------------------------------------------------------------------------
# 10. _read_doc_text fail-page filter
# ---------------------------------------------------------------------------


def test_read_doc_text_rejects_cloudflare_fail_page(tmp_path: Path) -> None:
    """A pdf.txt that is a Cloudflare challenge page is skipped → confidence=none."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "cf_doc")
    pdf_text = (
        "Just a moment...\n"
        "Enable JavaScript and cookies to continue\n"
        "Ray ID: 8abc123def\n"
        "Privacy and security by Cloudflare\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {"url": "https://blocked.example.com/paper"}
    result = extract_metadata(doc_dir, meta)

    assert result.confidence == "none"
    assert result.authors == []
    assert result.year is None


def test_read_doc_text_rejects_recaptcha(tmp_path: Path) -> None:
    """A pdf.txt that is a reCAPTCHA challenge is skipped → confidence=none."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "recap_doc")
    pdf_text = (
        "Are you a robot?\n"
        "Please complete the captcha below to continue.\n"
        "If you are a human, click the checkbox.\n"
    )
    _write_txt(doc_dir, "pdf.txt", pdf_text)

    meta = {"url": "https://blocked.example.com/paper"}
    result = extract_metadata(doc_dir, meta)

    assert result.confidence == "none"
    assert result.authors == []


# ---------------------------------------------------------------------------
# 11. _looks_like_author_line — web-title rejection rules
# ---------------------------------------------------------------------------


def test_looks_like_author_rejects_pipe_separator() -> None:
    """HTML <title> patterns with '|' must NOT be classified as authors."""
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("Some Paper Title | Journal of Finance") is False
    assert _looks_like_author_line("Market Making Strategies | SSRN") is False


def test_looks_like_author_rejects_trailing_brand() -> None:
    """Lines ending in ' - Brand' or ' — Brand' must NOT be classified as authors."""
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("Quantitative Finance Article - Springer") is False
    assert _looks_like_author_line("Some Working Paper — Elsevier") is False


def test_looks_like_author_rejects_long_brand_subtitle() -> None:
    """Long ' Title: Subtitle ' web headlines (no comma before colon) rejected.

    Author lines like 'Smith, J.: A Novel Approach' still pass (comma precedes colon).
    """
    from src.fetch.metadata import _looks_like_author_line

    assert (
        _looks_like_author_line("Optimal Execution: A New Framework for Algorithmic Trading")
        is False
    )
    # Author lines must still pass — comma before colon distinguishes them.
    assert _looks_like_author_line("Smith, J.: A Novel Approach to Hedging") is True


def test_looks_like_author_rejects_nav_phrases() -> None:
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("Jump to content") is False
    assert _looks_like_author_line("Skip to main content") is False
    assert _looks_like_author_line("Click here for more") is False
    assert _looks_like_author_line("Learn more about us") is False


def test_looks_like_author_rejects_leading_emoji() -> None:
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("🚀 FREE Guide ·  Quant Firm Tier List →") is False
    assert _looks_like_author_line("⭐ Top picks for traders") is False


def test_looks_like_author_rejects_trailing_arrow() -> None:
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("Read the full guide →") is False
    assert _looks_like_author_line("More articles »") is False


def test_looks_like_author_rejects_single_token() -> None:
    """Single-word strings ('Products' / 'QuantStart' / 'Mergify') are brands, not authors.

    Real single-name authors in modern papers are vanishingly rare; reject all.
    """
    from src.fetch.metadata import _looks_like_author_line

    assert _looks_like_author_line("Products") is False
    assert _looks_like_author_line("QuantStart") is False
    assert _looks_like_author_line("Mergify") is False


def test_split_authors_rejects_sentence_fragments() -> None:
    """PDF text-scan leak: prose paragraphs grabbed as 'authors'."""
    from src.fetch.metadata import _split_authors

    # Backed.fi disclosure (one of 7+ identical cases in the audit)
    line = "According to Art. Para. Sub-Para, Art. of the Regulation"
    assert _split_authors(line) == []

    # Cornell CS lecture (LaTeX)
    line2 = "$x, y in mathbb { R } ^ { n }$"
    assert _split_authors(line2) == []

    # Morgan Stanley boilerplate
    line3 = "This document is part of Morgan Stanley's ongoing efforts"
    assert _split_authors(line3) == []

    # Real authors still pass
    line4 = "Alice Smith, Bob Jones, and Carol Lee"
    out = _split_authors(line4)
    assert "Alice Smith" in out
    assert "Bob Jones" in out
    assert "Carol Lee" in out


def test_sitename_rejects_month_name() -> None:
    """'March' / 'February' must never end up as a sitename attribution."""
    from src.fetch.metadata import _valid_sitename

    assert _valid_sitename("March") is None
    assert _valid_sitename("February") is None
    assert _valid_sitename("December") is None
    # Real brand still passes
    assert _valid_sitename("Mergify") == "Mergify"


def test_sitename_rejects_weekday_name() -> None:
    """'Monday' from a conference schedule must not become a sitename."""
    from src.fetch.metadata import _valid_sitename

    assert _valid_sitename("Monday") is None
    assert _valid_sitename("Tuesday") is None


def test_sitename_rejects_fail_page_brand() -> None:
    """A CF/captcha stub's brand must not be attributed as the source author."""
    from src.fetch.metadata import _valid_sitename

    assert _valid_sitename("Cloudflare") is None
    assert _valid_sitename("reCAPTCHA") is None
    assert _valid_sitename("Akamai") is None


def test_sitename_canonicalizes_brand_case() -> None:
    """Case duplicates ('Github' / 'GitHub') canonicalize to one form."""
    from src.fetch.metadata import _canonicalize_brand, _valid_sitename

    assert _canonicalize_brand("github") == "GitHub"
    assert _canonicalize_brand("Github") == "GitHub"
    assert _canonicalize_brand("GitHub") == "GitHub"
    assert _canonicalize_brand("mdpi") == "MDPI"
    assert _canonicalize_brand("Mdpi") == "MDPI"
    assert _canonicalize_brand("researchgate") == "ResearchGate"
    assert _canonicalize_brand("Researchgate") == "ResearchGate"
    # Unknown brand passes through unchanged
    assert _canonicalize_brand("RandomBrand") == "RandomBrand"
    # Plumbed through _valid_sitename too
    assert _valid_sitename("github") == "GitHub"


def test_sitename_rejects_prose_in_sitename_slot() -> None:
    """Sentence-shape rejection also applies to sitename candidates."""
    from src.fetch.metadata import _valid_sitename

    assert _valid_sitename("VERSION OF DECEMBER 1ST") is None
    assert _valid_sitename("With the collaboration of") is None
    assert _valid_sitename("This document is part of the firm's policies.") is None


def test_split_authors_rejects_weekday_and_schedule_fragments() -> None:
    """Conference schedule garbage like 'Monday, 20:00-22: - Room' must not pass.

    Note: a bare single capitalized English word ('Track') will still pass —
    can't be rejected without losing real single-word surnames. The full
    ifors_2021 case below works because every non-weekday entry has digits.
    """
    from src.fetch.metadata import _split_authors

    assert _split_authors("Monday, 20:00-22: - Room") == []


def test_split_authors_keeps_short_simple_names() -> None:
    """Edge: 'Smith, J.' / 'F. Black' / 'Bob Li' must still pass."""
    from src.fetch.metadata import _split_authors

    out = _split_authors("Smith, J., F. Black, Bob Li")
    assert "Smith" in " ".join(out) or "J." in " ".join(out)
    assert "F. Black" in out
    assert "Bob Li" in out


def test_extract_metadata_platform_attribution(tmp_path: Path) -> None:
    """Wikipedia/Reddit/HN/Stack* attribute to the platform, not individuals."""
    from src.fetch.metadata import extract_metadata

    cases = [
        ("https://en.wikipedia.org/wiki/Kelly_criterion", ["Wikipedia contributors"]),
        ("https://www.reddit.com/r/quant/comments/abc/", ["Reddit"]),
        ("https://news.ycombinator.com/item?id=12345", ["Hacker News"]),
        ("https://stackoverflow.com/questions/abc", ["Stack Overflow"]),
    ]
    for host_url, expected in cases:
        doc_dir = _make_doc_dir(tmp_path, "platform_" + host_url.split("/")[2])
        _write_txt(doc_dir, "web.txt", "Some content\n")
        meta = {
            "url": host_url,
            "page_metadata": {"authors": ["Should Be Ignored"]},
        }
        result = extract_metadata(doc_dir, meta)
        assert result.authors == expected, f"platform attribution failed for {host_url}"
        # Platform attribution caps confidence at medium.
        assert result.confidence != "high"


def test_extract_metadata_platform_short_cite_uses_page_title(tmp_path: Path) -> None:
    """short_cite for platform sources carries the page title as signal."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "wiki_kelly")
    _write_txt(doc_dir, "web.txt", "Some content\n")
    meta = {
        "url": "https://en.wikipedia.org/wiki/Kelly_criterion",
        "page_metadata": {"title": "Kelly criterion - Wikipedia"},
    }
    result = extract_metadata(doc_dir, meta)
    assert result.authors == ["Wikipedia contributors"]
    # Title stripped of " - Wikipedia" then platform name appended.
    assert result.short_cite is not None
    assert "Kelly criterion" in result.short_cite
    assert "Wikipedia" in result.short_cite


def test_extract_metadata_platform_short_cite_includes_year_from_page_metadata(
    tmp_path: Path,
) -> None:
    """Trafilatura's date field flows into short_cite even when text has no year."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "wiki_dated")
    _write_txt(doc_dir, "web.txt", "no year in body\n")
    meta = {
        "url": "https://en.wikipedia.org/wiki/Kelly_criterion",
        "page_metadata": {"title": "Kelly criterion", "published": "2023-08-15"},
    }
    result = extract_metadata(doc_dir, meta)
    assert result.short_cite is not None
    assert "2023" in result.short_cite
    assert "Wikipedia" in result.short_cite


def test_extract_metadata_platform_short_cite_truncates_long_title(tmp_path: Path) -> None:
    """short_cite stays readable even for novel-length reddit titles."""
    from src.fetch.metadata import extract_metadata

    long_title = (
        "Some absurdly long reddit post title that goes on and on and "
        "describes the entire content of the thread in one breath without "
        "stopping for anything"
    )
    doc_dir = _make_doc_dir(tmp_path, "reddit_long")
    _write_txt(doc_dir, "web.txt", "x\n")
    meta = {
        "url": "https://www.reddit.com/r/whatever/comments/abc/",
        "page_metadata": {"title": long_title},
    }
    result = extract_metadata(doc_dir, meta)
    assert result.short_cite is not None
    assert len(result.short_cite) < 130
    assert result.short_cite.startswith("Some absurdly long")
    assert "Reddit" in result.short_cite


def test_extract_metadata_platform_short_cite_falls_back_to_top_level_title(tmp_path: Path) -> None:
    """When page_metadata.title is missing, use top-level meta['title']."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "wiki_no_pm")
    _write_txt(doc_dir, "web.txt", "Some content\n")
    meta = {
        "url": "https://en.wikipedia.org/wiki/Markov_chain",
        "title": "Markov chain",
    }
    result = extract_metadata(doc_dir, meta)
    assert result.short_cite is not None
    assert "Markov chain" in result.short_cite


def test_extract_metadata_sitename_from_title_tail(tmp_path: Path) -> None:
    """Title ' - Quant Blueprint' → sitename = 'Quant Blueprint'."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "blueprint")
    _write_txt(doc_dir, "web.txt", "x\n")
    meta = {
        "url": "https://www.quantblueprint.com/glossary/kelly-criterion",
        "page_metadata": {"title": "Kelly Criterion Explained - Quant Blueprint"},
    }
    result = extract_metadata(doc_dir, meta)
    assert result.authors == ["Quant Blueprint"]
    assert result.short_cite is not None
    assert "Quant Blueprint" in result.short_cite


def test_extract_metadata_sitename_from_url_host_when_no_title(tmp_path: Path) -> None:
    """No title, no sitename — derive from URL host as last resort."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "host_only")
    web_text = (
        "Article body with substantial prose to clear the fail-signature "
        "floor; the host fallback only kicks in when content reached us.\n"
    )
    _write_txt(doc_dir, "web.txt", web_text)
    meta = {"url": "https://www.quantstart.com/articles/some-page"}
    result = extract_metadata(doc_dir, meta)
    # Host = "quantstart.com" → "Quantstart" → canonicalized to "QuantStart".
    assert result.authors == ["QuantStart"]


def test_extract_metadata_web_source_no_text_scan_falls_back_to_sitename(tmp_path: Path) -> None:
    """Web-only sources without page_metadata.authors must NOT text-scan;
    fall back to sitename so the citation still has signal."""
    from src.fetch.metadata import extract_metadata

    for fname in ("web.txt", "nlm.txt"):
        doc_dir = _make_doc_dir(tmp_path, f"web_only_{fname}")
        web_text = (
            "# Some Article Title\n\n"
            "Jump to content\n\n"
            "By Anonymous Reader\n\n"
            "Article body with substantial prose that easily clears the quality "
            "floor in the fail signature detector so we exercise the web path.\n"
        )
        _write_txt(doc_dir, fname, web_text)

        meta = {
            "url": "https://example.com/article",
            "page_metadata": {"sitename": "Example"},
        }
        result = extract_metadata(doc_dir, meta)
        assert result.authors == ["Example"], f"sitename fallback failed for {fname}"


def test_extract_metadata_web_source_uses_page_metadata_when_present(tmp_path: Path) -> None:
    """Web source WITH validated page_metadata.authors still uses them."""
    from src.fetch.metadata import extract_metadata

    doc_dir = _make_doc_dir(tmp_path, "web_pm")
    web_text = (
        "# Some Article\n\nJump to content\n\nArticle body with enough prose "
        "to clear the quality floor in the fail-signature detector.\n"
    )
    _write_txt(doc_dir, "web.txt", web_text)

    meta = {
        "url": "https://example.com/article",
        "page_metadata": {"authors": ["Real Author Name"]},
    }
    result = extract_metadata(doc_dir, meta)
    assert result.authors == ["Real Author Name"]


def test_enrich_meta_json_force_bypasses_non_regression_guard(tmp_path: Path) -> None:
    """force=True overwrites old high-confidence with new low-confidence (cleanup mode)."""
    from src.fetch.metadata import enrich_meta_json

    doc_dir = _make_doc_dir(tmp_path, "force_doc")
    # pdf.txt that would produce confidence='none' (empty body, no signals)
    _write_txt(doc_dir, "pdf.txt", "Just a moment...\n")  # CF challenge → rejected
    meta_path = _write_meta(
        doc_dir,
        {
            "index": 1,
            "title": "x",
            "url": "https://example.com",
            "host": "example.com",
            "tier": "T5_web_blog",
            "fetch": {},
            "metadata": {
                "authors": ["Bogus Author"],
                "year": None,
                "short_cite": "Bogus",
                "confidence": "medium",  # bogus high-rank → guard would preserve
            },
        },
    )

    # Without force: guard preserves bogus author
    enrich_meta_json(meta_path, force=False)
    after_guard = json.loads(meta_path.read_text(encoding="utf-8"))
    assert after_guard["metadata"]["authors"] == ["Bogus Author"]

    # With force: bogus authors get wiped
    enrich_meta_json(meta_path, force=True)
    after_force = json.loads(meta_path.read_text(encoding="utf-8"))
    assert after_force["metadata"]["authors"] == []
    assert after_force["metadata"]["confidence"] == "none"
