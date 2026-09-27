# long-ok-file
"""Unit tests for fetch-time URL/DOI dedup, classifier fix, and metadata harvest.

Tests the primitives extracted from the fetch pipeline into ``src/fetch/``:
  - ``canonical_url()`` — URL normalization for dedup keys
  - ``classify()`` patch — URL recovery from title for SourceType.UNKNOWN
  - ``_harvest_page_metadata()`` — citation_* meta + JSON-LD extraction
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Test canonical_url
# ---------------------------------------------------------------------------


class TestCanonicalUrl:
    """URL normalization for dedup comparison."""

    def testcanonical_url_arxiv_collapses_versions(self) -> None:
        """arxiv URL variants (abs/pdf/html, versioned, .pdf extension) collapse to one key."""
        from src.fetch.classify import canonical_url

        v1 = canonical_url(_src(url="https://arxiv.org/abs/2204.12345v2"))
        v2 = canonical_url(_src(url="https://arxiv.org/pdf/2204.12345.pdf"))
        v3 = canonical_url(_src(url="https://arxiv.org/abs/2204.12345"))
        v4 = canonical_url(_src(url="http://arxiv.org/html/2204.12345v1"))

        assert v1 == "arxiv.org/abs/2204.12345"
        assert v1 == v2 == v3 == v4

    def testcanonical_url_6digit_id_not_collapsed_to_5digit(self) -> None:
        """A 6-digit arxiv suffix must not key the same as its 5-digit prefix.

        The old ``[0-9]{4,5}`` suffix truncated 2403.123456 -> 2403.12345,
        colliding a distinct paper onto another and dropping it as a dup.
        """
        from src.fetch.classify import canonical_url

        six = canonical_url(_src(url="https://arxiv.org/abs/2403.123456"))
        five = canonical_url(_src(url="https://arxiv.org/abs/2403.12345"))
        assert six == "arxiv.org/abs/2403.123456"
        assert six != five

    def testcanonical_url_strips_query_and_fragment(self) -> None:
        """Query params and fragments are dropped; scheme+host lowercased."""
        from src.fetch.classify import canonical_url

        result = canonical_url(
            _src(
                url="https://Example.COM/path/to/page?utm_source=x&y=2#section-3",
            )
        )
        assert result == "https://example.com/path/to/page"

    def testcanonical_url_strips_trailing_slash(self) -> None:
        """Trailing slash on path is stripped."""
        from src.fetch.classify import canonical_url

        a = canonical_url(_src(url="https://example.com/path/"))
        b = canonical_url(_src(url="https://example.com/path"))
        assert a == b == "https://example.com/path"

    def testcanonical_url_recovers_unknown_source_url_from_title(self) -> None:
        """SourceType.UNKNOWN with URL in title field is recovered and normalized."""
        from src.fetch.classify import canonical_url

        src = {
            "url": "",
            "title": "Check https://arxiv.org/abs/2508.04344 for details",
            "type": "SourceType.UNKNOWN",
        }
        result = canonical_url(src)
        assert result == "arxiv.org/abs/2508.04344"

    def testcanonical_url_returns_empty_for_pasted_text(self) -> None:
        """Sources without URL (pasted text / NLM-uploaded PDF) return empty string."""
        from src.fetch.classify import canonical_url

        assert canonical_url(_src(url="", title="Some notes", type="SourceType.PASTED_TEXT")) == ""
        assert canonical_url(_src(url="", title="Some notes", type="SourceType.MARKDOWN")) == ""
        # UNKNOWN with no URL in title
        assert canonical_url(_src(url="", title="Just a title", type="SourceType.UNKNOWN")) == ""
        # Empty url field, non-empty but no URL in title
        assert (
            canonical_url(_src(url="", title="A paper about markets", type="SourceType.UNKNOWN"))
            == ""
        )

    def testcanonical_url_ssrn_normalizes_abstract_id(self) -> None:
        """SSRN URLs with abstract_id query param normalize to /abstract=<n> form."""
        from src.fetch.classify import canonical_url

        a = canonical_url(
            _src(
                url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1234567",
            )
        )
        b = canonical_url(
            _src(
                url="https://ssrn.com/abstract=1234567",
            )
        )
        assert a == b == "https://ssrn.com/abstract=1234567"

    def testcanonical_url_ssrn_drops_other_query_params(self) -> None:
        """SSRN URLs without abstract_id just drop query string entirely."""
        from src.fetch.classify import canonical_url

        result = canonical_url(
            _src(
                url="https://ssrn.com/sol3/papers.cfm?download=yes",
            )
        )
        assert result == "https://ssrn.com/sol3/papers.cfm"

    def testcanonical_url_researchgate_trims_extra_path(self) -> None:
        """ResearchGate URLs keep only /publication/<digits>_<slug>."""
        from src.fetch.classify import canonical_url

        a = canonical_url(
            _src(
                url="https://www.researchgate.net/publication/12345_Some_Title/citations",
            )
        )
        b = canonical_url(
            _src(
                url="https://www.researchgate.net/publication/12345_Some_Title/figures?lo=1",
            )
        )
        base = canonical_url(
            _src(
                url="https://www.researchgate.net/publication/12345_Some_Title",
            )
        )
        assert a == b == base
        assert base == "https://www.researchgate.net/publication/12345_some_title"

    def testcanonical_url_doi_lowercases_suffix(self) -> None:
        """doi.org URLs lowercase the DOI suffix path."""
        from src.fetch.classify import canonical_url

        result = canonical_url(
            _src(
                url="https://doi.org/10.1234/AbCdEf",
            )
        )
        assert result == "https://doi.org/10.1234/abcdef"


# ---------------------------------------------------------------------------
# Test classify() patch — UNKNOWN source URL recovery
# ---------------------------------------------------------------------------


class TestClassifyUnknownUrlRecovery:
    """classify() now recovers URLs from title for SourceType.UNKNOWN sources."""

    def test_classify_recovers_arxiv_from_unknown_title(self) -> None:
        """UNKNOWN type with arxiv URL in title → T0_arxiv, not T5_web_blog."""
        from src.fetch.classify import classify

        src = {
            "type": "SourceType.UNKNOWN",
            "url": "",
            "title": "Some paper https://arxiv.org/abs/2508.04344",
        }
        assert classify(src) == "T0_arxiv"

    def test_classify_recovers_ssrn_from_unknown_title(self) -> None:
        """UNKNOWN type with SSRN URL in title → T2_ssrn."""
        from src.fetch.classify import classify

        src = {
            "type": "SourceType.UNKNOWN",
            "url": "",
            "title": "SSRN paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=123",
        }
        assert classify(src) == "T2_ssrn"

    def test_classify_recovers_researchgate_from_unknown_title(self) -> None:
        """UNKNOWN type with RG URL in title → T4_researchgate."""
        from src.fetch.classify import classify

        src = {
            "type": "SourceType.UNKNOWN",
            "url": "",
            "title": "RG https://www.researchgate.net/publication/12345_Title",
        }
        assert classify(src) == "T4_researchgate"

    def test_classify_falls_through_for_unknown_without_url(self) -> None:
        """UNKNOWN type with no URL in title falls through to T5_web_blog (existing behaviour)."""
        from src.fetch.classify import classify

        src = {
            "type": "SourceType.UNKNOWN",
            "url": "",
            "title": "A paper title without any URL",
        }
        # Falls through to the catch-all T5_web_blog
        assert classify(src) == "T5_web_blog"

    def test_classify_pdf_with_url_from_title_still_generic_pdf(self) -> None:
        """UNKNOWN type with a .pdf URL in title → T3_generic_pdf."""
        from src.fetch.classify import classify

        src = {
            "type": "SourceType.UNKNOWN",
            "url": "",
            "title": "Paper https://example.com/download/paper.pdf",
        }
        # url recovered from title → should route through the PDF-with-URL path
        assert classify(src) == "T3_generic_pdf"


# ---------------------------------------------------------------------------
# Test _harvest_page_metadata
# ---------------------------------------------------------------------------


class TestHarvestPageMetadata:
    """HTML metadata extraction: citation_* meta tags + JSON-LD ScholarlyArticle."""

    def test_harvest_page_metadata_extracts_citation_doi(self) -> None:
        """Extracts citation_doi from <meta name=\"citation_doi\">."""
        from src.fetch.classify import _harvest_page_metadata

        html = """<html><head>
        <meta name="citation_doi" content="10.1234/journal.foo.2025.01">
        <meta name="citation_title" content="A Study of Markets">
        <meta name="citation_author" content="Alice Smith">
        <meta name="citation_author" content="Bob Jones">
        <meta name="citation_publication_date" content="2025/03/15">
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        assert result["doi"] == "10.1234/journal.foo.2025.01"
        assert result["title"] == "A Study of Markets"
        assert result["authors"] == ["Alice Smith", "Bob Jones"]
        assert result["published"] == "2025/03/15"

    def test_harvest_page_metadata_jsonld_scholarly_article(self) -> None:
        """Extracts fields from a JSON-LD ScholarlyArticle block."""
        from src.fetch.classify import _harvest_page_metadata

        ld = {
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "name": "Quantum Trading Strategies",
            "author": [{"name": "Carol Cheng"}, {"name": "Dan Wu"}],
            "datePublished": "2026-01-15",
        }
        html = f"""<html><head>
        <script type="application/ld+json">
        {json.dumps(ld)}
        </script>
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        assert result["title"] == "Quantum Trading Strategies"
        assert result["authors"] == ["Carol Cheng", "Dan Wu"]
        assert result["published"] == "2026-01-15"

    def test_harvest_page_metadata_jsonld_overrides_meta_tags_on_conflict(self) -> None:
        """When both meta tags and JSON-LD present, JSON-LD wins on conflict."""
        from src.fetch.classify import _harvest_page_metadata

        ld = {
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "name": "The Real Title",
        }
        html = f"""<html><head>
        <meta name="citation_title" content="Wrong Title">
        <script type="application/ld+json">
        {json.dumps(ld)}
        </script>
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        # JSON-LD "name" wins over meta "citation_title"
        assert result["title"] == "The Real Title"

    def test_harvest_page_metadata_ignores_non_scholarly_jsonld(self) -> None:
        """JSON-LD blocks with @type != ScholarlyArticle are ignored."""
        from src.fetch.classify import _harvest_page_metadata

        ld = {
            "@context": "https://schema.org",
            "@type": "WebPage",
            "name": "Not a paper",
        }
        html = f"""<html><head>
        <meta name="citation_title" content="Actual Paper Title">
        <script type="application/ld+json">
        {json.dumps(ld)}
        </script>
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        # Non-ScholarlyArticle JSON-LD is ignored; meta tag is used
        assert result["title"] == "Actual Paper Title"

    def test_harvest_page_metadata_returns_empty_dict_on_no_metadata(self) -> None:
        """Returns {} when HTML has no relevant meta tags or JSON-LD."""
        from src.fetch.classify import _harvest_page_metadata

        assert _harvest_page_metadata("<html><body>Just text, no metadata</body></html>") == {}
        assert _harvest_page_metadata("") == {}
        assert _harvest_page_metadata("<html></html>") == {}

    def test_harvest_page_metadata_handles_malformed_jsonld(self) -> None:
        """Invalid JSON in a script tag is silently ignored."""
        from src.fetch.classify import _harvest_page_metadata

        html = """<html><head>
        <meta name="citation_doi" content="10.1234/foo">
        <script type="application/ld+json">
        {not valid json!!!
        </script>
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        # Malformed JSON-LD is ignored; meta tags still extracted
        assert result["doi"] == "10.1234/foo"

    def test_harvest_page_metadata_handles_jsonld_single_author_string(self) -> None:
        """JSON-LD author as a single string (not list of objects) is handled."""
        from src.fetch.classify import _harvest_page_metadata

        ld = {
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "author": "Single Author Name",
        }
        html = f"""<html><head>
        <script type="application/ld+json">
        {json.dumps(ld)}
        </script>
        </head><body>Content</body></html>"""

        result = _harvest_page_metadata(html)
        assert result["authors"] == ["Single Author Name"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _src(url: str = "", title: str = "", type: str = "SourceType.WEB") -> dict:  # noqa: A002
    """Minimal source dict for testing."""
    return {"url": url, "title": title, "type": type}
