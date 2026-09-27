"""Unit tests for src.fetch.page_metadata — trafilatura + citation tag harvester."""

from __future__ import annotations


def test_harvest_returns_authors_from_meta_author_tag() -> None:
    """Generic <meta name="author"> picked up via trafilatura."""
    from src.fetch.page_metadata import harvest_page_metadata

    html = (
        "<html><head>"
        '<meta name="author" content="Jane Doe">'
        "<title>A Blog Post</title>"
        "</head><body><article><p>...</p></article></body></html>"
    )
    result = harvest_page_metadata(html)
    assert result.get("authors") == ["Jane Doe"]


def test_harvest_returns_authors_from_og_article_author() -> None:
    """OpenGraph 'article:author' picked up via trafilatura."""
    from src.fetch.page_metadata import harvest_page_metadata

    html = (
        "<html><head>"
        '<meta property="article:author" content="Alice Smith">'
        "<title>Some Article</title>"
        "</head><body><article><h1>Some Article</h1>"
        "<p>Substantial article body for trafilatura to recognize as content. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>"
        "</article></body></html>"
    )
    result = harvest_page_metadata(html)
    assert "Alice Smith" in (result.get("authors") or [])


def test_harvest_returns_none_for_author_when_absent() -> None:
    """No author signals → no 'authors' key (or empty)."""
    from src.fetch.page_metadata import harvest_page_metadata

    html = "<html><head><title>No author here</title></head><body><p>x</p></body></html>"
    result = harvest_page_metadata(html)
    assert not result.get("authors")


def test_harvest_preserves_citation_doi_from_classify_helper() -> None:
    """citation_doi from scholarly meta tags must survive (citation-tag fallback)."""
    from src.fetch.page_metadata import harvest_page_metadata

    html = (
        "<html><head>"
        '<meta name="citation_doi" content="10.1000/foo.bar">'
        '<meta name="citation_author" content="Bob Schmoe">'
        "</head><body>x</body></html>"
    )
    result = harvest_page_metadata(html)
    assert result.get("doi") == "10.1000/foo.bar"


def test_harvest_splits_multi_author_semicolon() -> None:
    """Trafilatura returns 'A; B; C'; harvester splits on semicolons."""
    from src.fetch.page_metadata import _split_author_string

    assert _split_author_string("Alice Smith; Bob Jones; Carol Lee") == [
        "Alice Smith",
        "Bob Jones",
        "Carol Lee",
    ]


def test_harvest_keeps_comma_inside_single_author() -> None:
    """'Smith, J.' is one author when no semicolon is present."""
    from src.fetch.page_metadata import _split_author_string

    assert _split_author_string("Smith, J.") == ["Smith, J."]


def test_harvest_returns_empty_on_invalid_html() -> None:
    """Empty / garbage HTML must not raise."""
    from src.fetch.page_metadata import harvest_page_metadata

    assert isinstance(harvest_page_metadata(""), dict)
    assert isinstance(harvest_page_metadata("not html"), dict)
