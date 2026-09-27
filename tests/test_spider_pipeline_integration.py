"""Integration tests for the spider's per-source post-processing pipeline.

These tests pin the contract of ``FetchPipelineSpider.on_scraped_item``'s
artifact-producing surface — the static helpers that turn a raw fetch
into ingestable artifacts (``content_list.json``, ``meta.json``,
quarantine entries).

What's pinned:

1. **HTML path** — ``_write_structured_sidecar`` produces a
   ``content_list.json`` from raw HTML when ``_raw_html`` is on the item.
2. **NLM-only path** — ``_synthesize_from_nlm`` produces a
   ``content_list.json`` (or a quarantine entry) when only ``nlm.txt``
   is present.
3. **HTML wins over NLM** — when both run, the NLM path is a no-op
   because the HTML path already wrote the sidecar.
4. **Garbage handling** — bad NLM-only inputs are quarantined, never
   embedded.
5. **Failure isolation** — synthesizer crashes don't propagate up
   (the spider must finish processing the source).
6. **Idempotency** — re-running on the same dir doesn't rewrite a
   sidecar (matters for re-fetches on the cached path).

These are integration tests because they exercise the **real spider
methods** (not just the underlying synthesizer/extractor in isolation),
locking in the wiring across modules. A refactor that moves logic
without preserving the artifact contract will fail these.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from src.fetch.spider import FetchPipelineSpider

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_doc_dir(root: Path, name: str = "001__test_doc") -> Path:
    """Mimic the spider's per-source layout: <root>/<notebook>/<NNN__name>/."""
    doc = root / "test_notebook" / name
    doc.mkdir(parents=True)
    return doc


def _real_html(heading: str = "Test Article", body_paras: int = 6) -> str:
    """Substantive HTML so trafilatura doesn't filter it as boilerplate."""
    paras = "\n".join(
        f"<p>This is paragraph {i + 1} with enough text to pass the boilerplate "
        f"filter that trafilatura applies before extracting structured content "
        f"from a document. The article discusses market microstructure.</p>"
        for i in range(body_paras)
    )
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html><head><title>{heading}</title></head><body>
        <article>
        <h1>{heading}</h1>
        <h2>Introduction</h2>
        {paras}
        <h2>Conclusion</h2>
        <p>Concluding remarks with substantive text to clear the boilerplate
        filter and demonstrate the structured extraction pipeline working
        end-to-end on real-shaped HTML input.</p>
        </article>
        </body></html>""")


# ---------------------------------------------------------------------------
# 1. NLM-only path
# ---------------------------------------------------------------------------


class TestNlmOnlyPath:
    """Spider hook for sources where NotebookLM is the only signal."""

    def test_real_content_synthesizes_sidecar(self, tmp_path: Path) -> None:
        doc = _make_doc_dir(tmp_path)
        (doc / "nlm.txt").write_text(
            "# Real Article\n\n## Introduction\n\n"
            "The article discusses market microstructure. " * 30
        )

        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)

        sidecar = doc / "content_list.json"
        assert sidecar.exists(), "spider hook must produce content_list.json"
        items = json.loads(sidecar.read_text())
        assert items[0]["type"] == "text"
        assert items[0]["text_level"] == 1
        assert items[0]["text"] == "Real Article"

    def test_garbage_is_quarantined_not_embedded(self, tmp_path: Path) -> None:
        doc = _make_doc_dir(tmp_path)
        (doc / "nlm.txt").write_text("Access denied\nRay ID: abc123" + " filler" * 200)

        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)

        assert not (doc / "content_list.json").exists(), (
            "garbage must NOT yield a sidecar (would poison retrieval)"
        )
        manifest = tmp_path / "_quarantine_nlm_garbage" / "manifest.jsonl"
        assert manifest.exists()
        entry = json.loads(manifest.read_text().splitlines()[0])
        assert entry["reason"] in {"access_denied", "cloudflare", "temp_unavail"}

    def test_no_nlm_no_html_is_noop(self, tmp_path: Path) -> None:
        """PDF-only source dirs (no nlm.txt, no html) must not crash."""
        doc = _make_doc_dir(tmp_path, "001__pdf_only")
        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        assert not (doc / "content_list.json").exists()


# ---------------------------------------------------------------------------
# 2. HTML structured-sidecar path
# ---------------------------------------------------------------------------


class TestHtmlPath:
    """Spider hook for HTML fetches with raw markup available."""

    def test_html_extraction_writes_content_list(self, tmp_path: Path) -> None:
        doc = _make_doc_dir(tmp_path)
        (doc / "web.txt").write_text("text extraction placeholder")
        item: dict[str, Any] = {"_raw_html": _real_html()}

        FetchPipelineSpider._write_structured_sidecar(doc, item)

        sidecar = doc / "content_list.json"
        assert sidecar.exists(), "HTML path must produce content_list.json"
        items = json.loads(sidecar.read_text())
        assert items, "extracted content_list must not be empty"
        # Trafilatura/extract_structured produces typed items; at minimum
        # there's a heading-level text item somewhere.
        types = {it.get("type") for it in items if isinstance(it, dict)}
        assert "text" in types

    def test_no_raw_html_is_noop(self, tmp_path: Path) -> None:
        """Items missing _raw_html (e.g. failed fetches) leave the dir alone."""
        doc = _make_doc_dir(tmp_path)
        item: dict[str, Any] = {}
        FetchPipelineSpider._write_structured_sidecar(doc, item)
        assert not (doc / "content_list.json").exists()

    def test_html_extraction_does_not_raise_on_garbage_html(self, tmp_path: Path) -> None:
        """Malformed HTML must not crash — best-effort contract."""
        doc = _make_doc_dir(tmp_path)
        item: dict[str, Any] = {"_raw_html": "<html><body><p>not enough</body>"}
        # Must not raise — sidecar may or may not be written depending on
        # what trafilatura makes of the input, but the call must complete.
        FetchPipelineSpider._write_structured_sidecar(doc, item)


# ---------------------------------------------------------------------------
# 3. Combined paths (HTML + NLM both available)
# ---------------------------------------------------------------------------


class TestCombinedPaths:
    """When both HTML and NLM are available, HTML must win."""

    def test_html_path_winning_blocks_nlm_synthesis(self, tmp_path: Path) -> None:
        """HTML fetch wrote sidecar → NLM hook must not overwrite it.

        This pins the dispatch order in on_scraped_item: HTML first, NLM
        only fires when content_list.json doesn't yet exist.
        """
        doc = _make_doc_dir(tmp_path)
        (doc / "web.txt").write_text("placeholder")
        (doc / "nlm.txt").write_text("# NLM Source\n\nNLM-only content. " * 50)
        item: dict[str, Any] = {"_raw_html": _real_html()}

        # Mimics on_scraped_item's dispatch order
        FetchPipelineSpider._write_structured_sidecar(doc, item)
        before = (doc / "content_list.json").read_text()
        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        after = (doc / "content_list.json").read_text()

        assert before == after, "HTML-derived sidecar must be preserved; NLM is fallback only"

    def test_html_failure_falls_through_to_nlm(self, tmp_path: Path) -> None:
        """If HTML extraction returns nothing (no _raw_html), NLM kicks in."""
        doc = _make_doc_dir(tmp_path)
        (doc / "nlm.txt").write_text("# Salvaged Article\n\nNLM saved this fetch. " * 50)
        # No _raw_html → HTML path is no-op
        FetchPipelineSpider._write_structured_sidecar(doc, {})
        assert not (doc / "content_list.json").exists()

        # Now NLM kicks in
        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        assert (doc / "content_list.json").exists()
        items = json.loads((doc / "content_list.json").read_text())
        assert items[0]["text"] == "Salvaged Article"


# ---------------------------------------------------------------------------
# 4. Failure isolation + idempotency
# ---------------------------------------------------------------------------


class TestFailureContract:
    """Best-effort guarantees: synthesis failures must not abort fetch."""

    def test_synthesizer_exception_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crashing synthesizer must not propagate up."""
        doc = _make_doc_dir(tmp_path)
        (doc / "nlm.txt").write_text("# Title\n\nReal body content. " * 50)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("synthetic test failure")

        import scripts.nlm_to_content_list as nlm_mod

        monkeypatch.setattr(nlm_mod, "synthesize_from_nlm", _boom)

        # Must not raise.
        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        assert not (doc / "content_list.json").exists()

    def test_html_extractor_import_failure_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If structured-extraction module fails to load, fetch must complete."""
        doc = _make_doc_dir(tmp_path)
        item: dict[str, Any] = {"_raw_html": _real_html()}

        # Force the lazy import to fail.
        import builtins

        real_import = builtins.__import__

        def _import_wrapper(name: str, *args: object, **kwargs: object) -> object:
            if name == "src.fetch.structured":
                raise ImportError("synthetic — module unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import_wrapper)
        FetchPipelineSpider._write_structured_sidecar(doc, item)
        assert not (doc / "content_list.json").exists()


class TestIdempotency:
    """Re-fetches must not overwrite or duplicate artifacts."""

    def test_nlm_synthesizer_idempotent_on_re_fetch(self, tmp_path: Path) -> None:
        doc = _make_doc_dir(tmp_path)
        (doc / "nlm.txt").write_text("# Title\n\nReal body content for the synthesizer test " * 50)

        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        first = (doc / "content_list.json").read_text()

        # Simulate a second fetch that updated nlm.txt
        (doc / "nlm.txt").write_text(
            "# Different\n\nDifferent body content for the synthesizer test " * 50
        )
        FetchPipelineSpider._synthesize_from_nlm(doc, tmp_path)
        second = (doc / "content_list.json").read_text()
        assert first == second, "second call must not overwrite"

    def test_html_re_run_overwrites_with_same_content(self, tmp_path: Path) -> None:
        """HTML extractor doesn't have a skip-if-exists guard — running it
        twice with the same input gives the same output (deterministic).
        """
        doc = _make_doc_dir(tmp_path)
        item: dict[str, Any] = {"_raw_html": _real_html()}

        FetchPipelineSpider._write_structured_sidecar(doc, item)
        first = (doc / "content_list.json").read_text()
        FetchPipelineSpider._write_structured_sidecar(doc, item)
        second = (doc / "content_list.json").read_text()
        assert first == second
