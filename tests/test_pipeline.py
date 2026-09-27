# long-ok-file
"""Integration tests for src/pipeline/runner.py — run_pipeline orchestrator.

Uses explicit function injection (_fetch_fn / _parse_fn kwargs) rather than
monkeypatch so tests work with any multiprocessing start method (fork/spawn/
forkserver). All stub functions are module-level for pickle-ability.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Module-level stubs — must be at module level for multiprocessing pickling
# ---------------------------------------------------------------------------

FAKE_NLM_TEXT = "This is sample NLM fulltext content for testing.\n" * 5
FAKE_PDF_TEXT = "This is parsed PDF text content.\n" * 3


def _slug_stub(s: str) -> str:
    """Minimal slug for test fixtures."""
    import re

    out = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return out[:60] or "untitled"


def _stub_fetch_one_source(
    src: dict,
    out_root: Path,
    *,
    no_cffi: bool = False,
    pace_fn: object = None,
    nb_id: str | None = None,
) -> tuple[dict, list[Path]]:
    """Writes fake nlm.txt + optional source.pdf based on tier."""
    idx = src.get("index", 0)
    title = src.get("title", "untitled")
    tier = src.get("_tier", "T6_nlm_native")

    d = out_root / f"{idx:03d}__{_slug_stub(title)}"
    d.mkdir(parents=True, exist_ok=True)

    nlm_path = d / "nlm.txt"
    nlm_path.write_text(FAKE_NLM_TEXT)
    written = [nlm_path]

    pdf_tiers = {"T0_arxiv", "T3_generic_pdf"}
    if tier in pdf_tiers:
        pdf_path = d / "source.pdf"
        pdf_path.write_text("%PDF-1.4\n%stub\n")
        written.append(pdf_path)

    meta = {
        "index": idx,
        "id": src["id"],
        "title": title,
        "url": src.get("url", ""),
        "host": "",
        "tier": tier,
        "fetch": {"nlm": {"ok": True, "bytes": len(FAKE_NLM_TEXT)}},
    }
    if tier in pdf_tiers:
        meta["fetch"]["pdf"] = {"ok": True, "pdf_bytes": 42}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    return meta, written


def _stub_parse_pipeline_only(path: Path) -> str:
    """Returns fake text after a tiny delay (simulates parse work)."""
    time.sleep(0.01)
    out_path = path.parent / "pdf.txt"
    out_path.write_text(FAKE_PDF_TEXT)
    return FAKE_PDF_TEXT


def _stub_parse_fails(path: Path) -> str:
    """Always raises — simulates a parse failure."""
    raise RuntimeError("simulated parse failure")


def _stub_parse_must_not_be_called(path: Path) -> str:
    """Raises if invoked — used to assert parse_worker skips when pdf.txt exists."""
    raise AssertionError(f"parse_fn called for {path} but pdf.txt should already exist")


def _stub_fetch_with_existing_pdf_txt(
    src: dict,
    out_root: Path,
    *,
    no_cffi: bool = False,
    pace_fn: object = None,
    nb_id: str | None = None,
) -> tuple[dict, list[Path]]:
    """Like _stub_fetch_all_pdfs but also writes pdf.txt at fetch time.

    Simulates the restart-after-prior-parse scenario: source dir has a valid
    pdf.txt from a previous ragctl run.
    """
    meta, written = _stub_fetch_all_pdfs(
        src,
        out_root,
        no_cffi=no_cffi,
        pace_fn=pace_fn,
        nb_id=nb_id,
    )
    pdf_path = next(p for p in written if p.name == "source.pdf")
    (pdf_path.parent / "pdf.txt").write_text("pre-existing parsed text")
    return meta, written


def _stub_fetch_all_pdfs(
    src: dict,
    out_root: Path,
    *,
    no_cffi: bool = False,
    pace_fn: object = None,
    nb_id: str | None = None,
) -> tuple[dict, list[Path]]:
    """Like _stub_fetch_one_source but always writes a PDF (ignores tier)."""
    idx = src.get("index", 0)
    title = src.get("title", "untitled")

    d = out_root / f"{idx:03d}__{_slug_stub(title)}"
    d.mkdir(parents=True, exist_ok=True)

    nlm_path = d / "nlm.txt"
    nlm_path.write_text(FAKE_NLM_TEXT)
    written = [nlm_path]

    pdf_path = d / "source.pdf"
    pdf_path.write_text("%PDF-1.4\n%stub\n")
    written.append(pdf_path)

    meta = {
        "index": idx,
        "id": src["id"],
        "title": title,
        "url": src.get("url", ""),
        "host": "",
        "tier": src.get("_tier", "T0_arxiv"),
        "fetch": {
            "nlm": {"ok": True, "bytes": len(FAKE_NLM_TEXT)},
            "pdf": {"ok": True, "pdf_bytes": 42},
        },
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta, written


def _stub_fetch_selective_fail(
    src: dict,
    out_root: Path,
    *,
    no_cffi: bool = False,
    pace_fn: object = None,
    nb_id: str | None = None,
) -> tuple[dict, list[Path]]:
    """Source s2 always fails, others succeed."""
    if src.get("id") == "s2":
        raise RuntimeError("simulated fetch failure")
    return _stub_fetch_one_source(src, out_root, no_cffi=no_cffi, pace_fn=pace_fn, nb_id=nb_id)


def _stub_parse_always_dies(path: Path) -> str:
    """Always raises ConnectionRefusedError — verifies parse_fail without retry."""
    raise ConnectionRefusedError("simulated daemon permanently dead")


# ---------------------------------------------------------------------------
# Helper: build notebook fixture with embedded source lists
# ---------------------------------------------------------------------------


def _make_notebooks(specs: list[tuple[str, str, str, list[dict]]]) -> list[dict]:
    """Build notebook dicts with embedded _sources for test fixture."""
    notebooks = []
    for nb_id, tag, coll, sources in specs:
        nb = {"id": nb_id, "tag": tag, "collection": coll, "_sources": sources}
        notebooks.append(nb)
    return notebooks


def _make_source(idx: int, src_id: str, title: str, tier: str) -> dict:
    return {
        "index": idx,
        "id": src_id,
        "title": title,
        "url": f"http://example.com/{idx}",
        "_tier": tier,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_notebooks() -> list[dict]:
    """3 notebooks: 2 trading, 1 meta. Mix of PDF and web-only sources."""
    return _make_notebooks(
        [
            (
                "u1",
                "alpha",
                "trading",
                [
                    _make_source(1, "s1", "Alpha Paper 1", "T0_arxiv"),
                    _make_source(2, "s2", "Alpha Paper 2", "T0_arxiv"),
                    _make_source(3, "s3", "Alpha Blog", "T5_web_blog"),
                ],
            ),
            (
                "u2",
                "beta",
                "trading",
                [
                    _make_source(1, "s4", "Beta Blog 1", "T5_web_blog"),
                    _make_source(2, "s5", "Beta Blog 2", "T5_web_blog"),
                ],
            ),
            (
                "u3",
                "gamma",
                "meta",
                [
                    _make_source(1, "s6", "Gamma Paper", "T0_arxiv"),
                ],
            ),
        ]
    )


@pytest.fixture()
def one_notebook_two_sources() -> list[dict]:
    """Single notebook with 2 sources for failure tests."""
    return _make_notebooks(
        [
            (
                "u1",
                "alpha",
                "trading",
                [
                    _make_source(1, "s1", "Good Source", "T0_arxiv"),
                    _make_source(2, "s2", "Bad Source", "T0_arxiv"),
                ],
            ),
        ]
    )


@pytest.fixture()
def meta_notebook() -> list[dict]:
    """Single meta-collection notebook."""
    return _make_notebooks(
        [
            (
                "um",
                "meta_nb",
                "meta",
                [
                    _make_source(1, "s1", "Meta Source", "T0_arxiv"),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Pipeline stage tests assert legacy `src.pipeline.stage` subprocess "
    "calls. Post-flatten, stage/contextualize/ingest are no-ops in the "
    "orchestrator (run per-collection, not per-notebook). Rewrite or "
    "delete these tests when the orchestrator is fully retired for the "
    "flattened layout."
)
class TestRunnerDrainsNotebookFixture:
    """End-to-end: fetch+parse streaming, audit barrier, sequential post-barrier."""

    def test_runner_drains_3_notebook_fixture(
        self, three_notebooks: list[dict], tmp_path: Path
    ) -> None:
        """3 notebooks with mixed sources — all reach stage_ready, audit runs
        once, sequential post-barrier invoked for all 3."""
        from src.pipeline.runner import run_pipeline

        # Create fake audit manifest so _run_audit succeeds
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "cleanup_manifest.json").write_text(
            '{"summary": {"n_quarantine_poisoned": 0, '
            '"n_quarantine_dupes": 0, "n_canonical_groups": 0}}'
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                three_notebooks,
                stages=["fetch", "parse_pipeline", "stage", "audit", "contextualize", "ingest"],
                root=tmp_path,
                parallel=2,
                _fetch_fn=_stub_fetch_one_source,
                _parse_fn=_stub_parse_pipeline_only,
            )

        assert rc == 0

        # Collect subprocess.run calls (stage/ctx/ingest/audit)
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]

        # Audit scripts should have been called
        audit_calls = [
            c for c in call_strs if "audit" in c.lower() or "build_cleanup_manifest" in c
        ]
        assert len(audit_calls) >= 1, f"audit not called; calls: {call_strs}"

        # Stage should be called for all 3 notebooks
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 3, f"expected 3 stage calls, got {stage_calls}"

        # Contextualize for all 3
        ctx_calls = [c for c in call_strs if "contextualize" in c]
        assert len(ctx_calls) == 3, f"expected 3 contextualize calls, got {len(ctx_calls)}"

        # Ingest for only 2 (meta skipped)
        ingest_only = [c for c in call_strs if "src.ingest.ingest" in c]
        assert len(ingest_only) == 2, f"expected 2 ingest calls, got {ingest_only}"

        # Verify source files were written
        for nb in three_notebooks:
            sources_dir = tmp_path / "sources" / nb["tag"]
            assert sources_dir.exists(), f"sources dir missing for {nb['tag']}"

    def test_runner_handles_fetch_failure(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """One bad source doesn't block the rest of notebook from proceeding."""
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                _fetch_fn=_stub_fetch_selective_fail,
                _parse_fn=_stub_parse_pipeline_only,
            )

        # Should still succeed overall (fetch failure is non-fatal)
        assert rc == 0

        # Stage should still be called for the notebook
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 1, (
            f"expected 1 stage call despite fetch failure, got {stage_calls}"
        )

    def test_runner_handles_parse_failure(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """A parse failure decrements pdfs_expected so stage_ready is still reachable."""
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                _fetch_fn=_stub_fetch_all_pdfs,
                _parse_fn=_stub_parse_fails,
            )

        # Should still succeed (parse failures are non-fatal)
        assert rc == 0

        # Stage should still be called
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 1, (
            f"expected 1 stage call despite parse failure, got {stage_calls}"
        )

    def test_runner_skips_meta_collection_ingest(
        self, meta_notebook: list[dict], tmp_path: Path
    ) -> None:
        """Meta-collection notebooks should never have ingest invoked."""
        from src.pipeline.runner import run_pipeline

        # Create fake audit manifest
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "cleanup_manifest.json").write_text(
            '{"summary": {"n_quarantine_poisoned": 0, '
            '"n_quarantine_dupes": 0, "n_canonical_groups": 0}}'
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                meta_notebook,
                stages=["fetch", "parse_pipeline", "stage", "audit", "contextualize", "ingest"],
                root=tmp_path,
                _fetch_fn=_stub_fetch_one_source,
                _parse_fn=_stub_parse_pipeline_only,
            )

        assert rc == 0
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]

        # Ingest should NOT appear for meta-collection notebook
        ingest_calls = [c for c in call_strs if "src.ingest.ingest" in c]
        assert len(ingest_calls) == 0, (
            f"ingest should be skipped for meta collection, got: {ingest_calls}"
        )

    def test_two_notebooks_with_shared_url_dedupe_to_one_parse(self, tmp_path: Path) -> None:
        """2 notebooks share a canonical URL → parse called 3 times, not 4.

        Both notebooks reach stage_ready and the duplicate's source dir
        contains _dedup_pointer.json instead of content files.
        """
        from src.pipeline.runner import run_pipeline

        shared_url = "https://arxiv.org/abs/2204.12345"
        nb1 = _make_notebooks(
            [
                (
                    "u1",
                    "alpha",
                    "trading",
                    [
                        _make_source(1, "s1", "Alpha Paper 1", "T0_arxiv"),
                        _make_source(2, "s2", "Alpha Paper 2", "T0_arxiv"),
                    ],
                ),
            ]
        )
        nb2 = _make_notebooks(
            [
                (
                    "u2",
                    "beta",
                    "trading",
                    [
                        _make_source(1, "s3", "Beta Paper 1", "T0_arxiv"),
                        _make_source(2, "s4", "Beta Paper 2", "T0_arxiv"),
                    ],
                ),
            ]
        )
        # Override URLs so s1 and s3 share canonical URL; s2 and s4 are unique
        nb1[0]["_sources"][0]["url"] = shared_url
        nb1[0]["_sources"][1]["url"] = "https://example.com/alpha-unique"
        nb2[0]["_sources"][0]["url"] = shared_url
        nb2[0]["_sources"][1]["url"] = "https://example.com/beta-unique"

        notebooks = nb1 + nb2

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                notebooks,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=2,
                _fetch_fn=_stub_fetch_all_pdfs,
                _parse_fn=_stub_parse_pipeline_only,
            )

        assert rc == 0

        # Both notebooks should have stage called
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 2, f"expected 2 stage calls, got {stage_calls}"

        # Count pdf.txt files across both notebooks — expect 3, not 4
        all_pdf_txts = list(tmp_path.rglob("pdf.txt"))
        assert len(all_pdf_txts) == 3, (
            f"expected 3 pdf.txt (3 unique URLs), got {len(all_pdf_txts)}: "
            f"{[str(p) for p in all_pdf_txts]}"
        )

        # The duplicate source (s3 in beta) should have _dedup_pointer.json
        # and no pdf.txt of its own.
        dup_dirs = list((tmp_path / "sources" / "beta").glob("*/"))
        dedup_ptrs = [d / "_dedup_pointer.json" for d in dup_dirs]
        dedup_found = [p for p in dedup_ptrs if p.exists()]
        assert len(dedup_found) == 1, (
            f"expected 1 _dedup_pointer.json in beta, got {len(dedup_found)}"
        )
        ptr = json.loads(dedup_found[0].read_text())
        assert ptr["canonical_nb_tag"] == "alpha"
        assert ptr["canonical_url"] == "arxiv.org/abs/2204.12345"

        # The canonical source dir in alpha should have pdf.txt
        canon_dirs = list((tmp_path / "sources" / "alpha").glob("*/"))
        canon_pdfs = [d / "pdf.txt" for d in canon_dirs if (d / "pdf.txt").exists()]
        assert len(canon_pdfs) == 2, f"alpha should have 2 pdf.txt, got {len(canon_pdfs)}"

    def test_runner_dry_run_no_subprocess(
        self, three_notebooks: list[dict], tmp_path: Path
    ) -> None:
        """Dry-run flag prints plan and exits without doing any real work."""
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            rc = run_pipeline(
                three_notebooks,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                dry_run=True,
            )

        assert rc == 0
        # No subprocesses should have been spawned
        mock_run.assert_not_called()


class TestRunnerStageSelection:
    """Verify that stage filtering works correctly."""

    def test_fetch_only(self, three_notebooks: list[dict], tmp_path: Path) -> None:
        """Only fetch stage — no parse, no post-barrier, no audit."""
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                three_notebooks,
                stages=["fetch"],
                root=tmp_path,
                _fetch_fn=_stub_fetch_one_source,
            )

        assert rc == 0
        # No subprocess.run calls should happen for stage/audit/ctx/ingest
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 0, f"stage should not be called with fetch-only: {stage_calls}"

    def test_parse_pipeline_only_without_fetch(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """parse_pipeline without fetch — nothing to parse (empty source dirs)."""
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)

            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["parse_pipeline"],
                root=tmp_path,
            )

        # Nothing to parse — should still succeed
        assert rc == 0


@pytest.mark.skip(reason="See TestRunnerDrainsNotebookFixture: legacy stage assertions.")
class TestParseWorkerDaemonRestart:
    """parse_worker daemon-dead detection + auto-restart + circuit breaker."""

    def test_daemon_dead_results_in_parse_fail_no_retry(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """ConnectionRefusedError → immediate parse_fail; worker does NOT restart daemon.

        The worker-side restart path was removed because it caused cascading
        daemon respawns when transient errors triggered repeated restarts.
        Daemon lifecycle is owned by the main process. If the daemon is dead,
        parses fail visibly until the next ragctl run.
        """
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                _fetch_fn=_stub_fetch_all_pdfs,
                _parse_fn=_stub_parse_always_dies,
            )

        assert rc == 0
        sources_dir = tmp_path / "sources" / "alpha"
        pdf_txts = list(sources_dir.glob("*/pdf.txt"))
        assert len(pdf_txts) == 0, f"expected 0 pdf.txt (worker no longer retries), got {pdf_txts}"

    def test_stage_still_runs_when_all_parses_fail(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """All parses fail → stage still runs because pdfs_expected decrements on parse_fail.

        Coordinator decrements pdfs_expected on parse_fail so stage_ready is
        reachable even when no PDF parses successfully.
        """
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                _fetch_fn=_stub_fetch_all_pdfs,
                _parse_fn=_stub_parse_always_dies,
            )

        assert rc == 0
        calls = [c.args[0] for c in mock_run.call_args_list]
        call_strs = [" ".join(c) if isinstance(c, list) else str(c) for c in calls]
        stage_calls = [c for c in call_strs if "src.pipeline.stage" in c]
        assert len(stage_calls) == 1, (
            f"stage should still run after all parses fail, got {stage_calls}"
        )

    def test_parse_worker_skips_when_pdf_txt_exists(
        self, one_notebook_two_sources: list[dict], tmp_path: Path
    ) -> None:
        """Pre-existing pdf.txt → parse_fn never called; coordinator still emits parse_ok.

        Lets ragctl restarts resume cleanly without re-parsing already-done sources
        (~30s GPU saved per source).
        """
        from src.pipeline.runner import run_pipeline

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            rc = run_pipeline(
                one_notebook_two_sources,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                _fetch_fn=_stub_fetch_with_existing_pdf_txt,
                _parse_fn=_stub_parse_must_not_be_called,
            )

        assert rc == 0
        sources_dir = tmp_path / "sources" / "alpha"
        pdf_txts = list(sources_dir.glob("*/pdf.txt"))
        assert len(pdf_txts) == 2
        for p in pdf_txts:
            assert p.read_text() == "pre-existing parsed text", "existing pdf.txt clobbered"


# ---------------------------------------------------------------------------
# --retry-failed flag
# ---------------------------------------------------------------------------


def _write_meta_json(
    d: Path,
    *,
    idx: int,
    src_id: str,
    title: str,
    url: str,
    web_ok: bool | None = None,
    pdf_ok: bool | None = None,
) -> None:
    """Write a meta.json with the given fetch.web/pdf ok status."""
    fetch: dict[str, dict[str, object]] = {"nlm": {"ok": True, "bytes": 100}}
    if web_ok is not None:
        fetch["web"] = {
            "ok": web_ok,
            "bytes": 500,
            "reason": "ok:http" if web_ok else "all_tiers_failed",
        }
    if pdf_ok is not None:
        fetch["pdf"] = {
            "ok": pdf_ok,
            "bytes": 1000,
            "reason": "pdf:ok" if pdf_ok else "pdf:too_small",
        }
    (d / "meta.json").write_text(
        json.dumps(
            {
                "index": idx,
                "id": src_id,
                "title": title,
                "url": url,
                "host": "example.com",
                "tier": "T0_arxiv",
                "fetch": fetch,
            }
        )
    )


def _stub_fetch_records_calls(
    src: dict,
    out_root: Path,
    *,
    no_cffi: bool = False,
    pace_fn: object = None,
    nb_id: str | None = None,
) -> tuple[dict, list[Path]]:
    """Fetch stub that appends src['id'] to a marker file and writes basic outputs."""
    out_root.parent.parent.mkdir(parents=True, exist_ok=True)
    marker = out_root.parent.parent / "_fetch_calls.txt"
    with marker.open("a") as f:
        f.write(f"{src['id']}\n")
    return _stub_fetch_one_source(src, out_root, no_cffi=no_cffi, pace_fn=pace_fn, nb_id=nb_id)


class TestFilterFailedOnly:
    """Unit tests for the _filter_failed_only helper."""

    def test_keeps_sources_with_no_meta_json(self, tmp_path: Path) -> None:
        from src.pipeline.queues import FetchItem
        from src.pipeline.runner import _filter_failed_only

        items = [
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 1, "id": "s1", "title": "Paper", "url": "http://a/1"},
                no_cffi=False,
            )
        ]
        failed, evict = _filter_failed_only(items, tmp_path)
        assert failed == items
        assert evict == ["http://a/1"]

    def test_drops_source_with_ok_web(self, tmp_path: Path) -> None:
        from src.pipeline.queues import FetchItem
        from src.pipeline.runner import _filter_failed_only

        d = tmp_path / "alpha" / "paper"
        d.mkdir(parents=True)
        _write_meta_json(d, idx=1, src_id="s1", title="Paper", url="http://a/1", web_ok=True)
        items = [
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 1, "id": "s1", "title": "Paper", "url": "http://a/1"},
                no_cffi=False,
            )
        ]
        failed, evict = _filter_failed_only(items, tmp_path)
        assert failed == []
        assert evict == []

    def test_drops_source_with_ok_pdf(self, tmp_path: Path) -> None:
        from src.pipeline.queues import FetchItem
        from src.pipeline.runner import _filter_failed_only

        d = tmp_path / "alpha" / "paper"
        d.mkdir(parents=True)
        _write_meta_json(d, idx=1, src_id="s1", title="Paper", url="http://a/1", pdf_ok=True)
        items = [
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 1, "id": "s1", "title": "Paper", "url": "http://a/1"},
                no_cffi=False,
            )
        ]
        failed, evict = _filter_failed_only(items, tmp_path)
        assert failed == []

    def test_keeps_source_with_failed_web_and_no_pdf(self, tmp_path: Path) -> None:
        from src.pipeline.queues import FetchItem
        from src.pipeline.runner import _filter_failed_only

        d = tmp_path / "alpha" / "paper"
        d.mkdir(parents=True)
        _write_meta_json(d, idx=1, src_id="s1", title="Paper", url="http://a/1", web_ok=False)
        items = [
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 1, "id": "s1", "title": "Paper", "url": "http://a/1"},
                no_cffi=False,
            )
        ]
        failed, evict = _filter_failed_only(items, tmp_path)
        assert failed == items
        assert evict == ["http://a/1"]

    def test_mixed_batch_partitions_correctly(self, tmp_path: Path) -> None:
        from src.pipeline.queues import FetchItem
        from src.pipeline.runner import _filter_failed_only

        # A: ok, B: failed, C: no meta.json. Three different titles to avoid
        # slug collision in the new on-disk layout (one slug = one dir).
        titles = ["paper_a", "paper_b", "paper_c"]
        for idx, sid, web_ok in [(1, "sA", True), (2, "sB", False)]:
            d = tmp_path / "alpha" / titles[idx - 1]
            d.mkdir(parents=True)
            _write_meta_json(
                d, idx=idx, src_id=sid, title=titles[idx - 1], url=f"http://x/{idx}", web_ok=web_ok
            )
        items = [
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 1, "id": "sA", "title": "paper_a", "url": "http://x/1"},
                no_cffi=False,
            ),
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 2, "id": "sB", "title": "paper_b", "url": "http://x/2"},
                no_cffi=False,
            ),
            FetchItem(
                nb_tag="alpha",
                nb_id="u1",
                src={"index": 3, "id": "sC", "title": "paper_c", "url": "http://x/3"},
                no_cffi=False,
            ),
        ]
        failed, evict = _filter_failed_only(items, tmp_path)
        assert {item.src["id"] for item in failed} == {"sB", "sC"}
        assert set(evict) == {"http://x/2", "http://x/3"}


class TestEvictUrlCacheEntries:
    """Unit tests for evict_url_cache_entries."""

    def test_returns_zero_when_cache_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from src.fetch.util import evict_url_cache_entries

        assert evict_url_cache_entries(["http://a/1"]) == 0

    def test_evicts_only_listed_urls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        from src.fetch.util import (
            URL_CACHE_PATH,
            _save_url_cache,
            evict_url_cache_entries,
        )

        _save_url_cache(
            {
                "http://a/1": {"web": "/path/a"},
                "http://b/2": {"pdf": "/path/b"},
                "http://c/3": {"web": "/path/c"},
            }
        )
        n = evict_url_cache_entries(["http://a/1", "http://c/3"])
        assert n == 2
        cache = json.loads(URL_CACHE_PATH.read_text())
        assert list(cache.keys()) == ["http://b/2"]


class TestDiscoverFreePdf:
    """Free-PDF discovery from HTML — arXiv id context-gating."""

    def test_arxiv_url_context_yields_pdf(self) -> None:
        from src.fetch.util import discover_free_pdf

        html = '<a href="https://arxiv.org/abs/2403.12345">paper</a>'
        assert discover_free_pdf(html, "https://x/") == "https://arxiv.org/pdf/2403.12345"

    def test_arxiv_label_context_yields_pdf(self) -> None:
        from src.fetch.util import discover_free_pdf

        assert discover_free_pdf("see arXiv:2403.12345v2", "https://x/") == (
            "https://arxiv.org/pdf/2403.12345"
        )

    def test_6digit_id_discovered(self) -> None:
        """A 6-digit suffix in arXiv context resolves (old {4,5} missed it)."""
        from src.fetch.util import discover_free_pdf

        assert discover_free_pdf("arxiv.org/abs/2403.123456", "https://x/") == (
            "https://arxiv.org/pdf/2403.123456"
        )

    def test_bare_token_does_not_trigger_download(self) -> None:
        """An incidental NNNN.NNNNN in body text must NOT be treated as an arXiv id."""
        from src.fetch.util import discover_free_pdf

        html = "<p>Revenue was 2403.12345 million in Q1.</p>"
        assert discover_free_pdf(html, "https://x/") is None

    def test_falls_back_to_pdf_link(self) -> None:
        from src.fetch.util import discover_free_pdf

        html = '<p>2403.12345</p><a href="/files/report.pdf">dl</a>'
        assert discover_free_pdf(html, "https://host.com/page") == (
            "https://host.com/files/report.pdf"
        )


@pytest.mark.skip(reason="See TestRunnerDrainsNotebookFixture: legacy stage assertions.")
class TestRetryFailedFlag:
    """Integration tests for run_pipeline(retry_failed=True)."""

    def test_filters_to_failed_sources_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three sources: A ok, B failed, C never fetched. Only B and C reach fetch."""
        monkeypatch.chdir(tmp_path)
        from src.pipeline.runner import run_pipeline

        # Pre-populate sources/alpha with meta.json for A and B
        sources_dir = tmp_path / "sources" / "alpha"
        for idx, sid, web_ok in [(1, "sA", True), (2, "sB", False)]:
            d = sources_dir / f"{idx:03d}__{_slug_stub(f'Paper {sid}')}"
            d.mkdir(parents=True)
            _write_meta_json(
                d, idx=idx, src_id=sid, title=f"Paper {sid}", url=f"http://x/{idx}", web_ok=web_ok
            )

        notebooks = _make_notebooks(
            [
                (
                    "u1",
                    "alpha",
                    "trading",
                    [
                        _make_source(1, "sA", "Paper sA", "T0_arxiv"),
                        _make_source(2, "sB", "Paper sB", "T0_arxiv"),
                        _make_source(3, "sC", "Paper sC", "T0_arxiv"),
                    ],
                ),
            ]
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            rc = run_pipeline(
                notebooks,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                retry_failed=True,
                _fetch_fn=_stub_fetch_records_calls,
                _parse_fn=_stub_parse_pipeline_only,
            )

        assert rc == 0
        marker = tmp_path / "_fetch_calls.txt"
        called_ids = set(marker.read_text().split()) if marker.exists() else set()
        assert called_ids == {"sB", "sC"}, f"expected sB and sC to be re-fetched, got {called_ids}"

    def test_evicts_cache_for_retried_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache entry for the failed source should be removed before fetch."""
        monkeypatch.chdir(tmp_path)
        from src.fetch.util import URL_CACHE_PATH, _save_url_cache
        from src.pipeline.runner import run_pipeline

        # Source B has failed meta.json + a stale cache entry. URL must match
        # what _make_source generates for index=1: http://example.com/1.
        retry_url = "http://example.com/1"
        sources_dir = tmp_path / "sources" / "alpha"
        d = sources_dir / "001__paper_sb"
        d.mkdir(parents=True)
        _write_meta_json(d, idx=1, src_id="sB", title="Paper sB", url=retry_url, web_ok=False)
        _save_url_cache(
            {
                retry_url: {"web": "/stale/path"},
                "http://other/9": {"web": "/keep"},
            }
        )

        notebooks = _make_notebooks(
            [
                (
                    "u1",
                    "alpha",
                    "trading",
                    [
                        _make_source(1, "sB", "Paper sB", "T0_arxiv"),
                    ],
                ),
            ]
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            run_pipeline(
                notebooks,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                retry_failed=True,
                _fetch_fn=_stub_fetch_one_source,
                _parse_fn=_stub_parse_pipeline_only,
            )

        cache = json.loads(URL_CACHE_PATH.read_text())
        # The stale cache entry for the retried URL should be gone
        assert retry_url not in cache
        # Unrelated entries preserved
        assert "http://other/9" in cache

    def test_empty_after_filter_exits_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All sources successful → nothing to retry → fetch worker not invoked."""
        monkeypatch.chdir(tmp_path)
        from src.pipeline.runner import run_pipeline

        sources_dir = tmp_path / "sources" / "alpha"
        d = sources_dir / "001__paper_sa"
        d.mkdir(parents=True)
        _write_meta_json(d, idx=1, src_id="sA", title="Paper sA", url="http://x/1", web_ok=True)

        notebooks = _make_notebooks(
            [
                (
                    "u1",
                    "alpha",
                    "trading",
                    [
                        _make_source(1, "sA", "Paper sA", "T0_arxiv"),
                    ],
                ),
            ]
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            rc = run_pipeline(
                notebooks,
                stages=["fetch", "parse_pipeline", "stage"],
                root=tmp_path,
                parallel=1,
                retry_failed=True,
                _fetch_fn=_stub_fetch_records_calls,
                _parse_fn=_stub_parse_pipeline_only,
            )

        assert rc == 0
        marker = tmp_path / "_fetch_calls.txt"
        assert not marker.exists() or marker.read_text() == "", (
            "fetch should not have been called when nothing to retry"
        )
