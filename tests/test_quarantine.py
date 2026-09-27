"""Acceptance tests for gh #17 — crasher detection + quarantine.

Two-phase contract:

* The supervisor records ``last_completed_pdf`` and ``pending_pdf_at_death``
  for each daemon SIGABRT it observes. After two correlated deaths on the
  same suspect PDF, the suspect is quarantined: its source dir is moved
  under ``sources/_quarantine/<collection>/<slug>/`` with a sidecar JSON
  describing the death. Subsequent ragctl runs skip quarantined sources.

* ``bin/ragctl quarantine list|show|unquarantine`` provides operator
  visibility and manual recovery.

Fallback parser chain (parse_vlm → pdfplumber → skip) is deferred to a
follow-up PR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.quarantine import (
    QuarantineSidecar,
    is_quarantined,
    list_quarantined,
    quarantine_source,
    unquarantine_source,
)

# ---------------------------------------------------------------------------
# 1. quarantine_source moves the dir and writes a sidecar
# ---------------------------------------------------------------------------


class TestQuarantineSource:
    """The quarantine action moves ``sources/<coll>/<slug>/`` to
    ``sources/_quarantine/<coll>/<slug>/`` and drops a forensics sidecar
    so the operator can review the suspect later."""

    def test_moves_dir_under_quarantine_root(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        src_dir = sources_root / "trading" / "bad-paper"
        src_dir.mkdir(parents=True)
        (src_dir / "source.pdf").write_bytes(b"%PDF-1.4 fake")
        (src_dir / "meta.json").write_text(json.dumps({"url": "http://x"}))

        sidecar = QuarantineSidecar(
            reason="mineru_pipeline_glibc_corruption",
            deaths_observed=2,
            predecessor_pdf="prev-paper",
            daemon_log_refs=["pipeline_daemon_x.log"],
            url="http://x",
        )
        new_path = quarantine_source(
            sources_root=sources_root,
            collection="trading",
            slug="bad-paper",
            sidecar=sidecar,
        )

        # Source dir is gone from its old home.
        assert not src_dir.exists()
        # New dir lives under _quarantine.
        assert new_path == sources_root / "_quarantine" / "trading" / "bad-paper"
        assert new_path.is_dir()
        assert (new_path / "source.pdf").exists()
        # Sidecar written.
        sc = json.loads((new_path / "_quarantine.json").read_text())
        assert sc["reason"] == "mineru_pipeline_glibc_corruption"
        assert sc["deaths_observed"] == 2
        assert sc["predecessor_pdf"] == "prev-paper"
        assert sc["daemon_log_refs"] == ["pipeline_daemon_x.log"]
        assert sc["url"] == "http://x"
        assert "quarantined_at" in sc  # ISO-8601 timestamp

    def test_idempotent_when_already_quarantined(self, tmp_path: Path) -> None:
        """Re-quarantining an already-quarantined slug must not crash —
        the sidecar updates to reflect the latest death count."""
        sources_root = tmp_path / "sources"
        src_dir = sources_root / "trading" / "bad-paper"
        src_dir.mkdir(parents=True)
        (src_dir / "source.pdf").write_bytes(b"%PDF-1.4 fake")

        sidecar1 = QuarantineSidecar(
            reason="x",
            deaths_observed=2,
            predecessor_pdf="a",
            daemon_log_refs=[],
            url="",
        )
        quarantine_source(
            sources_root=sources_root, collection="trading", slug="bad-paper", sidecar=sidecar1
        )

        # Same slug quarantined again — must succeed without error.
        sidecar2 = QuarantineSidecar(
            reason="x",
            deaths_observed=4,
            predecessor_pdf="b",
            daemon_log_refs=[],
            url="",
        )
        # Recreate src dir to simulate a re-fetch that gets quarantined again.
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "source.pdf").write_bytes(b"%PDF-1.4 fake-2")
        new_path = quarantine_source(
            sources_root=sources_root,
            collection="trading",
            slug="bad-paper",
            sidecar=sidecar2,
        )
        sc = json.loads((new_path / "_quarantine.json").read_text())
        assert sc["deaths_observed"] == 4  # updated from latest call


# ---------------------------------------------------------------------------
# 2. is_quarantined / list_quarantined
# ---------------------------------------------------------------------------


class TestQuarantineLookup:
    def test_is_quarantined_true_after_quarantine(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        src_dir = sources_root / "trading" / "bad-paper"
        src_dir.mkdir(parents=True)
        sidecar = QuarantineSidecar(
            reason="x",
            deaths_observed=2,
            predecessor_pdf="a",
            daemon_log_refs=[],
            url="",
        )
        quarantine_source(
            sources_root=sources_root, collection="trading", slug="bad-paper", sidecar=sidecar
        )

        assert is_quarantined(sources_root, "trading", "bad-paper") is True
        assert is_quarantined(sources_root, "trading", "ok-paper") is False

    def test_list_quarantined_returns_metadata(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        for slug in ["bad-1", "bad-2"]:
            d = sources_root / "trading" / slug
            d.mkdir(parents=True)
            quarantine_source(
                sources_root=sources_root,
                collection="trading",
                slug=slug,
                sidecar=QuarantineSidecar(
                    reason="x",
                    deaths_observed=2,
                    predecessor_pdf="p",
                    daemon_log_refs=[],
                    url="",
                ),
            )

        rows = list_quarantined(sources_root)
        slugs = sorted(r["slug"] for r in rows)
        assert slugs == ["bad-1", "bad-2"]
        assert all(r["collection"] == "trading" for r in rows)
        assert all("quarantined_at" in r for r in rows)


# ---------------------------------------------------------------------------
# 3. unquarantine_source moves dir back
# ---------------------------------------------------------------------------


class TestUnquarantine:
    def test_moves_back_to_active_path(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        src_dir = sources_root / "trading" / "bad-paper"
        src_dir.mkdir(parents=True)
        (src_dir / "source.pdf").write_bytes(b"%PDF-1.4")
        sidecar = QuarantineSidecar(
            reason="x",
            deaths_observed=2,
            predecessor_pdf="a",
            daemon_log_refs=[],
            url="",
        )
        quarantine_source(
            sources_root=sources_root, collection="trading", slug="bad-paper", sidecar=sidecar
        )

        active_path = unquarantine_source(
            sources_root=sources_root,
            collection="trading",
            slug="bad-paper",
        )
        assert active_path == sources_root / "trading" / "bad-paper"
        assert active_path.is_dir()
        assert (active_path / "source.pdf").exists()
        # Sidecar is removed when the source returns to active.
        assert not (active_path / "_quarantine.json").exists()
        # Quarantine entry is gone.
        assert is_quarantined(sources_root, "trading", "bad-paper") is False

    def test_unquarantine_missing_slug_raises(self, tmp_path: Path) -> None:
        sources_root = tmp_path / "sources"
        with pytest.raises(FileNotFoundError):
            unquarantine_source(
                sources_root=sources_root,
                collection="trading",
                slug="never-existed",
            )


# ---------------------------------------------------------------------------
# 4. Quarantined sources are skipped by the pipeline orchestrator
# ---------------------------------------------------------------------------


class TestSourceResolutionSkipsQuarantine:
    """The existing source-walker filters dirs starting with ``_``. A
    quarantined slug lives under ``sources/_quarantine/`` — the leading
    underscore at the top-level directory name is what guarantees it
    never enters the iteration. This test is a regression guard: if
    someone refactors the walker, quarantined sources must stay invisible."""

    def test_select_doc_dirs_skips_underscore_top_dir(self, tmp_path: Path) -> None:
        from src.pipeline.orchestrate import select_doc_dirs

        sources_root = tmp_path / "sources"
        # Active source.
        (sources_root / "trading" / "good-paper").mkdir(parents=True)
        # Quarantined — should NOT appear in select_doc_dirs("trading").
        # Note: select_doc_dirs walks sources_root/<collection>/*/ — so
        # _quarantine/trading/<slug>/ is naturally outside the scope of
        # select_doc_dirs("trading"). What matters is that dirs whose
        # name starts with "_" are filtered, so future refactors that
        # change the layout don't accidentally pick up quarantined data.
        (sources_root / "trading" / "_should_skip").mkdir(parents=True)

        rows = select_doc_dirs("trading", sources_root)
        slugs = [Path(r["doc_dir"]).name for r in rows]
        assert slugs == ["good-paper"]

    def test_quarantine_root_is_underscore_prefixed(self, tmp_path: Path) -> None:
        """The quarantine root must start with underscore so it sorts
        and filters identically to existing skip rules."""
        sources_root = tmp_path / "sources"
        d = sources_root / "trading" / "bad-paper"
        d.mkdir(parents=True)
        sidecar = QuarantineSidecar(
            reason="x",
            deaths_observed=2,
            predecessor_pdf="a",
            daemon_log_refs=[],
            url="",
        )
        new_path = quarantine_source(
            sources_root=sources_root,
            collection="trading",
            slug="bad-paper",
            sidecar=sidecar,
        )
        # Top component below sources_root must start with `_`.
        rel = new_path.relative_to(sources_root)
        assert rel.parts[0].startswith("_"), (
            f"quarantine root must be underscore-prefixed for skip-filter "
            f"compatibility; got {rel.parts[0]!r}"
        )


# ---------------------------------------------------------------------------
# 5. CrasherTracker — supervisor-side suspect correlation
# ---------------------------------------------------------------------------


class TestCrasherTracker:
    """Pairs the daemon's last-completed POST with the next-in-flight POST
    around each SIGABRT to identify deterministic crashers. Only after a
    second correlated death (same predecessor → same pending) is the
    pending PDF tagged as a confirmed crasher."""

    def test_first_death_tags_attempted_not_confirmed(self) -> None:
        from src.pipeline.crasher_tracker import CrasherTracker

        t = CrasherTracker()
        outcome = t.record_death(
            last_completed_pdf="ok-paper",
            pending_pdf="suspect",
            log_ref="d1.log",
        )
        assert outcome.confirmed is False
        assert outcome.deaths_observed == 1
        assert t.attempts_for("suspect") == 1

    def test_second_correlated_death_confirms(self) -> None:
        from src.pipeline.crasher_tracker import CrasherTracker

        t = CrasherTracker()
        t.record_death(
            last_completed_pdf="ok-paper",
            pending_pdf="suspect",
            log_ref="d1.log",
        )
        outcome = t.record_death(
            last_completed_pdf="ok-paper",
            pending_pdf="suspect",
            log_ref="d2.log",
        )
        assert outcome.confirmed is True
        assert outcome.deaths_observed == 2
        assert outcome.suspect == "suspect"
        # Second outcome carries both log refs for forensics sidecar.
        assert "d1.log" in outcome.daemon_log_refs
        assert "d2.log" in outcome.daemon_log_refs

    def test_uncorrelated_second_death_does_not_confirm(self) -> None:
        """If the second SIGABRT is on a different pending PDF, neither
        is confirmed yet — could be transient, not deterministic."""
        from src.pipeline.crasher_tracker import CrasherTracker

        t = CrasherTracker()
        out1 = t.record_death(
            last_completed_pdf="a",
            pending_pdf="suspect-1",
            log_ref="d1.log",
        )
        out2 = t.record_death(
            last_completed_pdf="b",
            pending_pdf="suspect-2",
            log_ref="d2.log",
        )
        assert out1.confirmed is False
        assert out2.confirmed is False

    def test_unknown_pending_is_safe(self) -> None:
        """If the supervisor can't determine pending_pdf at SIGABRT time
        (race window, log line not flushed), the tracker must accept
        ``None`` and never confirm — quarantine action is skipped."""
        from src.pipeline.crasher_tracker import CrasherTracker

        t = CrasherTracker()
        out = t.record_death(
            last_completed_pdf="ok",
            pending_pdf=None,
            log_ref="d.log",
        )
        assert out.confirmed is False
        assert out.suspect is None


# ---------------------------------------------------------------------------
# 6. Supervisor wires CrasherTracker through to last_crasher_outcome
# ---------------------------------------------------------------------------


class TestSupervisorCrasherIntegration:
    """When constructed with a CrasherTracker, the supervisor records the
    outcome of every death on ``self.last_crasher_outcome`` so the
    coordinator (which has access to the sources_root + slug → dir
    mapping) can decide to quarantine without the supervisor needing
    to know about filesystem paths."""

    def test_supervisor_records_crasher_outcome_on_death(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon
        from src.pipeline.crasher_tracker import CrasherTracker
        from src.pipeline.daemon_supervisor import DaemonSupervisor

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(
            mineru_daemon,
            "current_daemon_log_path",
            lambda: Path("/tmp/d.log"),
        )
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        tracker = CrasherTracker()
        sup = DaemonSupervisor(crasher_tracker=tracker)

        # Coordinator-side hooks: a parse_start was just observed, no
        # parse_ok yet → that PDF is the pending suspect.
        sup.note_parse_start("suspect-paper")

        sup.tick(force=True)

        assert sup.last_crasher_outcome is not None
        assert sup.last_crasher_outcome.suspect == "suspect-paper"
        # First death — not yet confirmed.
        assert sup.last_crasher_outcome.confirmed is False

    def test_second_death_on_same_pending_confirms(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon
        from src.pipeline.crasher_tracker import CrasherTracker
        from src.pipeline.daemon_supervisor import DaemonSupervisor

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(
            mineru_daemon,
            "current_daemon_log_path",
            lambda: Path("/tmp/d.log"),
        )
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor(
            crasher_tracker=CrasherTracker(),
            circuit_deaths=10,
        )
        sup.note_parse_start("suspect-paper")
        sup.tick(force=True)
        # Second death on the same suspect (not yet observed parse_ok or
        # parse_fail for it, so it remains in pending).
        sup.tick(force=True)

        assert sup.last_crasher_outcome is not None
        assert sup.last_crasher_outcome.confirmed is True
        assert sup.last_crasher_outcome.suspect == "suspect-paper"
        assert sup.last_crasher_outcome.deaths_observed == 2

    def test_pending_pdf_pick_is_deterministic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multi-worker reality: up to N parse workers may have PDFs in
        flight when SIGABRT hits. The supervisor must pick the lowest-
        sorted suspect deterministically (not ``next(iter(set))``),
        otherwise set hash ordering — which varies with set composition
        and PYTHONHASHSEED — can rotate the suspect between ticks and
        confirm the wrong PDF as the crasher.
        """
        from src.pdf_parsers import mineru_daemon
        from src.pipeline.crasher_tracker import CrasherTracker
        from src.pipeline.daemon_supervisor import DaemonSupervisor

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(
            mineru_daemon,
            "current_daemon_log_path",
            lambda: Path("/tmp/d.log"),
        )
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor(
            crasher_tracker=CrasherTracker(),
            circuit_deaths=10,
        )
        # Insert in arbitrary order — the contract is that sup picks
        # the alphabetically first slug regardless of insertion order.
        for slug in ["paper-zebra", "paper-alpha", "paper-mango"]:
            sup.note_parse_start(slug)

        sup.tick(force=True)
        out = sup.last_crasher_outcome
        assert out is not None
        # Lowest-sorted slug — deterministic, reproducible across runs
        # and across set hash-seed permutations.
        assert out.suspect == "paper-alpha", (
            f"expected lexicographically-lowest suspect, got {out.suspect!r}"
        )

    def test_no_tracker_means_no_crasher_outcome(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backwards compat — supervisor without a tracker is a no-op
        on the crasher path."""
        from src.pdf_parsers import mineru_daemon
        from src.pipeline.daemon_supervisor import DaemonSupervisor

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor()  # no tracker
        sup.tick(force=True)
        assert sup.last_crasher_outcome is None


# ---------------------------------------------------------------------------
# 7. Coordinator → quarantine action on confirmed crasher
# ---------------------------------------------------------------------------


class TestCoordinatorQuarantinesConfirmed:
    """When the supervisor returns a tick whose ``last_crasher_outcome``
    is confirmed, the coordinator must invoke ``quarantine_source`` for
    the suspect's source dir."""

    def test_quarantine_called_on_confirmed_outcome(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pipeline import runner
        from src.pipeline.crasher_tracker import CrasherOutcome

        sources_root = tmp_path / "sources"
        # Pre-create the suspect's source dir so quarantine has something
        # to move.
        (sources_root / "trading" / "bad-paper").mkdir(parents=True)
        (sources_root / "trading" / "bad-paper" / "source.pdf").write_bytes(b"%PDF")

        # Drive runner._maybe_quarantine_confirmed_crasher directly: it's
        # the seam the coordinator uses after each death-tick.
        outcome = CrasherOutcome(
            confirmed=True,
            suspect="bad-paper",
            deaths_observed=2,
            daemon_log_refs=["d1.log", "d2.log"],
        )
        runner._maybe_quarantine_confirmed_crasher(  # type: ignore[attr-defined]
            outcome=outcome,
            collection="trading",
            sources_root=sources_root,
            last_completed_pdf="ok-paper",
        )

        # Source dir moved under _quarantine.
        assert not (sources_root / "trading" / "bad-paper").exists()
        qdir = sources_root / "_quarantine" / "trading" / "bad-paper"
        assert qdir.is_dir()
        sc = json.loads((qdir / "_quarantine.json").read_text())
        assert sc["reason"] == "mineru_pipeline_glibc_corruption"
        assert sc["deaths_observed"] == 2
        assert sc["daemon_log_refs"] == ["d1.log", "d2.log"]
        assert sc["predecessor_pdf"] == "ok-paper"

    def test_unconfirmed_outcome_is_noop(
        self,
        tmp_path: Path,
    ) -> None:
        from src.pipeline import runner
        from src.pipeline.crasher_tracker import CrasherOutcome

        sources_root = tmp_path / "sources"
        (sources_root / "trading" / "maybe-bad").mkdir(parents=True)

        outcome = CrasherOutcome(
            confirmed=False,
            suspect="maybe-bad",
            deaths_observed=1,
            daemon_log_refs=["d1.log"],
        )
        runner._maybe_quarantine_confirmed_crasher(  # type: ignore[attr-defined]
            outcome=outcome,
            collection="trading",
            sources_root=sources_root,
            last_completed_pdf=None,
        )

        # No quarantine yet — first death is "attempted", not "confirmed".
        assert (sources_root / "trading" / "maybe-bad").exists()
        assert not (sources_root / "_quarantine").exists()
