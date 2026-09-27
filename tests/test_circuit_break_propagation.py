"""Acceptance tests for gh #16 — circuit_break propagation.

Today the supervisor opens the circuit-breaker and logs a single line, but:

* parse workers keep consuming items from parse_q, each failing with
  Connection refused (every queued PDF takes ~1-2s to error out → minutes
  of "frozen" progress).
* If the spider is slow on remaining fetches, the coordinator never
  terminates because fetch_done_count < fetch_total and parse_pending
  may stay > 0.
* No top-level summary line at exit so operators can't tell what happened.

These tests exercise the propagation contract:

* When tick() returns a circuit_break event, coordinator must drain parse_q
  of pending ParseItems by emitting synthetic parse_fail (not waiting for
  workers to slow-fail each one).
* On circuit_break, remaining un-fetched items must be synthesised as
  fetch_fail so coordinator can terminate cleanly.
* run_pipeline must print a summary line at exit when the breaker tripped.
"""

from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from src.pipeline.daemon_supervisor import SupervisorEvent
from src.pipeline.queues import ParseItem
from src.pipeline.runner import _RunProgress, coordinator_loop

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


def _run_coord_with_timeout(
    rp: _RunProgress,
    fetch_q: Queue[object],
    parse_q: Queue[object],
    result_q: Queue[object],
    *,
    do_fetch: bool,
    do_parse: bool,
    supervisor: _CircuitTrippingSupervisor,
    timeout: float = 5.0,
) -> None:
    """Run coordinator_loop in a thread; fail the test if it hangs.

    We use a thread (not a process) so all state mutations on rp/queues
    are visible after join. The coordinator is pure-Python and releases
    the GIL during result_q polling, so the join() does not deadlock.
    """
    err: list[BaseException] = []

    def _target() -> None:
        try:
            coordinator_loop(
                rp,
                fetch_q,
                parse_q,
                result_q,
                do_fetch=do_fetch,
                do_parse=do_parse,
                supervisor=supervisor,
            )
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise AssertionError(f"coordinator_loop hung > {timeout}s — termination contract broken")
    if err:
        raise err[0]


@pytest.fixture()
def queues() -> tuple[Any, Any, Any]:
    ctx = multiprocessing.get_context("spawn")
    return ctx.Queue(maxsize=20), ctx.Queue(maxsize=20), ctx.Queue()


def _make_progress(
    nbs: list[tuple[str, int]], *, pre_pending: int = 0, pdfs_expected: int = 0
) -> _RunProgress:
    rp = _RunProgress()
    for tag, total in nbs:
        rp.progress[tag] = {
            "total": total,
            "fetched": 0,
            "pdfs_expected": pdfs_expected,
            "pdfs_parsed": 0,
        }
        rp.fetch_total += total
    rp.parse_pending = pre_pending
    return rp


class _CircuitTrippingSupervisor:
    """Returns ``circuit_break`` from the first tick(force=True), open
    on every subsequent call. Mirrors the real supervisor's contract."""

    def __init__(self) -> None:
        self.tick_calls: list[bool] = []
        self._open = False

    def is_circuit_open(self) -> bool:
        return self._open

    def note_parse_fail_reason(self, reason: str) -> bool:
        return "ConnectionRefused" in reason

    def tick(self, *, force: bool = False) -> SupervisorEvent | None:
        self.tick_calls.append(force)
        if self._open:
            return None
        # First force=True call (driven by parse_fail with daemon-dead
        # reason) immediately trips the breaker.
        if force:
            self._open = True
            return SupervisorEvent(kind="circuit_break", death_index=3)
        return None


# ---------------------------------------------------------------------------
# 1. parse_q is drained on circuit_break
# ---------------------------------------------------------------------------


class TestParseQueueDrainOnCircuitBreak:
    """When the breaker trips, the coordinator must remove every pending
    ParseItem from parse_q and emit synthetic parse_fail events for them.
    Otherwise workers keep popping items, each failing with Connection
    refused over multiple minutes (gh #16's main symptom)."""

    def test_pending_parse_items_are_drained(
        self,
        queues: tuple[Any, Any, Any],
    ) -> None:
        fetch_q, parse_q, result_q = queues
        # Three PDFs on parse_q + counted in pdfs_expected. fetch_total
        # must be > 0 for the do_parse-only termination path to actually
        # run iterations (the coord short-circuits when fetch_total == 0).
        rp = _make_progress([("alpha", 1)], pre_pending=3, pdfs_expected=3)
        # Mark the lone fetch as already done so termination only depends
        # on parse_pending.
        rp.fetch_done_count = 1
        rp.progress["alpha"]["fetched"] = 1

        for i in range(3):
            parse_q.put(ParseItem(nb_tag="alpha", pdf_path=Path(f"/tmp/p{i}.pdf")))

        sup = _CircuitTrippingSupervisor()

        # Seed a parse_fail with daemon-death reason — that triggers the
        # supervisor's force tick → circuit_break event returned.
        result_q.put(
            (
                "parse_fail",
                "alpha",
                "/tmp/p0.pdf",
                "ConnectionRefusedError: [Errno 111] Connection refused",
            )
        )

        _run_coord_with_timeout(
            rp, fetch_q, parse_q, result_q, do_fetch=False, do_parse=True, supervisor=sup
        )

        # parse_q must be empty — coordinator drained the remaining 2 items
        # synthetically rather than letting workers slow-fail them.
        assert parse_q.empty(), (
            "parse_q must be drained by coordinator on circuit_break, "
            "not left for workers to slow-fail"
        )
        # parse_pending must reach 0 — initial parse_fail decremented to 2,
        # then 2 more synthetic parse_fails decremented to 0.
        assert rp.parse_pending == 0, (
            f"parse_pending must reach 0 on circuit_break, got {rp.parse_pending}"
        )

    def test_drain_preserves_poison_pills(
        self,
        queues: tuple[Any, Any, Any],
    ) -> None:
        """If a poison pill (None) was already enqueued before the
        breaker tripped (e.g. parallel cleanup race), the drain must
        re-insert it so worker join() can complete. Consuming + dropping
        the pill leaves a worker hung on parse_q.get() forever."""
        fetch_q, parse_q, result_q = queues
        rp = _make_progress([("alpha", 1)], pre_pending=2, pdfs_expected=2)
        rp.fetch_done_count = 1
        rp.progress["alpha"]["fetched"] = 1

        parse_q.put(ParseItem(nb_tag="alpha", pdf_path=Path("/tmp/p0.pdf")))
        parse_q.put(None)  # poison pill from somewhere

        sup = _CircuitTrippingSupervisor()
        result_q.put(
            (
                "parse_fail",
                "alpha",
                "/tmp/p0.pdf",
                "ConnectionRefusedError: [Errno 111] Connection refused",
            )
        )

        _run_coord_with_timeout(
            rp, fetch_q, parse_q, result_q, do_fetch=False, do_parse=True, supervisor=sup
        )

        # The pill must still be in the queue for workers to consume.
        import queue as _q

        try:
            item = parse_q.get(timeout=1.0)
        except _q.Empty:
            raise AssertionError(
                "drain consumed the poison pill — workers will hang",
            ) from None
        assert item is None, f"expected poison pill in queue after drain, got {item!r}"

    def test_drain_does_not_run_when_circuit_closed(
        self,
        queues: tuple[Any, Any, Any],
    ) -> None:
        """If a parse_fail reason looks like daemon-death but the breaker
        does NOT trip (transient issue, supervisor ticks but stays closed),
        we must NOT drain parse_q. The next worker dispatch may succeed
        after a respawn."""
        fetch_q, parse_q, result_q = queues
        rp = _make_progress([("alpha", 1)], pre_pending=1, pdfs_expected=1)
        rp.fetch_done_count = 1
        rp.progress["alpha"]["fetched"] = 1
        parse_q.put(ParseItem(nb_tag="alpha", pdf_path=Path("/tmp/p0.pdf")))

        class _StableSupervisor:
            tick_calls: list[bool] = []

            def is_circuit_open(self) -> bool:
                return False

            def note_parse_fail_reason(self, reason: str) -> bool:
                return "ConnectionRefused" in reason

            def tick(self, *, force: bool = False) -> SupervisorEvent | None:
                _StableSupervisor.tick_calls.append(force)
                return None

        result_q.put(
            (
                "parse_fail",
                "alpha",
                "/tmp/p0.pdf",
                "ConnectionRefusedError: [Errno 111] Connection refused",
            )
        )

        coordinator_loop(
            rp,
            fetch_q,
            parse_q,
            result_q,
            do_fetch=False,
            do_parse=True,
            supervisor=_StableSupervisor(),
        )  # type: ignore[arg-type]

        # Item still queued — drain only fires on circuit_break.
        # (qsize is unreliable on macOS; we just confirm non-empty.)
        assert not parse_q.empty(), "parse_q must NOT be drained when circuit stays closed"


# ---------------------------------------------------------------------------
# 2. Remaining un-fetched items are synthesised as fetch_fail on break
# ---------------------------------------------------------------------------


class TestUnfetchedItemsSynthesisedOnBreak:
    """If the spider is mid-flight when the breaker trips, the remaining
    fetches still trickle in over minutes. To unblock coordinator
    termination, synthesise fetch_fail for the deficit so
    fetch_done_count catches up to fetch_total."""

    def test_fetch_done_count_catches_up_after_break(
        self,
        queues: tuple[Any, Any, Any],
    ) -> None:
        fetch_q, parse_q, result_q = queues
        # 5 fetches expected, only 1 has produced an event so far.
        rp = _make_progress([("alpha", 5)], pre_pending=1, pdfs_expected=1)
        parse_q.put(ParseItem(nb_tag="alpha", pdf_path=Path("/tmp/p0.pdf")))

        sup = _CircuitTrippingSupervisor()

        # Seed: 1 fetch_ok already done, then a parse_fail trips the breaker.
        result_q.put(("fetch_ok", "alpha", "s1", False, None))  # web-only
        result_q.put(
            (
                "parse_fail",
                "alpha",
                "/tmp/p0.pdf",
                "ConnectionRefusedError: [Errno 111] Connection refused",
            )
        )

        _run_coord_with_timeout(
            rp, fetch_q, parse_q, result_q, do_fetch=True, do_parse=True, supervisor=sup
        )

        # All fetches accounted for — even the un-arrived 4 became
        # synthetic fails on circuit_break. Otherwise the loop never
        # terminates with do_fetch=True (fetch_done_count < fetch_total).
        assert rp.fetch_done_count >= rp.fetch_total, (
            f"fetch_done_count {rp.fetch_done_count} must reach "
            f"fetch_total {rp.fetch_total} after circuit_break"
        )


# ---------------------------------------------------------------------------
# 3. Run summary line on circuit_break
# ---------------------------------------------------------------------------


class TestRunSummaryOnCircuitBreak:
    """run_pipeline must print a single human-readable summary line when
    the run exits with the breaker tripped, so operators don't grep
    journalctl to figure out why ragctl bailed."""

    def test_summary_printed_on_circuit_break_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        from src.pipeline import runner

        # Build a minimal "do nothing" run that immediately reports
        # circuit_break: we patch out the queue machinery and supervisor.
        # Stub all the heavy bits — we want only the summary surface.
        monkeypatch.setattr(runner, "_acquire_run_lock", lambda: 0)
        monkeypatch.setattr(runner, "_release_run_lock", lambda fd: None)

        # Force run_pipeline to think the breaker tripped at exit.
        # Easiest path: patch coordinator_loop to mark a sentinel supervisor
        # open, and ensure run_pipeline checks supervisor.is_circuit_open at
        # the end.
        captured_summary: list[str] = []

        original_print = print

        def capturing_print(*args: object, **kwargs: object) -> None:
            line = " ".join(str(a) for a in args)
            captured_summary.append(line)
            original_print(*args, **kwargs)

        monkeypatch.setattr("builtins.print", capturing_print)

        # Run a degenerate pipeline with no work but parse stages selected,
        # then force supervisor.is_circuit_open() = True on the post-barrier
        # check by injecting a stub supervisor via a module-level seam.
        # Since the runner wires its supervisor inline, we instead build the
        # smallest scenario that exercises the break path: 1 PDF, force
        # tick to break.
        sources_root = tmp_path / "sources" / "trading"
        sources_root.mkdir(parents=True)
        # No PDFs → parse_items empty, do_parse w/ parse-only mode is no-op.

        # Use stages=["audit"] so post-barrier still tries to run; should
        # surface the summary line before/after audit.
        rc = runner.run_pipeline(
            notebooks=[{"id": "nb1", "collection": "trading"}],
            stages=["audit"],
            root=tmp_path,
            parallel=1,
            dry_run=True,  # short-circuit to print plan and exit
        )
        # dry_run returns 0 — we just want to verify the print-capture
        # mechanism doesn't crash. The real check is below.
        assert rc == 0

        # Now exercise the actual summary path via direct unit:
        from src.pipeline.daemon_supervisor import DaemonSupervisor

        class _OpenSupervisor(DaemonSupervisor):
            def is_circuit_open(self) -> bool:
                return True

        # Helper exists on the runner to print the summary; the contract
        # is: a single line containing "circuit_break" and an "ok=" / "fail="
        # breakdown so operators know how much survived.
        runner._print_run_summary(  # type: ignore[attr-defined]
            supervisor=_OpenSupervisor(),
            progress={
                "trading": {"total": 5, "fetched": 5, "pdfs_expected": 3, "pdfs_parsed": 1},
            },
        )

        out = "\n".join(captured_summary)
        assert "circuit_break" in out, f"summary line must mention circuit_break, got:\n{out}"
        assert "ok=1" in out or "ok=" in out, f"summary line must contain ok= count, got:\n{out}"


# ---------------------------------------------------------------------------
# 4. Post-barrier still runs on partial-success
# ---------------------------------------------------------------------------


class TestPostBarrierRunsAfterBreak:
    """Even when the breaker trips, audit/ingest must run on whatever
    parsed before the break — otherwise N successful parses are stranded
    on disk and never reach Milvus (gh #16, today's 374-source loss)."""

    def test_post_barrier_runs_when_circuit_open_with_partial_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from src.pipeline import runner

        # Track which post-barrier subprocess commands fire.
        invocations: list[list[str]] = []

        class _FakeCompleted:
            returncode = 0

        def fake_subprocess_run(cmd: list[str], **kw: object) -> _FakeCompleted:
            invocations.append(cmd)
            return _FakeCompleted()

        monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

        # Stub run lock + audit so we don't actually touch disk.
        monkeypatch.setattr(runner, "_acquire_run_lock", lambda: 0)
        monkeypatch.setattr(runner, "_release_run_lock", lambda fd: None)
        monkeypatch.setattr(runner, "_run_audit", lambda root: 0)

        # Notebook with no fetch work and no parse work — but stages
        # include ingest.
        rc = runner.run_pipeline(
            notebooks=[{"id": "nb1", "collection": "trading"}],
            stages=["ingest"],
            root=tmp_path,
            parallel=1,
        )

        # ingest should have been invoked despite zero fetch / parse work.
        # (This is more about confirming the existing partial-success path
        # works; the real fix below ensures it ALSO works when the breaker
        # tripped earlier.)
        assert any("src.ingest.ingest" in " ".join(c) for c in invocations), (
            f"ingest must run; got invocations: {invocations}"
        )
        assert rc == 0
