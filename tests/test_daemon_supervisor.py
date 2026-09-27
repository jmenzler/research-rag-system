"""Unit tests for src/pipeline/daemon_supervisor.py.

The supervisor owns the main-process side of MinerU daemon liveness:
detect death, dump forensics, respawn, open circuit breaker after repeated
failures. These tests exercise the policy layer with mocked
``mineru_daemon`` primitives — no real process spawning here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.daemon_supervisor import DaemonSupervisor


@pytest.fixture()
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "mineru-pipeline.json"
    monkeypatch.setenv("RAG_MINERU_STATE", str(p))
    return p


# ---------------------------------------------------------------------------
# is_alive() — mineru_daemon side
# ---------------------------------------------------------------------------


class TestIsAlive:
    """``mineru_daemon.is_alive()`` end-to-end with mocked HTTP."""

    def test_returns_false_when_state_file_missing(self, state_file: Path) -> None:
        from src.pdf_parsers import mineru_daemon

        assert not state_file.exists()
        assert mineru_daemon.is_alive() is False

    def test_returns_false_when_pid_dead(self, state_file: Path) -> None:
        from src.pdf_parsers import mineru_daemon

        state_file.write_text(
            json.dumps(
                {
                    "url": "http://127.0.0.1:5000",
                    "pid": 999_999,
                    "log_path": "/tmp/x.log",
                }
            )
        )
        assert mineru_daemon.is_alive() is False

    def test_returns_false_when_http_probe_fails(
        self,
        state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os

        from src.pdf_parsers import mineru_daemon

        state_file.write_text(
            json.dumps(
                {
                    "url": "http://127.0.0.1:1",  # nothing listening on port 1
                    "pid": os.getpid(),
                    "log_path": "/tmp/x.log",
                }
            )
        )
        # Force the HTTP probe to fail without waiting for OS-level timeout.
        monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **kw: False)
        monkeypatch.setattr(mineru_daemon, "_socket_open", lambda *a, **kw: False)
        assert mineru_daemon.is_alive() is False

    def test_returns_true_when_pid_alive_and_http_probe_passes(
        self,
        state_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import os

        from src.pdf_parsers import mineru_daemon

        state_file.write_text(
            json.dumps(
                {
                    "url": "http://127.0.0.1:5000",
                    "pid": os.getpid(),
                    "log_path": "/tmp/x.log",
                }
            )
        )
        monkeypatch.setattr(mineru_daemon, "_http_probe", lambda *a, **kw: True)
        assert mineru_daemon.is_alive() is True


# ---------------------------------------------------------------------------
# current_daemon_log_path()
# ---------------------------------------------------------------------------


def test_current_daemon_log_path_reads_state_file(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    state_file.write_text(
        json.dumps(
            {
                "url": "http://127.0.0.1:5000",
                "pid": 1234,
                "log_path": "/var/log/mineru.log",
            }
        )
    )
    assert mineru_daemon.current_daemon_log_path() == Path("/var/log/mineru.log")


def test_current_daemon_log_path_returns_none_when_missing(state_file: Path) -> None:
    from src.pdf_parsers import mineru_daemon

    assert mineru_daemon.current_daemon_log_path() is None


def test_current_daemon_log_path_returns_none_when_field_absent(state_file: Path) -> None:
    """Pre-supervisor state files (older daemons) lack log_path — handle gracefully."""
    from src.pdf_parsers import mineru_daemon

    state_file.write_text(json.dumps({"url": "http://127.0.0.1:5000", "pid": 1}))
    assert mineru_daemon.current_daemon_log_path() is None


# ---------------------------------------------------------------------------
# DaemonSupervisor — policy layer
# ---------------------------------------------------------------------------


class TestSupervisorTick:
    """tick() flow: alive→noop, dead→respawn, repeated→circuit_break."""

    def test_alive_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: True)
        sup = DaemonSupervisor()
        assert sup.tick(force=True) is None

    def test_dead_then_alive_emits_respawned_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)
        respawn_calls: list[int] = []

        def fake_start() -> str:
            respawn_calls.append(1)
            return "http://127.0.0.1:5000"

        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", fake_start)

        sup = DaemonSupervisor()
        ev = sup.tick(force=True)

        assert ev is not None
        assert ev.kind == "respawned"
        assert ev.death_index == 1
        assert respawn_calls == [1]

    def test_three_deaths_within_window_open_circuit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor(circuit_deaths=3, circuit_window_s=600.0)
        ev1 = sup.tick(force=True)
        ev2 = sup.tick(force=True)
        ev3 = sup.tick(force=True)

        assert ev1 is not None and ev1.kind == "respawned"
        assert ev2 is not None and ev2.kind == "respawned"
        assert ev3 is not None and ev3.kind == "circuit_break"
        assert sup.is_circuit_open() is True

    def test_circuit_open_skips_further_respawn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once the breaker is open, tick() returns None — no further restarts."""
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)
        respawn_calls: list[int] = []

        def fake_start() -> str:
            respawn_calls.append(1)
            return "url"

        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", fake_start)

        sup = DaemonSupervisor(circuit_deaths=2, circuit_window_s=600.0)
        sup.tick(force=True)  # death #1 → respawned
        sup.tick(force=True)  # death #2 → circuit_break (>= circuit_deaths)
        assert sup.is_circuit_open()
        respawn_count_at_break = len(respawn_calls)

        # Further ticks must be no-ops.
        for _ in range(5):
            assert sup.tick(force=True) is None
        assert len(respawn_calls) == respawn_count_at_break

    def test_sliding_window_evicts_old_deaths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 4th death after the 1st ages out of the window must NOT trip the breaker."""
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        # Fake the monotonic clock so we can age deaths out deterministically.
        clock = [1000.0]

        def fake_monotonic() -> float:
            return clock[0]

        monkeypatch.setattr("src.pipeline.daemon_supervisor.time.monotonic", fake_monotonic)

        sup = DaemonSupervisor(circuit_deaths=3, circuit_window_s=600.0)
        sup.tick(force=True)  # death #1 at t=1000
        clock[0] += 100
        sup.tick(force=True)  # death #2 at t=1100
        clock[0] += 700  # 1st death now outside the 600s window
        ev3 = sup.tick(force=True)  # death #3 at t=1800; only deaths #2+#3 in window

        # Only 2 deaths in the current window → no break.
        assert ev3 is not None and ev3.kind == "respawned"
        assert sup.is_circuit_open() is False

    def test_time_gate_skips_check_when_not_forced(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two ticks in <interval seconds → second one is a no-op."""
        from src.pdf_parsers import mineru_daemon

        is_alive_calls = [0]

        def fake_is_alive() -> bool:
            is_alive_calls[0] += 1
            return True

        monkeypatch.setattr(mineru_daemon, "is_alive", fake_is_alive)

        sup = DaemonSupervisor(check_interval_s=10.0)
        sup.tick()  # first call always probes (last_check_ts == -inf)
        sup.tick()  # within interval → skipped
        assert is_alive_calls[0] == 1


class TestNoteParseFailReason:
    """Reason classifier — only narrowly true for daemon-death markers."""

    @pytest.mark.parametrize(
        "reason",
        [
            "URLError: <urlopen error [Errno 111] Connection refused>",
            "ConnectionRefusedError: [Errno 111] Connection refused",
            "ConnectionResetError: [Errno 104] Connection reset by peer",
            "RemoteDisconnected('Remote end closed connection without response')",
        ],
    )
    def test_recognises_daemon_dead_patterns(self, reason: str) -> None:
        sup = DaemonSupervisor()
        assert sup.note_parse_fail_reason(reason) is True

    @pytest.mark.parametrize(
        "reason",
        [
            "MinerU daemon returned no results for /tmp/x.pdf",
            "MinerU daemon returned empty markdown for /tmp/x.pdf",
            "ValueError: bad pdf",
            "JSONDecodeError: Expecting value",
        ],
    )
    def test_ignores_non_network_failures(self, reason: str) -> None:
        sup = DaemonSupervisor()
        assert sup.note_parse_fail_reason(reason) is False


class TestRespawnFailureCountsAsDeath:
    """If start_pipeline_daemon raises, the supervisor must count that
    as another death and trip the breaker eventually."""

    def test_respawn_failure_records_death_and_can_trip_breaker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: None)

        def boom() -> str:
            raise RuntimeError("port allocation failed")

        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", boom)

        sup = DaemonSupervisor(circuit_deaths=2, circuit_window_s=600.0)
        ev = sup.tick(force=True)
        # Respawn raised → counted as a 2nd death immediately, breaker trips.
        assert ev is not None
        assert ev.kind == "circuit_break"
        assert sup.is_circuit_open()


class TestDumpLogTail:
    """The supervisor prints the dying daemon's tail on detected death."""

    def test_dump_with_real_log_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon

        log_path = tmp_path / "mineru.log"
        log_path.write_text("\n".join(f"line {i}" for i in range(60)))

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(mineru_daemon, "current_daemon_log_path", lambda: log_path)
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor()
        sup.tick(force=True)

        out = capsys.readouterr().out
        assert "daemon_event: died" in out
        assert "line 59" in out, "expected last log line in tail dump"
        # tail -n 50 → first kept line is "line 10", lines 0..9 must NOT appear.
        assert "line 0\n" not in out

    def test_dump_with_missing_log_file_does_not_crash(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.pdf_parsers import mineru_daemon

        monkeypatch.setattr(mineru_daemon, "is_alive", lambda: False)
        monkeypatch.setattr(
            mineru_daemon, "current_daemon_log_path", lambda: Path("/nonexistent/path")
        )
        monkeypatch.setattr(mineru_daemon, "start_pipeline_daemon", lambda: "url")

        sup = DaemonSupervisor()
        # Should not raise even though tail's target doesn't exist.
        ev = sup.tick(force=True)
        assert ev is not None
        out = capsys.readouterr().out
        assert "daemon_event: died" in out
