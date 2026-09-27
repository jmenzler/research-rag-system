"""Unified pipeline observability.

Two-tier model:
1. Per-stage JSONL — one line per item processed, written to
   ``logs/<slug>_<stage>.jsonl``. Schema:
       {ts, slug, stage, item, status, elapsed_s, ...details}
2. Pipeline status JSONL — one line per stage transition, written to
   ``logs/pipeline_status.jsonl`` (single file, all slugs). Schema:
       {ts, slug, stage, action: "start"|"done"|"fail",
        n_items, n_ok, n_fail, elapsed_s}

CLIs in ``ragctl`` consume both files for live tail and post-hoc audit.

All log files live under ``<repo>/logs/``. Lines are JSON, one per line,
written with a flush after each line so live tailers see fresh output.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"

STAGES: tuple[str, ...] = (
    "fetch",
    "parse",
    "stage",
    "audit",
    "manifest",
    "apply",
    "contextualize",
    "ingest",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()


class StageLogger:
    """Writes per-stage JSONL + emits start/done/fail status events.

    Usage::

        with StageLogger("regime_conditional_kelly", "stage") as sl:
            for item in items:
                t0 = time.time()
                try:
                    process(item)
                    sl.item_ok(item.name, elapsed_s=time.time() - t0,
                               chosen="pdf.txt", chars=len(text))
                except Exception as e:
                    sl.item_fail(item.name, elapsed_s=time.time() - t0,
                                 reason=f"{type(e).__name__}: {e}")
    """

    def __init__(self, slug: str, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; allowed: {STAGES}")
        self.slug = slug
        self.stage = stage
        self.t0 = time.time()
        self.n_ok = 0
        self.n_fail = 0
        self.stage_log = LOGS_DIR / f"{slug}_{stage}.jsonl"
        self.status_log = LOGS_DIR / "pipeline_status.jsonl"

    def __enter__(self) -> StageLogger:
        self._status("start", n_items=None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._status("done", n_items=self.n_ok + self.n_fail)
        else:
            # Stage crashed — record as fail, don't suppress the exception.
            _append_jsonl(self.stage_log, {
                "ts": _now_iso(),
                "slug": self.slug,
                "stage": self.stage,
                "item": "<stage-crash>",
                "status": "crash",
                "elapsed_s": round(time.time() - self.t0, 2),
                "reason": f"{exc_type.__name__}: {exc_val}",
            })
            self._status("fail", n_items=self.n_ok + self.n_fail,
                         reason=f"{exc_type.__name__}: {exc_val}")

    def _status(
        self, action: str, n_items: int | None, **extra: Any  # noqa: ANN401
    ) -> None:
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "slug": self.slug,
            "stage": self.stage,
            "action": action,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "elapsed_s": round(time.time() - self.t0, 2),
        }
        if n_items is not None:
            record["n_items"] = n_items
        record.update(extra)
        _append_jsonl(self.status_log, record)

    def item_ok(
        self, item: str, *, elapsed_s: float | None = None, **details: Any  # noqa: ANN401
    ) -> None:
        self.n_ok += 1
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "slug": self.slug,
            "stage": self.stage,
            "item": item,
            "status": "ok",
        }
        if elapsed_s is not None:
            record["elapsed_s"] = round(elapsed_s, 2)
        record.update(details)
        _append_jsonl(self.stage_log, record)

    def item_fail(
        self,
        item: str,
        *,
        elapsed_s: float | None = None,
        reason: str = "",
        **details: Any,  # noqa: ANN401
    ) -> None:
        self.n_fail += 1
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "slug": self.slug,
            "stage": self.stage,
            "item": item,
            "status": "fail",
            "reason": reason,
        }
        if elapsed_s is not None:
            record["elapsed_s"] = round(elapsed_s, 2)
        record.update(details)
        _append_jsonl(self.stage_log, record)

    def item_skip(
        self, item: str, *, reason: str = "", **details: Any  # noqa: ANN401
    ) -> None:
        # Skipped items don't count toward ok/fail tallies.
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "slug": self.slug,
            "stage": self.stage,
            "item": item,
            "status": "skip",
            "reason": reason,
        }
        record.update(details)
        _append_jsonl(self.stage_log, record)


@contextlib.contextmanager
def stage(slug: str, stage_name: str) -> Iterator[StageLogger]:  # noqa: ANN401
    """Backwards-friendly ctx manager wrapper."""
    sl = StageLogger(slug, stage_name)
    with sl:
        yield sl


# ---------------------------------------------------------------------------
# Read-side helpers (consumed by ragctl status/tail commands)
# ---------------------------------------------------------------------------

def read_status_lines(limit: int | None = None) -> list[dict[str, Any]]:
    path = LOGS_DIR / "pipeline_status.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[-limit:]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def latest_per_slug() -> dict[str, dict[str, Any]]:
    """For each slug, return the most recent status event (across all stages)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in read_status_lines():
        slug = rec.get("slug", "?")
        out[slug] = rec  # last write wins (file is append-only)
    return out


def stage_log_path(slug: str, stage_name: str) -> Path:
    return LOGS_DIR / f"{slug}_{stage_name}.jsonl"
