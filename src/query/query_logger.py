# long-ok-file
"""Per-query audit-trail writer.

Every query through ``_cli`` in ``generate.py`` gets its own directory::

    logs/queries/YYYY-MM-DD/HHMMSSZ_<sha256[:8]>/
    ├── meta.json          ← written last (finalize)
    ├── 01_decompose.json
    ├── 02_stepback.json   ← optional
    ├── 03_milvus.json
    ├── 04_rerank.json
    ├── 05_mmr.json        ← only if USE_MMR=True
    ├── 06_synthesis.json  ← only when synthesis ran
    ├── 07_grader.json     ← only if USE_CRAG_LITE=True
    ├── 08_crag_retry.json ← only if CRAG triggered a retry
    └── chunks/
        ├── 01_<child_id>.txt
        └── ...

``write_stage`` is exception-safe — the directory is preserved with whatever
stages completed when a crash occurs (partial trace is better than nothing).
``finalize`` is always called at the end (``finally`` block in ``_cli``), even
on crash, so ``meta.json`` is written with partial totals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models import RetrievedChunk, Usage

logger = logging.getLogger(__name__)

# Stage name → numeric prefix (kept here so the list serves as documentation).
_STAGE_PREFIX: dict[str, str] = {
    "decompose": "01",
    "stepback": "02",
    "milvus": "03",
    "rerank": "04",
    "mmr": "05",
    "synthesis": "06",
    "grader": "07",
    "crag_retry": "08",
    # Phase 2 — multi-retriever sub-stages (locked by 02-01-LIBRARY-SPIKE.md).
    # Prefix 09 reserved for a future Phase 1 follow-up.
    # PaperQA2-style retriever (Plan 02-06):
    "paperqa_rerank": "10",
    "paperqa_read": "11",
    "paperqa_answer": "12",
    # HippoRAG/LightRAG-style retriever (Plan 02-07):
    "hipporag_entities": "13",
    "hipporag_walk": "14",
    "hipporag_synthesize": "15",
    # LazyGraphRAG-style retriever (Plan 02-08):
    "lazygraph_route": "16",
    "lazygraph_community": "17",
    "lazygraph_summarize": "18",
    # Prefix 19 reserved.
    # Fused aggregator (Plan 02-09):
    "fused_dispatch": "20",
    "fused_rerank": "21",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _safe_git_sha() -> str | None:
    """Return short git SHA of HEAD, or None on failure."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(_PROJECT_ROOT),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


class QueryLogger:
    """Audit-trail writer for one query execution.

    Create one instance per query (``_cli`` instantiates it at the top of the
    pipeline) and call ``write_stage`` after each pipeline stage completes.
    Call ``finalize`` in a ``finally`` block to ensure ``meta.json`` is written
    even on crash.

    Thread-safety: not thread-safe — each query runs in its own process/thread
    and gets its own instance.
    """

    def __init__(
        self,
        query: str,
        root: str | Path = "logs/queries",
        *,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Set up the per-query directory.

        Args:
            query:           The original user query string.
            root:            Base directory for all query audit logs.
            config_snapshot: Pipeline config dict included in meta.json.
        """
        self.query = query
        self.config_snapshot = config_snapshot or {}
        self.ts_started = datetime.now(tz=UTC)

        # Resolve root relative to project root when it's not absolute.
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = _PROJECT_ROOT / root_path

        date_str = self.ts_started.strftime("%Y-%m-%d")
        # Microsecond precision prevents same-second collisions. PID + TID are
        # appended to guarantee uniqueness even when two threads start within
        # the same microsecond on the same query string (forced-collision case
        # in parallel eval with --pipeline-workers > 1).
        # Format: HHMMSSffffffZ_<hash8>_<pid>_<tid>
        time_str = self.ts_started.strftime("%H%M%S%f") + "Z"
        qhash = hashlib.sha256(query.encode()).hexdigest()[:8]
        self.pid = os.getpid()
        self.tid = threading.get_ident()
        dir_name = f"{time_str}_{qhash}_{self.pid}_{self.tid}"

        self.log_dir = root_path / date_str / dir_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "chunks").mkdir(exist_ok=True)

        # In-memory accumulators. The lock guards the read-modify-write in
        # accumulate_usage so the parallel fused fan-out (pipeline_workers > 1)
        # cannot lose _n_llm_calls increments under the GIL bytecode boundary.
        self._accumulate_lock = threading.Lock()
        self._usage_by_stage: dict[str, Usage] = {}
        self._cost_by_stage: dict[str, float | None] = {}
        self._latency_by_stage: dict[str, int] = defaultdict(int)
        self._n_llm_calls: int = 0
        self._retrieval_latency_ms: int = 0
        self._synthesis_latency_ms: int = 0

        # Outcome flags (set by caller).
        self.n_chunks_to_synth: int = 0
        self.synthesis_skipped: bool = False
        self.crag_retried: bool = False
        self.crag_chose: str | None = None  # "original" | "retry" | None

        logger.debug("QueryLogger: writing to %s", self.log_dir)

    # ------------------------------------------------------------------
    # Stage writing
    # ------------------------------------------------------------------

    def write_stage(self, name: str, payload: dict[str, Any]) -> None:
        """Serialize *payload* to ``<NN>_<name>.json`` inside the log directory.

        Unknown stage names get prefix "99" so they still land somewhere
        sensible without crashing.
        """
        prefix = _STAGE_PREFIX.get(name, "99")
        filename = f"{prefix}_{name}.json"
        path = self.log_dir / filename
        try:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("QueryLogger.write_stage(%r) failed: %s", name, exc)

    def write_chunk(self, idx: int, rc: RetrievedChunk) -> None:
        """Write one parent chunk text to ``chunks/<NN>_<child_id>.txt``.

        Header format (6 lines, then blank line, then parent text)::

            child_id: <id>
            parent_chunk_id: <id>
            source_file: <path>
            page: <N>
            rerank_score: <float>
            modality: <str>

            <parent text>
        """
        nn = str(idx).zfill(2)
        # Sanitise child_id for use as a filename component.
        safe_id = rc.child.id.replace("/", "_").replace("\\", "_")[:40]
        filename = f"{nn}_{safe_id}.txt"
        path = self.log_dir / "chunks" / filename
        header = (
            f"child_id: {rc.child.id}\n"
            f"parent_chunk_id: {rc.child.parent_id}\n"
            f"source_file: {rc.parent.source_file}\n"
            f"page: {rc.parent.page_number}\n"
            f"rerank_score: {rc.rerank_score:.6f}\n"
            f"modality: {rc.parent.modality}\n"
        )
        try:
            path.write_text(header + "\n" + rc.parent.text, encoding="utf-8")
        except Exception as exc:
            logger.warning("QueryLogger.write_chunk idx=%d failed: %s", idx, exc)

    # ------------------------------------------------------------------
    # Usage accumulation
    # ------------------------------------------------------------------

    def accumulate_usage(
        self, stage: str, usage: Usage, cost_usd: float | None, latency_ms: int = 0
    ) -> None:
        """Record LLM usage from one stage for aggregation into meta.json totals."""
        with self._accumulate_lock:
            self._usage_by_stage[stage] = usage
            self._cost_by_stage[stage] = cost_usd
            self._latency_by_stage[stage] = latency_ms
            self._n_llm_calls += 1

    def set_retrieval_latency(self, ms: int) -> None:
        self._retrieval_latency_ms = ms

    def set_synthesis_latency(self, ms: int) -> None:
        self._synthesis_latency_ms = ms

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(
        self,
        *,
        total_latency_ms: int,
        reranker_identity: dict[str, str | None] | None = None,
    ) -> None:
        """Write ``meta.json`` with aggregated totals.

        Should be called in a ``finally`` block so it runs even on crash.
        Partial data is better than no data.
        """
        ts_ended = datetime.now(tz=UTC)

        # Sum token usage across all LLM stages.
        total_input = sum(u.input for u in self._usage_by_stage.values())
        total_output = sum(u.output for u in self._usage_by_stage.values())
        total_reasoning = sum(u.reasoning for u in self._usage_by_stage.values())
        total_tokens = sum(u.total for u in self._usage_by_stage.values())
        total_cost_parts = [c for c in self._cost_by_stage.values() if c is not None]
        total_cost: float | None = sum(total_cost_parts) if total_cost_parts else None

        meta: dict[str, Any] = {
            "query": self.query,
            "query_hash": hashlib.sha256(self.query.encode()).hexdigest()[:16],
            "ts_started": self.ts_started.isoformat(),
            "ts_ended": ts_ended.isoformat(),
            "git_sha": _safe_git_sha(),
            "host": socket.gethostname(),
            "pid": self.pid,
            "tid": self.tid,
            "config": self.config_snapshot,
            "reranker": reranker_identity,
            "totals": {
                "total_latency_ms": total_latency_ms,
                "retrieval_latency_ms": self._retrieval_latency_ms,
                "synthesis_latency_ms": self._synthesis_latency_ms,
                "n_llm_calls": self._n_llm_calls,
                "tokens": {
                    "input": total_input,
                    "output": total_output,
                    "reasoning": total_reasoning,
                    "total": total_tokens,
                },
                "cost_usd": total_cost,
            },
            "outcome": {
                "n_chunks_to_synth": self.n_chunks_to_synth,
                "synthesis_skipped": self.synthesis_skipped,
                "crag_retried": self.crag_retried,
                "crag_chose": self.crag_chose,
            },
        }

        path = self.log_dir / "meta.json"
        try:
            path.write_text(
                json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("QueryLogger.finalize: failed to write meta.json: %s", exc)

        # Render single-page Markdown audit report alongside meta.json.
        # Inline import keeps the renderer optional and avoids a circular dep.
        try:
            from src.query.report_render import render_report  # noqa: PLC0415

            report = render_report(self.log_dir)
            (self.log_dir / "report.md").write_text(report, encoding="utf-8")
        except Exception as exc:
            logger.warning("QueryLogger.finalize: failed to write report.md: %s", exc)

    @property
    def dir_path(self) -> Path:
        """Absolute path to this query's log directory."""
        return self.log_dir


def make_query_logger(
    query: str,
    *,
    config_snapshot: dict[str, Any] | None = None,
) -> QueryLogger:
    """Factory that resolves the log root from the ``QUERY_LOG_ROOT`` env var.

    Falls back to ``logs/queries`` (project-relative) when the env var is not set.
    """
    root = os.getenv("QUERY_LOG_ROOT", "logs/queries")
    return QueryLogger(query, root=root, config_snapshot=config_snapshot)
