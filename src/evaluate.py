"""Evaluation runner for the RAG system.

Loads a golden Q&A file, runs retrieve -> generate for each question, evaluates
with our custom RAGAS-equivalent metrics (faithfulness, answer_relevancy,
context_precision, context_recall) via ``src.eval.runner``, writes a
timestamped JSON report to ``logs/``, and prints a pass/fail table.

CLI usage::

    python -m src.evaluate --notebook example_topic --collection trading
    python -m src.evaluate --notebook example_topic --collection trading \\
        --golden /path/to/custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import (
    GEMINI_API_KEY,
    JUDGE_MODEL,
    RAGAS_THRESHOLDS,
)
from src.query.query_pipeline import run_query_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _golden_path(notebook: str) -> Path:
    """Return the default golden file path for *notebook*."""
    return _PROJECT_ROOT / "golden" / f"{notebook}.jsonl"


def _logs_dir() -> Path:
    """Return the logs directory, creating it if necessary."""
    d = _PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_golden(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL golden file.

    Each line must be a JSON object with at minimum ``question`` and
    ``ground_truth`` keys.

    Args:
        path: Absolute path to the ``.jsonl`` file.

    Returns:
        List of parsed row dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any line is missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj: dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Golden file {path}:{lineno} -- invalid JSON: {exc}"
                ) from exc
            for key in ("question", "ground_truth"):
                if key not in obj:
                    raise ValueError(
                        f"Golden file {path}:{lineno} -- missing required key {key!r}"
                    )
            rows.append(obj)

    if not rows:
        raise ValueError(f"Golden file {path} is empty -- no questions to evaluate.")

    return rows


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def _run_one(
    i: int,
    n: int,
    row: dict[str, Any],
    notebook: str,
    collection: str,
    use_stepback: bool | None,
    use_crag: bool | None,
) -> dict[str, Any]:
    """Run the pipeline for one golden question and return an eval row dict.

    Extracted so it can be called both sequentially (workers=1) and from a
    ThreadPoolExecutor worker (workers>1). Prints elapsed time after the
    pipeline call completes so the timing is accurate and the print is
    non-interleaved with other workers' prints at the character level.

    Args:
        i:           1-based index for display.
        n:           Total number of golden questions.
        row:         Golden row dict with ``question``, ``ground_truth``, and optional ``notebook``.
        notebook:    Default notebook tag (overridden by row["notebook"] if present).
        collection:  Milvus collection name.
        use_stepback: Stepback flag forwarded to the pipeline.
        use_crag:     CRAG flag forwarded to the pipeline.

    Returns:
        A dict with keys ``question``, ``answer``, ``contexts``, ``ground_truth``.
    """
    from src.models import RAGResponse  # noqa: PLC0415

    question: str = row["question"]
    ground_truth: str = row["ground_truth"]
    row_notebook: str = row.get("notebook") or notebook

    t0 = time.perf_counter()
    response, _trace = run_query_pipeline(
        query=question,
        collection=collection,
        notebook=row_notebook,
        use_stepback=use_stepback,
        use_crag=use_crag,
    )
    elapsed_s = time.perf_counter() - t0

    # Print after the call completes so elapsed time is accurate.
    print(
        f"[evaluate] ({i}/{n}) done in {elapsed_s:.1f}s: {question[:80]!r}"
    )

    if response is None:
        response = RAGResponse(
            query=question,
            answer="(no chunks retrieved)",
            citations=[],
            retrieved=[],
            latency_ms=0,
        )

    return {
        "question": question,
        "answer": response.answer,
        "contexts": [r.parent.text for r in response.retrieved],
        "ground_truth": ground_truth,
    }


def _build_rows_from_pipeline(
    golden_rows: list[dict[str, Any]],
    notebook: str,
    collection: str,
    use_stepback: bool | None,
    use_crag: bool | None,
    pipeline_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run the full retrieve+generate pipeline for each golden question.

    Slow path. Used when no cached pipeline output is available, or the
    caller wants fresh answers.

    Args:
        golden_rows:      List of golden Q&A dicts.
        notebook:         Default notebook tag.
        collection:       Milvus collection name.
        use_stepback:     Forward to pipeline.
        use_crag:         Forward to pipeline.
        pipeline_workers: Number of concurrent pipeline executions. Default 1
                          (sequential, bit-for-bit identical to pre-parallel
                          behavior). Values ≥ 2 use ThreadPoolExecutor.
                          Independent of --max-workers (judge concurrency).
    """
    n = len(golden_rows)

    if pipeline_workers <= 1:
        # Sequential path — unchanged behavior.
        return [
            _run_one(i, n, row, notebook, collection, use_stepback, use_crag)
            for i, row in enumerate(golden_rows, start=1)
        ]

    # Parallel path: submit all questions, collect in original order.
    results: list[dict[str, Any] | None] = [None] * n
    with ThreadPoolExecutor(
        max_workers=pipeline_workers,
        thread_name_prefix="eval-pipe",
    ) as ex:
        future_to_idx = {
            ex.submit(_run_one, i + 1, n, row, notebook, collection, use_stepback, use_crag): i
            for i, row in enumerate(golden_rows)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()  # Propagates worker exceptions to caller.

    assert all(r is not None for r in results), "ThreadPoolExecutor returned holes in results"
    return results  # type: ignore[return-value]  # assertion above guarantees no None


def _build_rows_from_jsonl(
    golden_rows: list[dict[str, Any]],
    jsonl_path: Path,
    collection: str,
) -> list[dict[str, Any]]:
    """Reuse cached pipeline outputs from ``logs/queries.jsonl``.

    For each golden question, picks the *most recent* matching row by exact
    question text. Builds eval contexts from:

    - ``contexts`` field if present (rows logged after we started persisting it)
    - else ``retrieved_ids`` → Milvus query for ``parent_chunk_id`` → parents
      sqlite for the parent text (backfill for older rows)

    Raises ``ValueError`` if any golden question has no matching cached row.
    """
    from src.milvus_client import get_client  # noqa: PLC0415
    from src.query.retrieve import _lookup_parents  # noqa: PLC0415

    if not jsonl_path.exists():
        raise FileNotFoundError(f"queries.jsonl not found: {jsonl_path}")

    # Index queries.jsonl by question text, keeping the most recent row per query.
    cached: dict[str, dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            cached[rec["query"]] = rec  # later rows overwrite earlier (most-recent wins)

    print(
        f"[evaluate] loaded {len(cached)} unique queries from {jsonl_path} "
        f"(most-recent wins)"
    )

    eval_rows: list[dict[str, Any]] = []
    needs_backfill: list[tuple[int, dict[str, Any]]] = []
    milvus_client = None

    for idx, row in enumerate(golden_rows):
        question: str = row["question"]
        ground_truth: str = row["ground_truth"]
        rec = cached.get(question)
        if rec is None:
            raise ValueError(
                f"No cached pipeline output found for golden question {idx + 1}: "
                f"{question[:80]!r}. Drop --from-jsonl or run the pipeline first."
            )

        contexts: list[str]
        if "contexts" in rec and rec["contexts"]:
            contexts = list(rec["contexts"])
        else:
            # Backfill via Milvus + parents.sqlite. Defer the lookup to a batched
            # pass below so we don't hit Milvus once per row.
            contexts = []
            needs_backfill.append((len(eval_rows), rec))

        eval_rows.append(
            {
                "question": question,
                "answer": rec["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    if needs_backfill:
        milvus_client = get_client()
        # Collect all child IDs across rows that need backfill, deduped.
        all_child_ids: set[str] = set()
        for _, rec in needs_backfill:
            all_child_ids.update(rec["retrieved_ids"])
        print(
            f"[evaluate] backfilling contexts for {len(needs_backfill)} cached rows "
            f"({len(all_child_ids)} unique child chunks via {collection})..."
        )
        results = milvus_client.query(
            collection_name=collection,
            ids=list(all_child_ids),
            output_fields=["parent_chunk_id"],
        )
        child_to_parent: dict[str, str] = {r["id"]: r["parent_chunk_id"] for r in results}

        all_parent_ids = list({pid for pid in child_to_parent.values()})
        from src.config import PARENTS_DB  # noqa: PLC0415

        parents = _lookup_parents(parent_ids=all_parent_ids, db_path=PARENTS_DB)

        for row_idx, rec in needs_backfill:
            ctx: list[str] = []
            for child_id in rec["retrieved_ids"]:
                pid = child_to_parent.get(child_id)
                if pid is None:
                    continue
                pc = parents.get(pid)
                if pc is None:
                    continue
                ctx.append(pc.text)
            eval_rows[row_idx]["contexts"] = ctx
            if not ctx:
                print(
                    f"[evaluate] WARN: row {row_idx + 1} backfilled 0 contexts "
                    f"(child IDs may have been re-indexed)"
                )

    return eval_rows


def evaluate_notebook(
    notebook: str,
    collection: str,
    golden_path: str | None = None,
    limit: int | None = None,
    use_stepback: bool | None = None,
    use_crag: bool | None = None,
    max_workers: int = 4,
    from_jsonl: str | None = None,
    pipeline_workers: int = 1,
    extras: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Run evaluation for *notebook* against the *collection*.

    Pipeline per golden question:
      1. ``src.retrieve.hybrid_search`` -> ``list[RetrievedChunk]``
      2. ``src.generate.generate`` -> ``RAGResponse``
      3. Collect question, answer, contexts (parent texts), ground_truth.
    Then scores the full batch with our custom RAGAS-equivalent metrics
    (``src.eval.runner``) and writes a report.

    Args:
        notebook:     Notebook tag (e.g. ``"example_topic"``).
        collection:   Milvus collection name (e.g. ``"trading"``).
        golden_path:  Optional explicit path to a ``.jsonl`` file. Defaults to
                      ``golden/<notebook>.jsonl`` relative to the project root.
        extras:       Dict of opt-in metric flags (e.g. ``{"citation": True}``).

    Returns:
        Report dict (same structure written to the JSON log file).

    Raises:
        FileNotFoundError: Golden file not found.
        ValueError:        Golden file malformed or empty.
        RuntimeError:      Required API key not set.
        Exception:         Any judge / Gemini API error (re-raised, never
                           swallowed).
    """
    # --- load golden data ---
    gpath = Path(golden_path) if golden_path else _golden_path(notebook)
    golden_rows = _load_golden(gpath)
    if limit is not None and limit > 0:
        golden_rows = golden_rows[:limit]
    n = len(golden_rows)

    flag_summary = (
        f" stepback={use_stepback if use_stepback is not None else 'cfg'}"
        f" crag={use_crag if use_crag is not None else 'cfg'}"
    )
    source = f" from_jsonl={from_jsonl}" if from_jsonl else ""
    print(
        f"[evaluate] notebook={notebook!r} collection={collection!r} n={n}"
        f"{flag_summary}{source}"
        + (" (limited from full set)" if limit else "")
    )

    # --- build eval rows: cache replay or fresh pipeline ---
    if from_jsonl is not None:
        eval_rows = _build_rows_from_jsonl(
            golden_rows=golden_rows,
            jsonl_path=Path(from_jsonl),
            collection=collection,
        )
    else:
        eval_rows = _build_rows_from_pipeline(
            golden_rows=golden_rows,
            notebook=notebook,
            collection=collection,
            use_stepback=use_stepback,
            use_crag=use_crag,
            pipeline_workers=pipeline_workers,
        )

    # --- run our schema-enforced metrics ---
    import asyncio as _asyncio  # noqa: PLC0415

    from src.eval.runner import run_our_metrics  # noqa: PLC0415

    print(
        f"[evaluate] running our metrics max_workers={max_workers} "
        f"(schema-enforced, no parse-retry storms)..."
    )
    result = _asyncio.run(
        run_our_metrics(
            rows=eval_rows,
            judge_model=JUDGE_MODEL,
            max_workers=max_workers,
            extras=extras,
            collection=collection,
            notebook=notebook,
        )
    )

    # Collect all metric names present in the result (core + any extras).
    all_metric_names = list(result["aggregate"].keys())
    extra_names = [n for n in all_metric_names if n not in _METRIC_NAMES]

    aggregate: dict[str, float] = {
        m: round(float(result["aggregate"][m]), 4) for m in all_metric_names
    }
    passed = {m: aggregate[m] > RAGAS_THRESHOLDS[m] for m in _METRIC_NAMES}
    per_question: list[dict[str, Any]] = [
        {
            "question": row["question"],
            "answer": row["answer"],
            "scores": {m: round(float(scores[m]), 4) for m in all_metric_names},
        }
        for row, scores in zip(eval_rows, result["per_question"])
    ]
    all_passed = all(passed.values())

    # --- stage recall + audit aggregates (always-on, from audit dirs) ---
    stage_recall_rows: list[dict[str, Any] | None] = []
    audit_dirs_for_aggs: list[str] = []

    # Read queries.jsonl to get audit_dir for each golden question.
    queries_log = _PROJECT_ROOT / "logs" / "queries.jsonl"
    if queries_log.exists():
        cached: dict[str, dict[str, Any]] = {}
        with queries_log.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                cached[rec["query"]] = rec

        # Pass 1: gather audit_dir + per-row child_ids from 03_milvus.json so
        # we can do a single batched Milvus child→parent lookup across all
        # rows instead of N round-trips.
        from src.eval.stage_recall import (  # noqa: PLC0415
            _read_milvus_child_ids,
            compute_stage_recall,
        )

        per_row_audit: list[tuple[dict[str, Any], str | None, list[str]]] = []
        for g_row in golden_rows:
            rec = cached.get(g_row["question"])
            audit_dir = rec.get("audit_dir") if rec else None
            cids = _read_milvus_child_ids(audit_dir) if audit_dir else []
            per_row_audit.append((g_row, audit_dir, cids))
            if audit_dir:
                audit_dirs_for_aggs.append(audit_dir)

        # Pass 2: one batched Milvus query for ALL rows' child_ids at once.
        global_c2p: dict[str, str] = {}
        rows_with_cids = [cids for _, _, cids in per_row_audit if cids]
        if rows_with_cids:
            from src.retrieval_metrics import (  # noqa: PLC0415
                _resolve_children_to_parents,
            )

            try:
                parents_per_row = _resolve_children_to_parents(
                    rows_with_cids, collection
                )
                # Flatten into a single child→parent dict shared across rows.
                for cids, pids in zip(rows_with_cids, parents_per_row):
                    for cid, pid in zip(cids, pids):
                        if pid:
                            global_c2p[cid] = pid
            except Exception:
                # Milvus may be unreachable — stage recall degrades gracefully.
                pass

        # Pass 3: compute stage recall per row using the shared c2p mapping.
        for g_row, audit_dir, _cids in per_row_audit:
            if audit_dir:
                ref_ids = g_row.get("reference_chunk_ids") or []
                sr = compute_stage_recall(
                    audit_dir,
                    ref_ids,
                    child_to_parent=global_c2p or None,
                )
                stage_recall_rows.append(sr)
            else:
                stage_recall_rows.append(None)

    # Compute audit aggregates from the collected dirs.
    audit_agg: dict[str, Any] | None = None
    if audit_dirs_for_aggs:
        from src.eval.audit_aggregates import (  # noqa: PLC0415
            latency_percentiles,
            mmr_activation_rate,
            mmr_drop_overlap,
        )

        mmr_rate = mmr_activation_rate(audit_dirs_for_aggs)
        mmr_drop = mmr_drop_overlap(audit_dirs_for_aggs, golden_rows)
        latencies = latency_percentiles(audit_dirs_for_aggs)

        audit_agg = {
            "mmr_activation_rate": mmr_rate,
            "mmr_drop_overlap_with_golden": mmr_drop,
            **latencies,
        }

    # --- assemble report ---
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "notebook": notebook,
        "collection": collection,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "n_questions": n,
        "metrics": aggregate,
        "thresholds": RAGAS_THRESHOLDS,
        "passed": passed,
        "all_passed": all_passed,
        "per_question": per_question,
    }
    if any(sr is not None for sr in stage_recall_rows):
        report["stage_recall"] = stage_recall_rows
    if audit_agg is not None:
        report["audit_aggregates"] = audit_agg

    # --- write log ---
    log_file = _logs_dir() / f"eval_{notebook}_{timestamp}.json"
    log_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[evaluate] report written to {log_file}")

    # --- print pass/fail table ---
    _print_table(aggregate, passed, extra_names)

    # --- print stage recall summary ---
    valid_sr = [sr for sr in stage_recall_rows if sr is not None]
    if valid_sr:
        _print_stage_recall(valid_sr)

    # --- print audit aggregates ---
    if audit_agg is not None:
        _print_audit_aggregates(audit_agg)

    return report


def _print_table(
    aggregate: dict[str, float], passed: dict[str, bool], extra_names: list[str] | None = None
) -> None:
    """Print a formatted pass/fail table to stdout."""
    extra_names = extra_names or []
    header = f"{'Metric':<25} {'Score':>7}  {'Threshold':>10}  {'Status':>6}"
    print()
    print(header)
    print("-" * len(header))
    for metric, score in aggregate.items():
        threshold = RAGAS_THRESHOLDS.get(metric, 0.0)
        if metric in _METRIC_NAMES:
            status = "PASS" if passed[metric] else "FAIL"
        else:
            status = "—"
        print(f"{metric:<25} {score:>7.4f}  {threshold:>10.2f}  {status:>6}")
    print()
    overall = "ALL PASSED" if all(passed.values()) else "FAILED"
    print(f"Overall: {overall}")
    print()


def _print_stage_recall(stage_recall_rows: list[dict[str, Any]]) -> None:
    """Print a stage recall summary table."""
    # Pick the first non-None row to get the keys.
    sr = stage_recall_rows[0]
    keys = [k for k in sr if k != "question"]
    if not keys:
        return

    # Compute aggregate means across rows.
    print("Stage Recall (chunk-id recall@k):")
    header = f"{'Stage':<30}" + "".join(f" {'Mean':>7}" for _ in keys)
    print(header)
    print("-" * len(header))
    agg_row = ""
    for key in keys:
        vals = [float(r[key]) for r in stage_recall_rows if r.get(key) is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        agg_row += f" {mean:>7.4f}"
    stage_names = {
        "recall_at_retrieve_k20": "Post-Milvus (retrieve)",
        "recall_at_rerank_k20": "Post-rerank",
        "recall_at_final_k20": "Post-MMR (final)",
    }
    for key in keys:
        label = stage_names.get(key, key)
        vals = [float(r[key]) for r in stage_recall_rows if r.get(key) is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"{label:<30} {mean:>7.4f}")
    print()


def _print_audit_aggregates(audit_agg: dict[str, Any]) -> None:
    """Print MMR activation + latency aggregates."""
    print("Audit Aggregates:")
    mmr_rate = audit_agg.get("mmr_activation_rate")
    if mmr_rate is not None:
        print(f"  MMR activation rate:    {mmr_rate:.2%}")
    mmr_drop = audit_agg.get("mmr_drop_overlap_with_golden")
    if mmr_drop is not None:
        print(f"  MMR drop overlap:       {mmr_drop:.2%}")

    p50 = audit_agg.get("latency_p50_ms", {})
    if p50:
        print(f"  Latency p50: {', '.join(f'{s}={v:.0f}ms' for s, v in sorted(p50.items()))}")
    p95 = audit_agg.get("latency_p95_ms", {})
    if p95:
        print(f"  Latency p95: {', '.join(f'{s}={v:.0f}ms' for s, v in sorted(p95.items()))}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Run RAGAS-equivalent evaluation for a given notebook.",
    )
    parser.add_argument(
        "--notebook",
        required=True,
        help="Notebook tag (e.g. example_topic).  Matches golden/<notebook>.jsonl.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection name (e.g. trading, ecology, notes, system).",
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="Optional explicit path to a .jsonl golden file.",
    )
    parser.add_argument(
        "-N",
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions from the golden set (debug / cost control).",
    )
    parser.add_argument(
        "--stepback",
        action="store_true",
        default=None,
        help="Enable step-back prompting (1 abstract foundational sub-query alongside "
        "decomposition). Overrides config.USE_STEPBACK.",
    )
    parser.add_argument(
        "--no-stepback",
        action="store_false",
        dest="stepback",
        help="Force-disable step-back even if config.USE_STEPBACK is True.",
    )
    # CRAG defaults to OFF for eval. Rationale: CRAG retries fire conditionally
    # on grader scores, adding stochastic noise on top of any change being
    # measured. Two runs with identical retrieval can take different paths
    # because the grader scores answers slightly differently. For ablation
    # eval (the default use case here), that confounds attribution. Pass
    # --crag explicitly for production-parity validation runs that should
    # mirror the CLI behavior.
    parser.add_argument(
        "--crag",
        action="store_true",
        default=False,
        help="Enable CRAG-lite groundedness verification + retry-once on low "
        "score. Off by default in eval to keep ablation attribution clean. "
        "Pass for production-parity validation runs.",
    )
    parser.add_argument(
        "--no-crag",
        action="store_false",
        dest="crag",
        help="Explicitly disable CRAG-lite (this is the default; the flag exists "
        "for parity with --crag).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent metrics judge calls (default 4). Lower this if "
        "the judge provider returns 503/throttling under high concurrency.",
    )
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help="Path to logs/queries.jsonl. When set, skip the retrieve+generate "
        "pipeline and reuse cached pipeline outputs (matched by question text, "
        "most-recent wins). Older rows missing the 'contexts' field are "
        "backfilled via Milvus + parents.sqlite. Useful for re-running judging "
        "without paying the pipeline cost again.",
    )
    parser.add_argument(
        "--pipeline-workers",
        type=int,
        default=1,
        help="Concurrent pipeline executions over the golden set. Default 1 "
        "(sequential, bit-for-bit identical to pre-parallel behavior). "
        "Recommended 4 for ~2-3× wall-clock speedup on the trading collection. "
        "Independent of --max-workers (which controls metrics judge concurrency). "
        "Higher values may saturate the reranker GPU and DeepSeek rate limits.",
    )
    parser.add_argument(
        "--with-noise-sensitivity",
        action="store_true",
        default=False,
        help="Opt-in: run BM25-adversarial noise injection + re-synthesize + "
        "delta-faithfulness per row. Adds ~$0.05/eval at N=10.",
    )
    parser.add_argument(
        "--with-aspect-citation",
        action="store_true",
        default=False,
        help="Opt-in: run citation accuracy LLM judge per row (fraction of "
        "citations that correctly attribute claims). Adds ~$0.02/eval at N=10.",
    )
    parser.add_argument(
        "--with-aspect-disambiguation",
        action="store_true",
        default=False,
        help="Opt-in: run concept disambiguation LLM judge per row (binary pass/fail). "
        "Adds ~$0.02/eval at N=10.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m src.evaluate``."""
    args = _parse_args(argv)

    # Configure logging once at the entry point. Without this, all logger.info
    # output (including per-row metric progress in src.eval.runner) is silently
    # dropped because Python's default level is WARNING. Override via env var
    # LOG_LEVEL (e.g. DEBUG, WARNING) for noisier or quieter runs.
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate API key early so the error is clear before any slow imports.
    if not os.getenv("GEMINI_API_KEY") and not GEMINI_API_KEY:
        print(
            "ERROR: GEMINI_API_KEY is not set. "
            "Export it in your shell or add it to .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    extras_flags: dict[str, bool] = {}
    if getattr(args, "with_aspect_citation", False):
        extras_flags["citation"] = True
    if getattr(args, "with_aspect_disambiguation", False):
        extras_flags["disambiguation"] = True
    if getattr(args, "with_noise_sensitivity", False):
        extras_flags["noise"] = True

    report = evaluate_notebook(
        notebook=args.notebook,
        collection=args.collection,
        golden_path=args.golden,
        limit=args.limit,
        use_stepback=args.stepback,
        use_crag=args.crag,
        max_workers=args.max_workers,
        from_jsonl=args.from_jsonl,
        pipeline_workers=args.pipeline_workers,
        extras=extras_flags if extras_flags else None,
    )

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
