"""Orchestrator over the four metric functions.

``run_our_metrics(rows, judge_model, max_workers)`` is the entry point used
by ``src.evaluate``.

Row shape on input (what ``_build_rows_from_pipeline`` /
``_build_rows_from_jsonl`` produce):

    {
        "question":     str,
        "answer":       str,
        "contexts":     list[str],
        "ground_truth": str,
    }

Internally we adapt to the metric functions' shape (``user_input`` /
``response`` / ``retrieved_contexts`` / ``reference``) and fan out via
``asyncio.gather`` under a semaphore.

Returned dict::

    {
        "per_question": [
            {
                "question": str,
                "faithfulness": float,
                "answer_relevancy": float,
                "context_precision": float,
                "context_recall": float,
            },
            ...
        ],
        "aggregate": {
            "faithfulness": float,        # mean over non-NaN per-row scores
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float,
        },
    }
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from src.eval.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)

# Type alias for a metric callable: takes (adapted_row, judge_model) → float.
_MetricFn = Callable[[dict[str, Any], str], Awaitable[float]]

_METRIC_NAMES: tuple[str, str, str, str] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# Extras are opt-in metrics added conditionally via the ``extras`` dict.
# Each entry maps a short flag name → (metric_fn, output_field_name).
_EXTRA_FLAG_MAP: dict[str, tuple[str, str]] = {
    "citation": ("citation_accuracy", "citation_accuracy"),
    "disambiguation": ("concept_disambiguation", "concept_disambiguation"),
}


def _adapt_row(row: dict[str, Any]) -> dict[str, Any]:
    """Translate eval row keys (golden/pipeline shape) to metric input keys.

    eval row keys        → metric input keys
    ``question``         → ``user_input``
    ``answer``           → ``response``
    ``contexts``         → ``retrieved_contexts``
    ``ground_truth``     → ``reference``
    """
    return {
        "user_input": row["question"],
        "response": row["answer"],
        "retrieved_contexts": list(row.get("contexts", [])),
        "reference": row.get("ground_truth", ""),
    }


async def run_our_metrics(
    rows: list[dict[str, Any]],
    judge_model: str,
    max_workers: int = 4,
    extras: dict[str, bool] | None = None,
    collection: str | None = None,
    notebook: str | None = None,
) -> dict[str, Any]:
    """Run all four metrics over all rows.

    All metric calls (across rows × metrics) are serialized through a single
    ``asyncio.Semaphore(max_workers)``. ``max_workers`` is the maximum number
    of metric coroutines that can be *running* at once — it bounds peak LLM
    fan-out across the whole eval, not per-row.

    When *extras* flags are set, additional opt-in metrics are appended to
    the per-row ``asyncio.gather`` and their per-question + aggregate fields
    appear in the output.

    Noise sensitivity (``extras["noise"]``) is shape-different from aspect
    critics: it runs as a single batch *after* the per-row loop because it
    needs the baseline ``faithfulness`` already computed, and it returns a
    dict-of-fields rather than a single float. It also requires
    ``collection`` and ``notebook`` to drive BM25-adversarial chunk lookup;
    if either is missing the noise step is silently skipped.

    Returns a dict with ``per_question`` and ``aggregate`` keys (see module
    docstring). NaN per-row scores are excluded from the aggregate mean.
    """
    # Resolve extras: lazily import metric functions for enabled flags.
    extras = extras or {}
    extra_metric_fns: list[tuple[str, _MetricFn]] = []
    if extras:
        for flag_name, (mod_attr, field_name) in _EXTRA_FLAG_MAP.items():
            if extras.get(flag_name):
                # Import the metric function from src.eval.aspects.
                import importlib  # noqa: PLC0415

                aspects_mod = importlib.import_module("src.eval.aspects")
                fn = getattr(aspects_mod, mod_attr)
                extra_metric_fns.append((field_name, fn))

    sem = asyncio.Semaphore(max_workers)
    n_rows = len(rows)

    async def _wrap(
        coro_fn: _MetricFn, name: str, adapted: dict[str, Any], row_idx: int
    ) -> float:
        async with sem:
            t0 = time.perf_counter()
            try:
                score: float = await coro_fn(adapted, judge_model)
                logger.info(
                    "[metrics] (%d/%d) %s done in %.1fs → %.4f",
                    row_idx,
                    n_rows,
                    name,
                    time.perf_counter() - t0,
                    score,
                )
                return score
            except Exception as exc:
                logger.error(
                    "[metrics] (%d/%d) %s FAILED in %.1fs: %r",
                    row_idx,
                    n_rows,
                    name,
                    time.perf_counter() - t0,
                    exc,
                )
                raise

    def _coerce(score_or_exc: Any) -> float:  # noqa: ANN401
        """Convert a per-metric result into a float.

        ``return_exceptions=True`` means ``asyncio.gather`` returns the
        exception object instead of raising. Log it loudly and score nan so
        the rest of the eval continues. Schema-parse failures already return
        nan inside ``call_with_schema``; this branch handles unexpected errors
        (including transient errors that exhausted retries somehow without
        being caught downstream).
        """
        if isinstance(score_or_exc, BaseException):
            logger.error(
                "run_our_metrics: metric call raised — scoring as nan. "
                "Error: %r",
                score_or_exc,
            )
            return math.nan
        return float(score_or_exc)

    per_question: list[dict[str, Any]] = []
    # All metric field names in output order (core + extras).
    all_metric_names = list(_METRIC_NAMES) + [fn for fn, _ in extra_metric_fns]

    overall_t0 = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        row_t0 = time.perf_counter()
        logger.info(
            "[metrics] (%d/%d) starting: %r",
            i,
            n_rows,
            (row.get("question") or "")[:80],
        )
        adapted = _adapt_row(row)

        # Core metrics.
        core_coros = [
            _wrap(faithfulness, "faithfulness", adapted, i),
            _wrap(answer_relevancy, "answer_relevancy", adapted, i),
            _wrap(context_precision, "context_precision", adapted, i),
            _wrap(context_recall, "context_recall", adapted, i),
        ]
        # Extra metrics.
        extra_coros = [
            _wrap(fn, name, adapted, i) for name, fn in extra_metric_fns
        ]
        raw_scores = await asyncio.gather(
            *core_coros, *extra_coros, return_exceptions=True
        )
        scores = [_coerce(s) for s in raw_scores]

        entry: dict[str, Any] = {"question": row["question"]}
        # The scores list has core scores first, then extras in order.
        for idx, name in enumerate(all_metric_names):
            entry[name] = scores[idx]
        per_question.append(entry)

        # Compact log line: core metrics first.
        core_str = (
            f"F={scores[0]:.3f} AR={scores[1]:.3f} "
            f"CP={scores[2]:.3f} CR={scores[3]:.3f}"
        )
        if extra_metric_fns:
            extra_str = " ".join(
                f"{name}={s:.3f}"
                for s, (name, _) in zip(
                    scores[len(_METRIC_NAMES):], extra_metric_fns
                )
            )
            core_str += " " + extra_str
        logger.info(
            "[metrics] (%d/%d) done in %.1fs — %s",
            i,
            n_rows,
            time.perf_counter() - row_t0,
            core_str,
        )

    logger.info(
        "[metrics] all %d rows complete in %.1fs total",
        n_rows,
        time.perf_counter() - overall_t0,
    )

    # Noise sensitivity: post-loop because it needs baseline faithfulness from
    # the per-row entries we just built. Skipped silently when extras flag
    # absent or when collection/notebook were not provided by the caller.
    noise_aggregate_field: str | None = None
    if extras.get("noise") and collection and notebook:
        # Pass baseline faithfulness through to noise so it can compute deltas.
        rows_with_baseline = [
            {**row, "faithfulness": entry.get("faithfulness")}
            for row, entry in zip(rows, per_question)
        ]
        from src.eval.noise import compute_all_noise  # noqa: PLC0415

        noise_results = await compute_all_noise(
            rows=rows_with_baseline,
            judge_model=judge_model,
            collection=collection,
            notebook=notebook,
            extras=extras,
        )
        # Merge per-row noise dicts into per_question entries. compute_all_noise
        # returns one slot per row; None means injection failed for that row.
        for entry, nresult in zip(per_question, noise_results):
            if nresult is None:
                continue
            entry.update(nresult)
        noise_aggregate_field = "noise_delta_faithfulness"

    aggregate: dict[str, float] = {}
    for metric in all_metric_names:
        values = [
            float(entry[metric])
            for entry in per_question
            if entry[metric] is not None and not math.isnan(float(entry[metric]))
        ]
        aggregate[metric] = sum(values) / len(values) if values else math.nan

    if noise_aggregate_field is not None:
        deltas = [
            float(entry[noise_aggregate_field])
            for entry in per_question
            if noise_aggregate_field in entry
            and not math.isnan(float(entry[noise_aggregate_field]))
        ]
        if deltas:
            aggregate[noise_aggregate_field] = sum(deltas) / len(deltas)

    return {"per_question": per_question, "aggregate": aggregate}
