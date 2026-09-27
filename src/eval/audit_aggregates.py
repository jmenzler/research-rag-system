# long-ok-file
"""Audit-trail aggregates: MMR activation, drop overlap, latency percentiles.

Pure roll-ups over the audit directories already on disk. No LLM calls.
All functions are deterministic and accept a list of audit directory paths
(strings) plus optional golden reference data.

Backward compat: when a directory is missing or the relevant stage file
is absent, the function degrades gracefully (skips the directory, returns
``None`` when no data is available).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MMR activation rate
# ---------------------------------------------------------------------------


def mmr_activation_rate(audit_dirs: list[str]) -> float | None:
    """Fraction of queries where MMR actually swapped at least one chunk.

    Reads ``04_rerank.json`` from each audit directory. A query is counted
    as "MMR active" when its rerank trace contains a non-null ``mmr`` block
    with a non-empty ``dropped`` list. Missing directories or rerank files
    are counted as not active.

    Returns ``None`` when there are no directories to compute over.
    """
    if not audit_dirs:
        return None

    active = 0
    total = 0
    for ad in audit_dirs:
        total += 1
        data = _load_json(Path(ad) / "04_rerank.json")
        if data is None:
            continue
        mmr = data.get("mmr")
        if mmr and mmr.get("dropped"):
            active += 1

    if total == 0:
        return None
    return active / total


# ---------------------------------------------------------------------------
# MMR drop overlap with golden reference chunks
# ---------------------------------------------------------------------------


def mmr_drop_overlap(
    audit_dirs: list[str],
    golden_rows: list[dict[str, Any]],
) -> float | None:
    """Fraction of MMR-dropped chunks that appear in goldens' reference_chunk_ids.

    ``mmr.dropped`` holds Milvus *child* chunk IDs, while golden
    ``reference_chunk_ids`` are *parent* chunk IDs — different namespaces. Each
    dropped child is resolved to its parent via the ``survivors`` trace (which
    carries ``child_id`` → ``parent_chunk_id`` for the full rerank pool the MMR
    drops came from), then the resolved parents are intersected with the pooled
    golden refs. A high overlap (>0.3) indicates MMR is discarding relevant
    content; a low overlap (<0.05) means MMR is doing its job (throwing out
    redundant near-duplicates).

    Returns ``None`` when there are no MMR-dropped chunks or no golden refs.
    """
    if not audit_dirs or not golden_rows:
        return None

    # Pool all golden ref chunk IDs.
    all_refs: set[str] = set()
    for row in golden_rows:
        refs = row.get("reference_chunk_ids") or []
        all_refs.update(refs)

    if not all_refs:
        return None

    # Collect MMR-dropped IDs, resolved child→parent via the survivors trace.
    dropped_parents: list[str] = []
    for ad in audit_dirs:
        data = _load_json(Path(ad) / "04_rerank.json")
        if data is None:
            continue
        mmr = data.get("mmr")
        if not mmr:
            continue
        d = mmr.get("dropped") or []
        if not d:
            continue
        cid_to_pid = _survivor_child_to_parent(data)
        # Fall back to the raw id when no mapping exists (older traces without a
        # survivors block, where dropped already carries parent-shaped IDs).
        dropped_parents.extend(cid_to_pid.get(cid, cid) for cid in d)

    if not dropped_parents:
        return None

    overlap = sum(1 for pid in dropped_parents if pid in all_refs)
    return overlap / len(dropped_parents)


def _survivor_child_to_parent(rerank_data: dict[str, Any]) -> dict[str, str]:
    """Build child_id → parent_chunk_id from a 04_rerank.json survivors trace."""
    mapping: dict[str, str] = {}
    for s in rerank_data.get("survivors") or []:
        cid = s.get("child_id")
        pid = s.get("parent_chunk_id")
        if cid and pid:
            mapping[cid] = pid
    return mapping


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------


# Stage keys in meta.json#totals that carry latency_ms values.
_STAGE_LATENCY_KEYS: tuple[str, ...] = (
    "retrieval_latency_ms",
    "synthesis_latency_ms",
)

# Map totals keys to short stage names for the output dict.
_STAGE_NAME_MAP: dict[str, str] = {
    "retrieval_latency_ms": "retrieval",
    "synthesis_latency_ms": "synthesis",
    "rerank_latency_ms": "rerank",
    "milvus_latency_ms": "milvus",
}


def latency_percentiles(
    audit_dirs: list[str],
) -> dict[str, dict[str, float]]:
    """Compute median and p95 latency per stage across audit directories.

    Reads ``meta.json#totals`` from each directory and extracts latency
    keys matching ``*_latency_ms`` (retrieval, synthesis, rerank, milvus).

    Returns:
        ``{"latency_p50_ms": {stage: ms, ...}, "latency_p95_ms": {stage: ms, ...}}``.
        Empty inner dicts when no data is found.
    """
    # Accumulate per-stage latencies.
    acc: dict[str, list[float]] = {}

    for ad in audit_dirs:
        meta = _load_json(Path(ad) / "meta.json")
        if meta is None:
            continue
        totals = meta.get("totals") or {}
        # Collect all latency keys from totals.
        for key in list(totals.keys()):
            if not key.endswith("_latency_ms"):
                continue
            val = totals.get(key)
            if val is None:
                continue
            try:
                ms = float(val)
            except (TypeError, ValueError):
                continue
            stage_name = _STAGE_NAME_MAP.get(key, key.replace("_latency_ms", ""))
            acc.setdefault(stage_name, []).append(ms)

    p50: dict[str, float] = {}
    p95: dict[str, float] = {}
    for stage, values in acc.items():
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        p50[stage] = float(np.percentile(arr, 50))
        p95[stage] = float(np.percentile(arr, 95))

    return {"latency_p50_ms": p50, "latency_p95_ms": p95}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on any failure."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("audit_aggregates: failed to read %s: %s", path, exc)
        return None
