# long-ok-file
"""Stage-decomposed chunk-id recall@k from per-query audit directories.

Reads the audit files written by ``QueryLogger`` (03_milvus.json,
04_rerank.json) and computes recall@k of golden ``reference_chunk_ids`` at
three pipeline boundaries: post-Milvus retrieval, post-cross-encoder rerank,
and post-MMR diversity filter. The deltas between stages show where relevant
chunks are being lost — critical for tuning the retrieval-to-rerank ratio,
the cross-encoder score threshold, and the MMR lambda sweep.

All computation is deterministic (no LLM calls). When ``audit_dir`` is missing
or stage files are empty/unreadable, the function returns ``None`` so the eval
report can degrade gracefully (same pattern as ``retrieval_metrics.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit file readers
# ---------------------------------------------------------------------------


def _read_milvus_child_ids(audit_dir: str) -> list[str]:
    """Read 03_milvus.json and return deduped child chunk IDs in retrieval order.

    Child IDs are collected across all sub-queries. Order is preserved
    (first appearance wins), so the list reflects the actual retrieval
    ranking before cross-encoder reranking.
    """
    path = Path(audit_dir) / "03_milvus.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("stage_recall: failed to read %s: %s", path, exc)
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for sq in data.get("sub_queries", []) or []:
        for cand in sq.get("candidates", []) or []:
            cid = cand.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                ordered.append(cid)
    return ordered


def _read_rerank_parent_ids(audit_dir: str) -> list[str]:
    """Read 04_rerank.json and return parent_chunk_ids from survivors.

    Survivors are in cross-encoder score order (highest first). Parent IDs
    are already present in the survivor trace entries (resolved at retrieval
    time), so no Milvus lookup is needed for this stage.
    """
    path = Path(audit_dir) / "04_rerank.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("stage_recall: failed to read %s: %s", path, exc)
        return []

    pids: list[str] = []
    for s in data.get("survivors", []) or []:
        pid = s.get("parent_chunk_id")
        if pid and pid not in pids:
            pids.append(pid)
    return pids


def _read_final_parent_ids(audit_dir: str) -> list[str]:
    """Read the post-MMR (or post-rerank when MMR is off) parent_chunk_ids.

    When 04_rerank.json contains an ``mmr.picks`` block, returns the
    parent_chunk_ids of the MMR-selected chunks. Otherwise returns the
    survivors list (which is post-rerank, pre-MMR — i.e. the final set
    when MMR did not fire).

    Each pick carries ``parent_chunk_id`` directly. As a fallback for older
    audit dirs we join via the survivors list.
    """
    path = Path(audit_dir) / "04_rerank.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("stage_recall: failed to read %s: %s", path, exc)
        return []

    survivors = data.get("survivors", []) or []
    mmr = data.get("mmr")

    if mmr and mmr.get("picks"):
        cid_to_pid: dict[str, str] = {}
        for s in survivors:
            cid = s.get("child_id")
            pid = s.get("parent_chunk_id")
            if cid and pid:
                cid_to_pid[cid] = pid

        picks = mmr.get("picks", []) or []
        seen: set[str] = set()
        ordered: list[str] = []
        for pick in picks:
            pid = pick.get("parent_chunk_id") or cid_to_pid.get(pick.get("child_id") or "")
            if pid and pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered

    return _read_rerank_parent_ids(audit_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_stage_recall(
    audit_dir: str,
    reference_chunk_ids: list[str],
    *,
    child_to_parent: dict[str, str] | None = None,
    k: int = 20,
) -> dict[str, float | None] | None:
    """Compute recall@k of *reference_chunk_ids* at three pipeline stages.

    Args:
        audit_dir:            Path to the per-query audit directory (from
                              ``queries.jsonl#audit_dir``).
        reference_chunk_ids:  Parent chunk IDs that the golden answer was
                              grounded in.
        child_to_parent:      Optional mapping from Milvus child chunk IDs to
                              parent chunk IDs. When ``None``, the Milvus
                              stage is skipped (returned as ``None``). Useful
                              for testing without Milvus; the eval path
                              resolves this via a batched Milvus query.
        k:                    Truncation depth for recall@k.

    Returns:
        A dict with keys ``recall_at_retrieve_k<N>``,
        ``recall_at_rerank_k<N>``, ``recall_at_final_k<N>``, or ``None`` if
        the audit directory does not exist or is unreadable. Individual stage
        values are ``None`` when the corresponding stage file is missing.
    """
    from src.retrieval_metrics import recall_at_k  # noqa: PLC0415

    # No ground-truth chunk IDs → recall is undefined, not 0.0. recall_at_k
    # would return a concrete 0.0, which the eval roll-up averages in (it only
    # filters None), tanking stage recall for any notebook without golden
    # reference_chunk_ids.
    if not reference_chunk_ids:
        return None

    ad = Path(audit_dir)
    if not ad.is_dir() or audit_dir == "":
        return None

    milvus_child_ids = _read_milvus_child_ids(audit_dir)
    rerank_parent_ids = _read_rerank_parent_ids(audit_dir)
    final_parent_ids = _read_final_parent_ids(audit_dir)

    result: dict[str, float | None] = {}

    # Milvus stage: resolve child IDs → parent IDs via the injected mapping.
    # Without a mapping, we cannot compute recall (child IDs and parent IDs
    # are in different namespaces).
    key = f"recall_at_retrieve_k{k}"
    if milvus_child_ids and child_to_parent is not None:
        milvus_parents = [
            pid for cid in milvus_child_ids if (pid := child_to_parent.get(cid))
        ]
        result[key] = recall_at_k(milvus_parents, reference_chunk_ids, k)
    elif milvus_child_ids:
        # Child IDs available but no mapping — cannot compute.
        result[key] = None
    else:
        result[key] = None

    # Rerank stage: parent IDs are already in the survivor trace.
    key = f"recall_at_rerank_k{k}"
    if rerank_parent_ids:
        result[key] = recall_at_k(rerank_parent_ids, reference_chunk_ids, k)
    else:
        result[key] = None

    # Final stage: post-MMR (or post-rerank when MMR is off).
    key = f"recall_at_final_k{k}"
    if final_parent_ids:
        result[key] = recall_at_k(final_parent_ids, reference_chunk_ids, k)
    else:
        result[key] = None

    return result
