"""Deterministic retrieval metrics (no LLM judge).

Two metric levels, computed side-by-side:

- **Source-file level (headline):** did the retriever surface chunks from the
  paper that contains the answer? Inputs are basenamed source-file paths.
  Retriever-agnostic — works for Milvus, PaperQA2, HippoRAG, LazyGraphRAG,
  and any future retriever that emits a ranked source list.
- **Parent-chunk-id level (within-collection diagnostic):** did the retriever
  surface the *specific* parent chunks the golden was grounded in? Useful for
  chunker tuning within one collection; broken across collections because
  ``chunk_id = sha256(source_file|page|chunk_index|kind)`` invalidates on any
  chunker change that reorders chunks within a page.

### Inputs

- ``retrieved_sources``: rank-ordered, deduped, basenamed source files from the
  retriever (preferred — one field every retriever can produce). Read from
  ``logs/queries.jsonl``.
- ``retrieved_ids``: rank-ordered child chunk IDs (Milvus path only). Used as a
  fallback to derive ``retrieved_sources`` when absent.
- ``reference_source_files``: source files the golden was grounded in (basenames).
- ``reference_chunk_ids``: parent chunk IDs the golden was grounded in.

### Retriever contract

Any retriever that wants to be evaluated by this module must produce
``retrieved_sources: list[str]`` in rank order, with paths basenamed and
order-preserving deduped. The eval log writer is the integration point —
see ``src/query/generate.py:_append_log`` for the Milvus path.

### Why not RAGAS for this

RAGAS' LLMContextPrecisionWithReference (issue #1905) compares chunks to the
*answer*, not the *query*, and produces unstable binary verdicts on technical
text. Set-intersection metrics on grounded chunk IDs / source files answer the
operational question deterministically, at $0/run, judge-swap-immune.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def recall_at_k(retrieved_parents: list[str], reference: list[str], k: int) -> float:
    """Fraction of reference chunks present in the top-k retrieved.

    Args:
        retrieved_parents: Parent chunk IDs in rank order.
        reference:         Set of parent chunk IDs the question was grounded in.
        k:                 Truncation depth.

    Returns:
        |retrieved[:k] ∩ reference| / |reference|, or 0.0 if reference is empty.
    """
    if not reference:
        return 0.0
    top_k = set(retrieved_parents[:k])
    ref = set(reference)
    return len(top_k & ref) / len(ref)


def mrr_at_k(retrieved_parents: list[str], reference: list[str], k: int) -> float:
    """Reciprocal rank of the first reference chunk in the top-k retrieved.

    Returns 0.0 if no reference chunk appears in the top-k.
    """
    if not reference:
        return 0.0
    ref = set(reference)
    for rank, pid in enumerate(retrieved_parents[:k], start=1):
        if pid in ref:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_parents: list[str], reference: list[str], k: int) -> float:
    """Did *any* reference chunk appear in the top-k? Binary 0/1."""
    if not reference:
        return 0.0
    return 1.0 if set(retrieved_parents[:k]) & set(reference) else 0.0


def _norm_source(path: str) -> str:
    """Normalize a source-file path to a stable per-document key.

    Layout: ``sources/<collection>/<slug>/{source.pdf,pdf.txt,nlm.txt,...}``.

    Steps:
      1. Strip a trailing generic filename (``source.pdf`` / ``source.txt`` /
         ``nlm.txt`` / ``pdf.txt`` / ``content_list.json``) so the parent dir
         is the identity.
      2. Strip a ``.txt`` suffix when only a leaf basename was passed.

    Result: ``optimal_high_frequency_market_making_stanford.txt`` and
    ``sources/trading/optimal_high_frequency_market_making_stanford/source.pdf``
    both normalise to ``optimal_high_frequency_market_making_stanford``.
    """
    base = os.path.basename(path)
    if base in {"source.pdf", "source.txt", "nlm.txt", "pdf.txt", "content_list.json"}:
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            base = parent
    if base.endswith(".txt"):
        base = base[:-4]
    return base


def _dedupe_ranked(items: list[str]) -> list[str]:
    """Order-preserving dedupe — first appearance wins (so rank is preserved)."""
    return list(dict.fromkeys(items))


def source_recall_at_k(
    retrieved_sources: list[str], reference_sources: list[str], k: int
) -> float:
    """Fraction of reference source files present in the top-k retrieved.

    Inputs are expected basenamed; this function normalizes defensively to
    tolerate callers that pass full paths.
    """
    if not reference_sources:
        return 0.0
    top_k = {_norm_source(s) for s in retrieved_sources[:k]}
    ref = {_norm_source(s) for s in reference_sources}
    return len(top_k & ref) / len(ref)


def source_mrr_at_k(
    retrieved_sources: list[str], reference_sources: list[str], k: int
) -> float:
    """Reciprocal rank of the first reference source in the top-k retrieved."""
    if not reference_sources:
        return 0.0
    ref = {_norm_source(s) for s in reference_sources}
    for rank, src in enumerate(retrieved_sources[:k], start=1):
        if _norm_source(src) in ref:
            return 1.0 / rank
    return 0.0


def source_hit_at_k(
    retrieved_sources: list[str], reference_sources: list[str], k: int
) -> float:
    """Did *any* reference source appear in the top-k? Binary 0/1."""
    if not reference_sources:
        return 0.0
    top_k = {_norm_source(s) for s in retrieved_sources[:k]}
    ref = {_norm_source(s) for s in reference_sources}
    return 1.0 if top_k & ref else 0.0


def _resolve_children_to_parents_and_sources(
    child_ids_by_query: list[list[str]],
    collection: str,
) -> tuple[list[list[str]], list[list[str]]]:
    """Map each query's child chunk IDs to ``(parent_ids, basenamed_sources)``
    via a single batched Milvus query, preserving rank order.

    Returns two parallel lists, both rank-ordered and order-preserving deduped:
    one with parent chunk IDs (chunk-level metric), one with basenamed source
    files (source-level metric). Children that can't be resolved (rare —
    corrupted log or re-indexed collection) are silently dropped.
    """
    from src.milvus_client import get_client  # noqa: PLC0415

    all_children: set[str] = set()
    for ids in child_ids_by_query:
        all_children.update(ids)

    if not all_children:
        return [[] for _ in child_ids_by_query], [[] for _ in child_ids_by_query]

    client = get_client()
    rows: list[dict[str, Any]] = client.query(
        collection_name=collection,
        ids=list(all_children),
        output_fields=["parent_chunk_id", "source_file"],
    )
    child_to_parent: dict[str, str] = {r["id"]: r["parent_chunk_id"] for r in rows}
    child_to_source: dict[str, str] = {
        r["id"]: _norm_source(r["source_file"]) for r in rows
    }

    parents_out: list[list[str]] = []
    sources_out: list[list[str]] = []
    for ids in child_ids_by_query:
        seen_p: set[str] = set()
        ordered_p: list[str] = []
        seen_s: set[str] = set()
        ordered_s: list[str] = []
        for cid in ids:
            pid = child_to_parent.get(cid)
            if pid is not None and pid not in seen_p:
                seen_p.add(pid)
                ordered_p.append(pid)
            src = child_to_source.get(cid)
            if src is not None and src not in seen_s:
                seen_s.add(src)
                ordered_s.append(src)
        parents_out.append(ordered_p)
        sources_out.append(ordered_s)
    return parents_out, sources_out


def _resolve_children_to_parents(
    child_ids_by_query: list[list[str]],
    collection: str,
) -> list[list[str]]:
    """Backward-compat wrapper — returns only parent IDs.

    Kept for any callers outside this module. Internal code uses the joint
    resolver above to avoid two Milvus round-trips.
    """
    parents, _ = _resolve_children_to_parents_and_sources(
        child_ids_by_query, collection
    )
    return parents


def evaluate_from_logs(
    golden_path: Path,
    queries_log_path: Path,
    collection: str,
    k_values: tuple[int, ...] = (5, 10, 15),
) -> dict[str, Any]:
    """Compute source-level + chunk-level retrieval metrics from cached logs.

    Joins ``golden_path`` (questions + ``reference_source_files`` and/or
    ``reference_chunk_ids``) with the most recent matching row per question
    in ``queries_log_path``.

    For each cached row:

    - **Source-level (headline, retriever-agnostic):** uses ``retrieved_sources``
      directly when present. Otherwise falls back to the Milvus path:
      child IDs → ``source_file`` (basenamed, deduped, rank-preserving).
    - **Chunk-level (within-collection diagnostic):** uses cached ``retrieved_ids``
      → parent IDs via Milvus. Skipped silently when the log has no chunk IDs
      (e.g. a graph retriever).

    Per-question score blocks omit metrics whose reference field is missing
    on the golden row, so partially-annotated goldens still produce a report.
    Aggregates average only over questions that have the relevant reference.

    Raises ``ValueError`` if any golden question lacks a cached row.
    """
    goldens = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]

    cached: dict[str, dict[str, Any]] = {}
    for line in queries_log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cached[rec["query"]] = rec  # most-recent wins

    cached_rows: list[dict[str, Any]] = []
    for g in goldens:
        rec = cached.get(g["question"])
        if rec is None:
            raise ValueError(f"No cached row for: {g['question'][:80]!r}")
        cached_rows.append(rec)

    # Identify which rows need a Milvus fallback (no retrieved_sources OR need
    # parent-id metric). We always pre-resolve via Milvus when chunk IDs are
    # present and the eval may need them — single batched call covers all.
    needs_milvus_idxs: list[int] = [
        i for i, rec in enumerate(cached_rows) if rec.get("retrieved_ids")
    ]
    parent_lists_by_idx: dict[int, list[str]] = {}
    source_lists_by_idx: dict[int, list[str]] = {}
    if needs_milvus_idxs:
        child_lists_resolve = [cached_rows[i]["retrieved_ids"] for i in needs_milvus_idxs]
        parents, sources = _resolve_children_to_parents_and_sources(
            child_lists_resolve, collection
        )
        for idx, p, s in zip(needs_milvus_idxs, parents, sources):
            parent_lists_by_idx[idx] = p
            source_lists_by_idx[idx] = s

    per_question: list[dict[str, Any]] = []
    for i, (g, rec) in enumerate(zip(goldens, cached_rows)):
        # Source-level: prefer log's retrieved_sources; fall back to Milvus-derived.
        if "retrieved_sources" in rec and rec["retrieved_sources"] is not None:
            retrieved_sources = [_norm_source(s) for s in rec["retrieved_sources"]]
            retrieved_sources = list(dict.fromkeys(retrieved_sources))
        else:
            retrieved_sources = source_lists_by_idx.get(i, [])

        retrieved_parents = parent_lists_by_idx.get(i, [])

        ref_sources_raw = g.get("reference_source_files") or []
        ref_sources = list(dict.fromkeys(_norm_source(s) for s in ref_sources_raw))
        ref_chunks = g.get("reference_chunk_ids") or []

        scores: dict[str, float] = {}
        for k in k_values:
            if ref_sources:
                scores[f"src_recall@{k}"] = source_recall_at_k(retrieved_sources, ref_sources, k)
                scores[f"src_mrr@{k}"] = source_mrr_at_k(retrieved_sources, ref_sources, k)
                scores[f"src_hit@{k}"] = source_hit_at_k(retrieved_sources, ref_sources, k)
            if ref_chunks:
                scores[f"recall@{k}"] = recall_at_k(retrieved_parents, ref_chunks, k)
                scores[f"mrr@{k}"] = mrr_at_k(retrieved_parents, ref_chunks, k)
                scores[f"hit@{k}"] = hit_at_k(retrieved_parents, ref_chunks, k)

        per_question.append(
            {
                "question": g["question"],
                "n_reference_sources": len(ref_sources),
                "n_reference_chunks": len(ref_chunks),
                "n_retrieved_sources": len(retrieved_sources),
                "n_retrieved_parents": len(retrieved_parents),
                "scores": {m: round(v, 4) for m, v in scores.items()},
            }
        )

    def _avg(metric_key: str) -> float | None:
        vals = [pq["scores"][metric_key] for pq in per_question if metric_key in pq["scores"]]
        return round(sum(vals) / len(vals), 4) if vals else None

    source_aggregate: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    for k in k_values:
        for m in ("recall", "mrr", "hit"):
            sk = f"src_{m}@{k}"
            v = _avg(sk)
            if v is not None:
                source_aggregate[sk] = v
            ck = f"{m}@{k}"
            v = _avg(ck)
            if v is not None:
                aggregate[ck] = v

    return {
        "n_questions": len(per_question),
        "k_values": list(k_values),
        "source_aggregate": source_aggregate,
        "aggregate": aggregate,
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m src.retrieval_metrics --golden ... --collection ...``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval_metrics",
        description="Compute deterministic retrieval metrics (recall@k, mrr@k, hit@k) "
        "from cached pipeline outputs in logs/queries.jsonl. Joins by question text.",
    )
    parser.add_argument(
        "--golden",
        required=True,
        help="Path to golden .jsonl with reference_chunk_ids.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection (for child→parent map).",
    )
    parser.add_argument(
        "--queries-log",
        default=str(_PROJECT_ROOT / "logs" / "queries.jsonl"),
        help="Path to logs/queries.jsonl (default: <repo>/logs/queries.jsonl).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="k values to score (default: 5 10 15).",
    )
    parser.add_argument("--json", action="store_true", help="Emit full report as JSON to stdout.")
    args = parser.parse_args(argv)

    report = evaluate_from_logs(
        golden_path=Path(args.golden),
        queries_log_path=Path(args.queries_log),
        collection=args.collection,
        k_values=tuple(args.k),
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"\n[retrieval_metrics] N={report['n_questions']} questions, k={report['k_values']}\n")

    if report["source_aggregate"]:
        print("Source-level (headline, retriever-agnostic):")
        print(f"  {'metric':<16}  {'score':>7}")
        print("  " + "-" * 26)
        for k in report["k_values"]:
            for m in ("hit", "recall", "mrr"):
                key = f"src_{m}@{k}"
                if key in report["source_aggregate"]:
                    print(f"  {key:<16}  {report['source_aggregate'][key]:>7.4f}")
            print()
    else:
        print("Source-level: no reference_source_files in golden — skipped.\n")

    if report["aggregate"]:
        print("Chunk-level (within-collection diagnostic):")
        print(f"  {'metric':<12}  {'score':>7}")
        print("  " + "-" * 22)
        for k in report["k_values"]:
            for m in ("hit", "recall", "mrr"):
                key = f"{m}@{k}"
                if key in report["aggregate"]:
                    print(f"  {key:<12}  {report['aggregate'][key]:>7.4f}")
            print()
    else:
        print("Chunk-level: no reference_chunk_ids or no log retrieved_ids — skipped.\n")

    # Per-question table — prefer source metrics if present, else chunk.
    use_source = bool(report["source_aggregate"])
    prefix = "src_" if use_source else ""
    print(f"Per-question ({'source' if use_source else 'chunk'}-level):")
    print(f"{'#':>3}  {'q[:60]':<60}  ", end="")
    print("  ".join(f"{f'r@{k}':>5}  {f'h@{k}':>5}  {f'mrr@{k}':>5}" for k in report["k_values"]))
    for i, pq in enumerate(report["per_question"], 1):
        line = f"{i:>3}  {pq['question'][:60]:<60}  "
        parts: list[str] = []
        for k in report["k_values"]:
            r = pq["scores"].get(f"{prefix}recall@{k}")
            h = pq["scores"].get(f"{prefix}hit@{k}")
            mrr = pq["scores"].get(f"{prefix}mrr@{k}")
            parts.append(
                f"{(r if r is not None else 0):>5.2f}  "
                f"{(h if h is not None else 0):>5.2f}  "
                f"{(mrr if mrr is not None else 0):>5.2f}"
            )
        line += "  ".join(parts)
        print(line)
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
