# long-ok-file
"""Bulk corpus contextualization — produces `content_list.json.ctx.json` sidecars.

For each per-source content_list.json (without an existing sidecar, unless --all):
  1. Chunk via the SAME structural chunker (`chunk_sidecar`) ingest uses, so the
     (parent_idx, child_idx) keys align exactly with the children ingest emits.
  2. For each child chunk, generate a 1-2 sentence context summary anchored in
     the full document, using `src.contextualize.contextualize_document`.
  3. Write the sidecar `content_list.json.ctx.json` keyed by (parent_idx, child_idx).

ingest.py reads the sidecar and prepends each context to its chunk before
embedding — Anthropic Contextual Retrieval (Sept 2024) -35% retrieval failure.

Sidecar acts as the "done" marker (option b): present means contextualized,
absent means pending. Re-runs are cheap (skips done files). Force retrofit
with --all (deletes sidecars first).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.chunking import chunk_sidecar
from src.chunking.render import content_list_to_markdown
from src.chunking.walker import load_sidecar
from src.ingest.chunker import (
    _count_tokens,
    _get_encoder,
    _load_source_meta,
    iter_child_keys,
)
from src.ingest.contextualize import contextualize_document

ROOT = Path(__file__).resolve().parent.parent


def _sidecar_path(source: Path) -> Path:
    return source.with_suffix(source.suffix + ".ctx.json")


def _advance_source_contextualized(doc_dir: Path) -> None:
    """Best-effort: advance the source row to ``contextualized``.

    Missing meta.json or db errors log + continue — contextualization output
    is already on disk; bookkeeping must not block subsequent stages.
    """
    meta_path = doc_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        from src.ingest.sources import advance_status, canonical_id  # noqa: PLC0415
        from src.ingest.storage import _open_parents_db  # noqa: PLC0415

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            return
        cid, _ = canonical_id(meta)
        conn = _open_parents_db()
        try:
            advance_status(conn, cid, "contextualized")
        finally:
            conn.close()
    except Exception as exc:
        print(
            f"  ! advance_source_contextualized failed for {doc_dir}: "
            f"{type(exc).__name__}: {exc}"
        )


def _partition_of(source: Path) -> str:
    """Resolve collection (Milvus partition value) from source path.

    Post-flatten layout: ``sources/<collection>/<slug>/content_list.json`` →
    ``<collection>``. Falls back to the immediate parent name for any path
    not under ``sources/``.
    """
    parts = source.parts
    try:
        idx = parts.index("sources")
    except ValueError:
        return source.parent.name
    return parts[idx + 1] if idx + 1 < len(parts) else source.parent.name


def contextualize_file(source: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Build sidecar for one content_list.json. Returns stats dict.

    ``source`` is the doc's ``content_list.json``. We chunk it with the SAME
    structural chunker ingest uses (``chunk_sidecar``) so the emitted
    ``(parent_idx, child_idx)`` keys match the children ingest produces.
    """
    notebook = _partition_of(source)
    enc = _get_encoder()

    meta = _load_source_meta(source.parent)
    _title, proto_chunks = chunk_sidecar(source, enc, meta=meta)
    child_keys = iter_child_keys(proto_chunks)

    if dry_run:
        return {
            "file": str(source),
            "parents": len({p for p, _c, _t in child_keys}),
            "children": len(child_keys),
            "dry_run": True,
        }

    doc_text = content_list_to_markdown(load_sidecar(source))
    keys = [(p, c) for p, c, _t in child_keys]
    chunk_texts = [t for _p, _c, t in child_keys]

    started = time.monotonic()
    contexts, ctx_stats = contextualize_document(
        doc_text=doc_text,
        chunks=chunk_texts,
        notebook=notebook,
        doc_token_estimate=_count_tokens(doc_text, enc),
    )
    elapsed = time.monotonic() - started

    chunks_out = [
        {"parent_idx": p, "child_idx": c, "context": ctx}
        for (p, c), ctx in zip(keys, contexts)
    ]

    from src.ingest.contextualize import CONTEXTUALIZE_MODEL  # noqa: PLC0415

    sidecar = _sidecar_path(source)
    sidecar.write_text(json.dumps({
        "source": str(source.relative_to(ROOT)),
        "notebook": notebook,
        "n_parents": len({p for p, _c in keys}),
        "n_children": len(chunks_out),
        "model": CONTEXTUALIZE_MODEL,
        "used_cache": ctx_stats["used_cache"],
        "cached_input_tokens": ctx_stats["cached_input_tokens"],
        "chunks": chunks_out,
    }, indent=2))

    _advance_source_contextualized(source.parent)

    return {
        "file": str(source.relative_to(ROOT)),
        "parents": len({p for p, _c in keys}),
        "children": len(chunks_out),
        "elapsed_s": round(elapsed, 1),
        "cache": ctx_stats["used_cache"],
        "cached_tok": ctx_stats["cached_input_tokens"],
        "miss_tok": ctx_stats.get("cache_miss_tokens", 0),
        "out_tok": ctx_stats.get("completion_tokens", 0),
        "reason_tok": ctx_stats.get("reasoning_tokens", 0),
        "usd": ctx_stats.get("estimated_usd_cost", 0.0),
    }


def _run_one_subprocess(source_path: str, dry_run: bool) -> dict[str, Any]:
    """Module-level worker for ProcessPoolExecutor.

    Each subprocess imports src.contextualize fresh → its own genai.Client,
    its own httpx pool, its own threads. No shared state across procs.
    """
    return contextualize_file(Path(source_path), dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--paths",
        nargs="+",
        default=["sources/*/*/content_list.json"],
        help="Glob(s) for per-source content_list.json files. Default: every collection's docs.",
    )
    ap.add_argument(
        "--all", action="store_true", help="Force retrofit: delete existing sidecars first."
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan only; don't call the model.")
    ap.add_argument("--limit", type=int, default=None, help="Cap files processed (smoke testing).")
    ap.add_argument("--slug", default=None,
                    help="slug for observability (auto-derived from --paths if omitted).")
    args = ap.parse_args()

    # Derive slug for observability from the first --paths pattern when possible.
    # `sources/<collection>/...` patterns yield <collection>; fall back to _corpus_.
    obs_slug = args.slug
    if not obs_slug:
        first = args.paths[0]
        parts = Path(first).parts
        if len(parts) >= 2 and parts[0] == "sources" and parts[1] not in ("*", ""):
            obs_slug = parts[1]
        else:
            obs_slug = "_corpus_"

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    stage_logger = StageLogger(obs_slug, "contextualize")
    stage_logger.__enter__()

    files: list[Path] = []
    for pat in args.paths:
        files.extend(sorted(ROOT.glob(pat)))
    files = [
        f for f in files
        if f.name == "content_list.json"
        and not any(p.startswith("_") for p in f.relative_to(ROOT).parts)
    ]

    pending = [f for f in files if args.all or not _sidecar_path(f).exists()]
    if args.all:
        for f in pending:
            _sidecar_path(f).unlink(missing_ok=True)

    if args.limit:
        pending = pending[: args.limit]

    print(f"corpus: {len(files)} files | pending: {len(pending)} | dry_run={args.dry_run}")
    if not pending:
        print("nothing to do.")
        stage_logger.__exit__(None, None, None)
        return

    # Multiprocessing across files: server serializes per-cache_id at ~2 rps,
    # so within-file threading is capped. The lever is N independent processes,
    # each owning its own cache → N × per-cache rps.
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Was 4 (Gemini Tier 1 4M TPM cap); bumped to 8 for DeepSeek which has
    # no published RPM cap. Total concurrency = proc_workers × CHUNK_WORKERS
    # = 8 × 20 = 160 chunks in flight (semaphore floors at CTX_INFLIGHT=60).
    # Override via CTX_WORKERS env var if hitting issues.
    proc_workers = int(os.environ.get("CTX_WORKERS", "8"))
    print(f"contextualize: {proc_workers} parallel processes")

    stats: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=proc_workers) as pool:
        futures = {pool.submit(_run_one_subprocess, str(f), args.dry_run): f for f in pending}
        for fut in as_completed(futures):
            done += 1
            f = futures[fut]
            try:
                s = fut.result()
                if args.dry_run:
                    tag = "(dry)"
                else:
                    cache_tag = "C" if s.get("cache") else "-"
                    tag = (
                        f"{s.get('elapsed_s', 0):>5.1f}s "
                        f"{cache_tag}hit={s.get('cached_tok', 0):>7d} "
                        f"miss={s.get('miss_tok', 0):>5d} "
                        f"out={s.get('out_tok', 0):>4d}"
                        f"{('+r'+str(s.get('reason_tok', 0))) if s.get('reason_tok') else ''} "
                        f"${s.get('usd', 0.0):.4f}"
                    )
                print(
                    f"  [{done:>4d}/{len(pending)}] "
                    f"kids={s['children']:>3d} {tag} {s['file']}",
                    flush=True,
                )
                stats.append(s)
                stage_logger.item_ok(
                    Path(s["file"]).name,
                    elapsed_s=s.get("elapsed_s"),
                    children=s.get("children"),
                    cached_tok=s.get("cached_tok"),
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"  [{done:>4d}/{len(pending)}] FAIL {type(e).__name__}: {e} {f}", flush=True)
                stats.append({
                    "file": str(f.relative_to(ROOT)),
                    "error": f"{type(e).__name__}: {e}",
                })
                stage_logger.item_fail(f.name, reason=f"{type(e).__name__}: {e}")

    out = ROOT / "logs" / "contextualize_run.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(s) for s in stats))
    n_ok = sum(1 for s in stats if "error" not in s)
    total_usd = sum(s.get("usd", 0.0) for s in stats if "error" not in s)
    total_hit = sum(s.get("cached_tok", 0) for s in stats if "error" not in s)
    total_miss = sum(s.get("miss_tok", 0) for s in stats if "error" not in s)
    total_out = sum(s.get("out_tok", 0) for s in stats if "error" not in s)
    total_reason = sum(s.get("reason_tok", 0) for s in stats if "error" not in s)
    print(
        f"\nDONE: {n_ok}/{len(stats)} ok | "
        f"hit={total_hit:,} miss={total_miss:,} out={total_out:,}"
        f"{f' reasoning={total_reason:,}' if total_reason else ''} | "
        f"~${total_usd:.4f} | log: {out}"
    )
    stage_logger.__exit__(None, None, None)


if __name__ == "__main__":
    main()
