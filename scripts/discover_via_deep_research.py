"""Discover sources via NotebookLM Deep Research, then ingest into the RAG.

Treats NotebookLM as ephemeral compute:
  1. Create a temp notebook
  2. Trigger Deep Research with the user's query
  3. Wait for it to finish (web search + import-all)
  4. Drop sources we already have (cross-corpus dedup by canonical URL)
  5. Run our normal fetch + parse pipeline (`ragctl run`) against the temp
     notebook — the Spider hardlinks any already-cached artifacts, MinerU
     skips already-parsed PDFs (idempotency by content_list.json presence)
  6. Ingest into the user-specified collection under the `research_briefs`
     partition tag
  7. Delete the NotebookLM notebook

The NotebookLM CLI is auth'd on the PC; this script is intended to run on
the PC (the runtime that owns Milvus, parents.sqlite, the URL cache, and
MinerU). Run it directly there or via `ragctl discover` on either machine
(ragctl shells out, so on Mac you'd want to SSH in).

Usage:
    python -m scripts.discover_via_deep_research \\
        --query "How does LVR scale with pool depth?" \\
        --collection trading \\
        [--mode deep|fast] [--timeout 1800] [--keep]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.paths import COLLECTIONS  # noqa: E402
from src.fetch.classify import canonical_url  # noqa: E402

SOURCES_DIR = ROOT / "sources"
DEFAULT_PARTITION_TAG = "research_briefs"


def _max_parallel_dr() -> int:
    return int(os.getenv("MAX_PARALLEL_DR", "4"))


# ---------------------------------------------------------------------------
# notebooklm CLI helpers (subprocess wrappers)
# ---------------------------------------------------------------------------


def _nlm(
    args: list[str], *, capture: bool = False, timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run a notebooklm subcommand. Streams to terminal unless capture=True."""
    cmd = ["notebooklm", *args]
    return subprocess.run(
        cmd, check=False, timeout=timeout, text=True,
        capture_output=capture,
    )


def _nlm_error_detail(r: subprocess.CompletedProcess[str]) -> str:
    """Extract a human-actionable message from a failed notebooklm --json call.

    The CLI emits its error as a JSON body on stdout (stderr is often empty),
    e.g. an expired-session message ending 'Run notebooklm login to
    re-authenticate.' Surface that instead of a bare rc=N.
    """
    for stream in (r.stdout, r.stderr):
        if not stream:
            continue
        try:
            payload = json.loads(stream)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("error") and payload.get("message"):
            return str(payload["message"])
    return (r.stderr or r.stdout or "").strip()


def nlm_create(title: str) -> str:
    """Create a notebook, return its UUID."""
    r = _nlm(["create", title, "--json"], capture=True)
    if r.returncode != 0:
        detail = _nlm_error_detail(r)
        if "authentic" in detail.lower() or "notebooklm login" in detail.lower():
            raise RuntimeError(
                "NotebookLM session expired — re-auth on the PC via 'notebooklm login', "
                f"then retry. (notebooklm create rc={r.returncode}: {detail})"
            )
        raise RuntimeError(f"notebooklm create failed: rc={r.returncode}: {detail}")
    payload = json.loads(r.stdout)
    nb_id: str = payload["notebook"]["id"]
    return nb_id


def nlm_add_research(nb_id: str, query: str, *, mode: str) -> None:
    """Trigger Deep Research; --no-wait so we can show progress via research wait."""
    r = _nlm(
        [
            "source", "add-research", query,
            "--notebook", nb_id,
            "--mode", mode,
            "--import-all",
            "--no-wait",
        ],
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"notebooklm source add-research failed: rc={r.returncode}")


def nlm_research_wait(nb_id: str, *, timeout: int) -> None:
    """Block until research completes (notebooklm imports sources as it goes)."""
    r = _nlm(
        [
            "research", "wait",
            "--notebook", nb_id,
            "--timeout", str(timeout),
            "--import-all",
        ],
        timeout=timeout + 60,  # outer ceiling > inner timeout
    )
    if r.returncode != 0:
        raise RuntimeError(f"notebooklm research wait failed: rc={r.returncode}")


def nlm_source_list(nb_id: str) -> list[dict[str, Any]]:
    """Return imported sources after research finished."""
    r = _nlm(["source", "list", "--notebook", nb_id, "--json"], capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"notebooklm source list failed: rc={r.returncode}\n{r.stderr}")
    payload = json.loads(r.stdout)
    sources: list[dict[str, Any]] = payload.get("sources", [])
    return sources


def nlm_source_delete(nb_id: str, src_id: str) -> bool:
    """Delete a single source from the notebook. Returns True on success."""
    r = _nlm(
        ["source", "delete", src_id, "--notebook", nb_id, "--yes"],
        capture=True,
        timeout=30,
    )
    return r.returncode == 0


def nlm_delete(nb_id: str) -> bool:
    """Delete the entire notebook. Returns True on success."""
    r = _nlm(["delete", "--notebook", nb_id, "--yes"], capture=True, timeout=30)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Cross-corpus dedup
# ---------------------------------------------------------------------------


def existingcanonical_urls() -> set[str]:
    """Walk sources/*/*/meta.json across every collection, return canonical URLs.

    Reuses canonical_url from src.fetch.classify so the normalization rules
    (arxiv abs/pdf collapse, SSRN abstract_id, RG /publication/<id>, DOI lower)
    match the spider's own seed-time dedup.
    """
    if not SOURCES_DIR.is_dir():
        return set()
    seen: set[str] = set()
    for meta_p in SOURCES_DIR.glob("*/*/meta.json"):
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cu = canonical_url(meta)
        if cu:
            seen.add(cu)
    return seen


def filter_dups_via_nlm_delete(nb_id: str, sources: list[dict[str, Any]],
                                existing: set[str]) -> tuple[int, int]:
    """For each source already in our corpus, delete from NotebookLM.

    Returns (n_dups_deleted, n_kept).
    """
    n_del = 0
    n_keep = 0
    for src in sources:
        cu = canonical_url(src)
        if cu and cu in existing:
            ok = nlm_source_delete(nb_id, src["id"])
            if ok:
                n_del += 1
            else:
                # Best-effort: if NLM-side delete fails, the spider's seed-time
                # dedup will still catch it via canonical URL. Just count it
                # as kept so we don't lie about the dedup count.
                n_keep += 1
        else:
            n_keep += 1
    return n_del, n_keep


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------


def run_fetch_parse(nb_id: str, collection: str) -> int:
    """Invoke ragctl run --notebooks <UUID> --collection <COLL>.

    The fetch + parse stages run for the temp UUID; sources land in
    ``sources/<collection>/<slug>/``. Audit / contextualize / ingest are
    deferred to ``run_ingest`` so the temporary research-brief partition
    tag is applied at the ingest call.
    """
    cmd = [
        "uv", "run", str(ROOT / "bin" / "ragctl"), "run",
        "--collection", collection,
        "--notebooks", nb_id,
        "--stages", "fetch", "parse_pipeline", "parse_vlm",
    ]
    r = subprocess.run(cmd, cwd=ROOT, check=False)
    return r.returncode


def run_ingest(collection: str, partition: str, paths: list[str] | None = None) -> int:
    """Ingest docs into Milvus under ``partition``.

    ``paths`` should be the newly-fetched content_list.json files only.
    Falls back to the full collection glob when None (legacy / manual usage).
    The Milvus ``notebook`` column is set to ``partition`` so discover-time
    briefs land in their own partition within ``<collection>``.
    """
    path_args: list[str] = paths if paths else [f"sources/{collection}/*/content_list.json"]
    cmd = [
        "uv", "run", "python", "-m", "src.ingest.ingest",
        "--collection", collection,
        "--notebook", partition,
        "--paths", *path_args,
    ]
    r = subprocess.run(cmd, cwd=ROOT, check=False)
    return r.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_one_query(
    query: str, *, mode: str, timeout: int, existing: set[str],
) -> dict[str, Any]:
    """Run Deep Research for one query and return a per_query result dict.

    Does NOT call ragctl run / ingest — those happen once across all queries
    in ``run_discovery_bulk``. The notebook is left alive; the caller decides
    whether to delete based on overall success and ``--keep``.

    Returns keys: query, notebook_id, notebook_kept (False — caller updates),
    status, n_imported, n_deduped, n_kept, error.
    """
    pq: dict[str, Any] = {
        "query": query,
        "notebook_id": None,
        "notebook_kept": False,
        "status": "error",
        "n_imported": 0,
        "n_deduped": 0,
        "n_kept": 0,
        "error": None,
    }
    nb_title = f"discover: {query[:80]}"
    print(f"[discover:{query[:30]}] creating notebook…")
    try:
        nb_id = nlm_create(nb_title)
        pq["notebook_id"] = nb_id
    except Exception as exc:
        pq["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[discover:{query[:30]}] create FAIL: {pq['error']}", file=sys.stderr)
        return pq

    try:
        print(f"[discover:{query[:30]}] triggering Deep Research mode={mode}…")
        nlm_add_research(nb_id, query, mode=mode)
        print(f"[discover:{query[:30]}] waiting up to {timeout}s…")
        nlm_research_wait(nb_id, timeout=timeout)

        sources = nlm_source_list(nb_id)
        pq["n_imported"] = len(sources)
        if not sources:
            pq["status"] = "no_new_sources"
            return pq

        n_del, n_keep = filter_dups_via_nlm_delete(nb_id, sources, existing)
        pq["n_deduped"] = n_del
        pq["n_kept"] = n_keep
        if n_keep == 0:
            pq["status"] = "no_new_sources"
        else:
            pq["status"] = "ok"
        return pq
    except Exception as exc:
        pq["status"] = "error"
        pq["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[discover:{query[:30]}] FAIL: {pq['error']}", file=sys.stderr)
        return pq


def _run_fetch_parse_batch(nb_ids: list[str], collection: str) -> int:
    """Single ragctl run covering N notebooks. ``--notebooks`` takes nargs=+."""
    cmd = [
        "uv", "run", str(ROOT / "bin" / "ragctl"), "run",
        "--collection", collection,
        "--notebooks", *nb_ids,
        "--stages", "fetch", "parse_pipeline", "parse_vlm",
    ]
    r = subprocess.run(cmd, cwd=ROOT, check=False)
    return r.returncode


def run_discovery_bulk(
    queries: list[str],
    collection: str,
    *,
    mode: str,
    timeout: int,
    keep: bool,
    partition: str = DEFAULT_PARTITION_TAG,
    max_parallel: int | None = None,
    bulk_deadline_s: float | None = None,
) -> dict[str, Any]:
    """Run N Deep Research jobs in parallel, then a SINGLE ragctl+ingest pass.

    Aggregated result schema::

        {
          status:     "ok" | "no_new_sources" | "error",
          elapsed_s:  float,
          collection, partition, mode,
          queries:    list[str],
          per_query:  [{query, notebook_id, notebook_kept,
                        status, n_imported, n_deduped, n_kept, error}],
          n_imported, n_deduped, n_kept: int (summed across per_query),
          error?:     str (only when every query errored),
        }

    Top-level status is ``ok`` if at least one per_query is ok; otherwise
    ``no_new_sources`` if all returned that; otherwise ``error``.

    ``bulk_deadline_s`` (#36): wall-clock cap on awaiting per-query completion.
    When a single stuck IMPORT_RESEARCH retry-storms inside notebooklm, this
    cap unblocks the rest of the batch so fetch+parse fires for the queries
    that succeeded. Stuck queries are marked ``status="abandoned"`` in
    ``per_query`` and their threads orphaned (they keep running until the
    inner subprocess.run timeout terminates them). ``None`` = wait forever,
    legacy behaviour.
    """
    if not queries:
        raise ValueError("queries must be non-empty")

    t_start = time.time()
    cap = max_parallel if max_parallel is not None else _max_parallel_dr()
    n_workers = max(1, min(cap, len(queries)))

    print(f"[discover] computing cross-corpus dedup once for {len(queries)} queries…")
    existing = existingcanonical_urls()
    print(f"[discover]   {len(existing)} canonical URLs already in corpus")

    print(f"[discover] running {len(queries)} Deep Research jobs in parallel "
          f"(max_workers={n_workers}, bulk_deadline_s={bulk_deadline_s})…")

    ex = ThreadPoolExecutor(max_workers=n_workers)
    per_query: list[dict[str, Any]] = []
    try:
        futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {
            ex.submit(
                _run_one_query,
                q, mode=mode, timeout=timeout, existing=existing,
            ): q
            for q in queries
        }
        try:
            for fut in concurrent.futures.as_completed(
                futures, timeout=bulk_deadline_s
            ):
                per_query.append(fut.result())
        except concurrent.futures.TimeoutError:
            for fut, q in futures.items():
                if fut.done():
                    continue
                fut.cancel()
                per_query.append({
                    "query": q,
                    "notebook_id": None,
                    "notebook_kept": False,
                    "status": "abandoned",
                    "n_imported": 0,
                    "n_deduped": 0,
                    "n_kept": 0,
                    "error": (
                        f"bulk deadline {bulk_deadline_s}s exceeded; "
                        "fetch+parse proceeded for the queries that finished"
                    ),
                })
            print(
                f"[discover] bulk deadline {bulk_deadline_s}s exceeded — "
                f"abandoned {sum(1 for p in per_query if p['status'] == 'abandoned')} "
                "stuck query/queries (#36)",
                file=sys.stderr,
            )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    n_imp = sum(pq["n_imported"] for pq in per_query)
    n_dedup = sum(pq["n_deduped"] for pq in per_query)
    n_kept = sum(pq["n_kept"] for pq in per_query)

    statuses = {pq["status"] for pq in per_query}
    if "ok" in statuses:
        top_status = "ok"
    elif statuses == {"no_new_sources"}:
        top_status = "no_new_sources"
    else:
        top_status = "error"

    result: dict[str, Any] = {
        "status": top_status,
        "elapsed_s": 0.0,
        "collection": collection,
        "partition": partition,
        "mode": mode,
        "queries": list(queries),
        "per_query": per_query,
        "n_imported": n_imp,
        "n_deduped": n_dedup,
        "n_kept": n_kept,
    }
    if top_status == "error":
        errors = [pq["error"] for pq in per_query if pq["error"]]
        result["error"] = "; ".join(errors) if errors else "all queries failed"

    success_nb_ids = [
        pq["notebook_id"] for pq in per_query
        if pq["status"] == "ok" and pq["notebook_id"]
    ]
    def _append_error(msg: str) -> None:
        prior = result.get("error")
        result["error"] = f"{prior}; {msg}" if prior else msg

    if success_nb_ids:
        # Snapshot dirs that already have content_list.json before the fetch so
        # we can scope ingest to only the newly produced files afterwards.
        coll_dir = SOURCES_DIR / collection
        pre_fetch_dirs: set[Path] = set()
        if coll_dir.is_dir():
            pre_fetch_dirs = {p.parent for p in coll_dir.glob("*/content_list.json")}

        print(f"[discover] running batched fetch+parse for {len(success_nb_ids)} notebooks…")
        rc = _run_fetch_parse_batch(success_nb_ids, collection)
        if rc != 0:
            result["status"] = "error"
            _append_error(f"ragctl run failed: rc={rc}")
        else:
            # Only ingest content_list.json files that appeared during this run.
            new_paths: list[str] = []
            if coll_dir.is_dir():
                new_paths = [
                    str(p) for p in coll_dir.glob("*/content_list.json")
                    if p.parent not in pre_fetch_dirs
                ]
            if new_paths:
                print(
                    f"[discover] ingesting {len(new_paths)} new doc(s) "
                    f"into {collection}/{partition}…"
                )
            else:
                print("[discover] no new content_list.json found — nothing to ingest")
            rc = run_ingest(collection, partition, new_paths or None)
            if rc != 0:
                result["status"] = "error"
                _append_error(f"ingest failed: rc={rc}")

    # Cleanup: delete notebooks that finished successfully unless --keep.
    for pq in per_query:
        nb_id = pq["notebook_id"]
        if not nb_id:
            continue
        if pq["status"] == "error":
            pq["notebook_kept"] = True
            continue
        if keep:
            pq["notebook_kept"] = True
            continue
        if nlm_delete(nb_id):
            pq["notebook_kept"] = False
        else:
            pq["notebook_kept"] = True
            print(f"[discover] WARN: notebook delete failed for {nb_id}", file=sys.stderr)

    result["elapsed_s"] = round(time.time() - t_start, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.discover_via_deep_research",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--query", required=True, nargs="+",
        help="Deep Research query/queries — pass one or more, all run in parallel "
             "with a single batched fetch+parse pass.",
    )
    ap.add_argument("--collection", required=True,
                    choices=COLLECTIONS,
                    help="Milvus collection that gets the new chunks")
    ap.add_argument("--partition", default=DEFAULT_PARTITION_TAG,
                    help=(f"Within-collection partition tag for the new chunks "
                          f"(default: {DEFAULT_PARTITION_TAG!r}). Use the "
                          f"collection name (e.g. 'trading') to ingest into the "
                          f"main partition rather than a side partition."))
    ap.add_argument("--mode", default="deep", choices=("fast", "deep"),
                    help="NotebookLM search mode (default: deep)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Seconds to wait for research to finish (default: 1800)")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete the NotebookLM notebook on success "
                         "(useful for inspection). Default: delete on success.")
    ap.add_argument("--result-json", type=Path, default=None,
                    help="If set, write structured result JSON to this path "
                         "on completion (success or failure). Used by the "
                         "FastAPI /discover endpoints to detect run completion.")
    ap.add_argument("--bulk-deadline", type=float, default=None,
                    help="Wall-clock cap (seconds) on awaiting per-query "
                         "completion across the batch. Stuck queries past "
                         "this cap are marked abandoned and fetch+parse "
                         "proceeds for the rest. Default: wait indefinitely. (#36)")
    args = ap.parse_args()

    result = run_discovery_bulk(
        queries=args.query,
        collection=args.collection,
        partition=args.partition,
        mode=args.mode,
        timeout=args.timeout,
        keep=args.keep,
        bulk_deadline_s=args.bulk_deadline,
    )

    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write — server reads result.json as the completion signal.
        tmp = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.result_json)

    return 0 if result["status"] in ("ok", "no_new_sources") else 1


if __name__ == "__main__":
    raise SystemExit(main())
