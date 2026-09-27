# long-ok-file
"""Ingest pipeline: parse → chunk → embed → store.

CLI usage:
    python -m src.ingest --notebook example_topic --collection notes \
        --paths "sources/example_topic/*"

Supported modalities by extension:
    .pdf                   — MinerU parse → hierarchical chunk
    .png .jpg .jpeg .webp  — Gemini describe → single child+parent (modality="image")
    .mp3 .wav .m4a .flac   — Gemini transcribe → hierarchical chunk (modality="audio")
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import tiktoken
from google import genai
from google.genai import types

from src import config
from src.chunking import ProtoChunk, chunk_sidecar
from src.milvus_client import ensure_collection, ensure_partition
from src.models import ChildChunk, ParentChunk, chunk_id
from src.pdf_parsers import parse_pdf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".html", ".md", ".txt"})
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp"}
)
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".flac"}
)

_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}

# MinerU parser configuration lives in src.pdf_parsers (mineru.py / mineru_daemon.py).


from src.ingest.chunker import (  # noqa: E402
    _build_hierarchical_chunks,
    _get_encoder,
    _load_source_meta,
    _proto_chunks_to_models,
)
from src.ingest.embed import (  # noqa: E402
    _embed_image,
    _embed_text_batch,
)
from src.ingest.sources import (  # noqa: E402
    canonical_id,
    has_been_ingested_by_hash,
    mark_ingested_with_hash,
)
from src.ingest.storage import (  # noqa: E402
    _file_hash,
    _insert_children,
    _open_parents_db,
    _store_parent,
    _truncate_utf8,
    existing_child_ids,
)

# ---------------------------------------------------------------------------
# Document parsing dispatch
# ---------------------------------------------------------------------------


def _parse_text_document(path: Path) -> str:
    """Parse a text document via MinerU (pipeline + VLM fallback)."""
    return parse_pdf(path)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _build_gemini_client() -> genai.Client:
    """Return a configured Gemini client (validates API key first)."""
    config.validate_api_key()
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _describe_image(client: genai.Client, path: Path) -> str:
    """Return a textual caption/description of an image via Gemini 2.5 Flash."""
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    image_bytes = path.read_bytes()
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    prompt = types.Part.from_text(
        text=(
            "Describe this image in detail. Include all text, figures, tables,"
            " and visual elements you can see. Be thorough — this description"
            " will be used to index the image for retrieval."
        )
    )
    response = client.models.generate_content(
        model=config.GEN_MODEL,
        contents=cast(types.ContentListUnionDict, [part, prompt]),
    )
    text = response.text
    if not text:
        raise RuntimeError(f"Gemini returned empty description for image: {path}")
    return text


def _transcribe_audio(client: genai.Client, path: Path) -> str:
    """Return a transcript of an audio file via Gemini 2.5 Flash audio understanding."""
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "audio/mpeg")
    audio_bytes = path.read_bytes()
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    prompt = types.Part.from_text(
        text=(
            "Transcribe this audio file completely and accurately."
            " Include all spoken words, speaker changes if identifiable,"
            " and any important non-speech sounds. Output only the transcript."
        )
    )
    response = client.models.generate_content(
        model=config.GEN_MODEL,
        contents=cast(types.ContentListUnionDict, [part, prompt]),
    )
    text = response.text
    if not text:
        raise RuntimeError(f"Gemini returned empty transcript for audio: {path}")
    return text



# ---------------------------------------------------------------------------
# Per-file ingest handlers
# ---------------------------------------------------------------------------


def _filter_unembedded(
    collection: str,
    children: list[ChildChunk],
    *,
    force: bool,
) -> tuple[list[ChildChunk], int]:
    """Split children into (to-embed, skip-count) by Milvus presence.

    ``force`` re-embeds everything (the upgrade-path escape hatch). Otherwise
    children whose deterministic id already lives in ``collection`` are dropped
    so a re-run of an unchanged source costs 0 embed tokens.
    """
    if force or not children:
        return children, 0
    existing = existing_child_ids(collection, [c.id for c in children])
    new = [c for c in children if c.id not in existing]
    return new, len(children) - len(new)


def _ingest_text_file(
    path: Path,
    notebook: str,
    collection: str,
    gemini_client: genai.Client,
    db_conn: sqlite3.Connection,
    enc: tiktoken.Encoding,
    *,
    modality_override: str | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Ingest a text-based document (pdf, docx, html, md, txt).

    `modality_override`, when set, replaces the auto-detected document-level
    modality and is applied to BOTH parent and child chunks (structural and
    hierarchical paths). Used by the Phase 2.7 deep-research path to tag
    synthesized briefs as ``synthesized_brief``.

    Returns:
        (n_parents, n_children)
    """
    ext = path.suffix.lower()
    # Determine modality label
    if modality_override is not None:
        modality = modality_override
    elif ext == ".pdf":
        modality = "pdf"
    else:
        modality = "text"

    sidecar = path.parent / "content_list.json"
    use_structural = sidecar.is_file()
    proto_chunks: list[ProtoChunk] = []

    if use_structural:
        meta = _load_source_meta(sidecar.parent)
        try:
            _title, proto_chunks = chunk_sidecar(sidecar, enc, meta=meta)
        except Exception as exc:
            logger.warning(
                "structural chunker failed for %s: %s — falling back",
                path,
                exc,
            )
            proto_chunks = []

    if proto_chunks:
        parents, children = _proto_chunks_to_models(
            proto_chunks=proto_chunks,
            source_file=str(path),
            notebook=notebook,
            modality=modality,
            ctx_sidecar=str(sidecar),
        )
        print(
            f"  [structural] {len(parents)} parents, {len(children)} children "
            f"from {len(proto_chunks)} proto-chunks"
        )
    elif ext == ".pdf":
        print(f"  Parsing {path.name} via MinerU…")
        text = _parse_text_document(path)
        if not text.strip():
            print(f"  WARNING: MinerU returned empty text for {path.name} — skipping.")
            return 0, 0

        parents, children = _build_hierarchical_chunks(
            text=text,
            source_file=str(path),
            notebook=notebook,
            modality=modality,
            page_number=0,
            enc=enc,
        )
    else:
        # Non-PDF input (.html/.md/.txt) requires the structural sidecar — there
        # is no MinerU fallback for these. The pre-Docling-retirement code path
        # could parse them, but Docling has been removed. Surface this clearly
        # rather than silently calling MinerU on text and producing garbage.
        print(
            f"  WARNING: {path.name} has no content_list.json sidecar and is not "
            f"a PDF — skipping (run MinerU on the upstream PDF first)."
        )
        return 0, 0

    if modality_override is not None:
        # Structural chunker assigns per-chunk modalities (text/table/equation)
        # from the proto. The override forces a doc-level label across ALL
        # children + parents — required so synthesized briefs are uniformly
        # filterable on Milvus's `modality` field.
        from dataclasses import replace as _dc_replace  # noqa: PLC0415
        parents = [_dc_replace(p, modality=modality_override) for p in parents]
        children = [_dc_replace(c, modality=modality_override) for c in children]

    # Persist parents to SQLite (idempotent INSERT OR REPLACE — always cheap).
    for parent in parents:
        _store_parent(db_conn, parent)
    db_conn.commit()

    to_embed, n_skipped = _filter_unembedded(collection, children, force=force)
    if not to_embed:
        print(f"  {n_skipped} children already ingested — skipped (0 tokens)")
        return len(parents), len(children)

    print(
        f"  {len(parents)} parents, {len(to_embed)} children — embedding"
        f" ({n_skipped} already ingested)…"
    )
    t0 = time.time()
    vectors = _embed_text_batch(gemini_client, [c.text for c in to_embed])
    embed_sec = time.time() - t0
    print(f"  Embedded {len(vectors)} children in {embed_sec:.1f}s")

    t1 = time.time()
    _insert_children(collection, to_embed, vectors)
    insert_sec = time.time() - t1
    print(f"  Upserted {len(to_embed)} rows into Milvus in {insert_sec:.1f}s")

    return len(parents), len(children)


def _ingest_image_file(
    path: Path,
    notebook: str,
    collection: str,
    gemini_client: genai.Client,
    db_conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Ingest an image file.

    Produces a single parent + single child, both with modality="image".
    Dense embedding is computed natively (image bytes → 768d).
    Sparse (BM25) embedding indexes the caption text.

    Returns:
        (n_parents, n_children)  — always (1, 1)
    """
    print(f"  Describing image {path.name} via Gemini…")
    caption = _describe_image(gemini_client, path)

    p_id = chunk_id(str(path), 0, 0, "parent")
    c_id = chunk_id(str(path), 0, 0, "child")

    parent = ParentChunk(
        id=p_id,
        text=caption,
        source_file=str(path),
        notebook=notebook,
        modality="image",
        page_number=0,
        image_path=str(path),
    )
    child = ChildChunk(
        id=c_id,
        parent_id=p_id,
        text=_truncate_utf8(caption, 4096),
        source_file=str(path),
        notebook=notebook,
        modality="image",
        page_number=0,
    )

    _store_parent(db_conn, parent)
    db_conn.commit()

    to_embed, n_skipped = _filter_unembedded(collection, [child], force=force)
    if not to_embed:
        print("  image already ingested — skipped (0 tokens)")
        return 1, 1

    print("  Embedding image natively…")
    t0 = time.time()
    vector = _embed_image(gemini_client, path)
    embed_sec = time.time() - t0
    print(f"  Embedded image in {embed_sec:.1f}s")

    t1 = time.time()
    _insert_children(collection, [child], [vector])
    insert_sec = time.time() - t1
    print(f"  Upserted 1 image row into Milvus in {insert_sec:.1f}s")

    return 1, 1


def _ingest_audio_file(
    path: Path,
    notebook: str,
    collection: str,
    gemini_client: genai.Client,
    db_conn: sqlite3.Connection,
    enc: tiktoken.Encoding,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Ingest an audio file via Gemini transcription → hierarchical chunk.

    Returns:
        (n_parents, n_children)
    """
    print(f"  Transcribing audio {path.name} via Gemini…")
    transcript = _transcribe_audio(gemini_client, path)
    if not transcript.strip():
        raise RuntimeError(f"Gemini returned empty transcript for: {path}")

    parents, children = _build_hierarchical_chunks(
        text=transcript,
        source_file=str(path),
        notebook=notebook,
        modality="audio",
        page_number=0,
        enc=enc,
    )

    for parent in parents:
        _store_parent(db_conn, parent)
    db_conn.commit()

    to_embed, n_skipped = _filter_unembedded(collection, children, force=force)
    if not to_embed:
        print(f"  {n_skipped} children already ingested — skipped (0 tokens)")
        return len(parents), len(children)

    print(
        f"  {len(parents)} parents, {len(to_embed)} children — embedding"
        f" ({n_skipped} already ingested)…"
    )
    t0 = time.time()
    vectors = _embed_text_batch(gemini_client, [c.text for c in to_embed])
    embed_sec = time.time() - t0
    print(f"  Embedded {len(vectors)} children in {embed_sec:.1f}s")

    t1 = time.time()
    _insert_children(collection, to_embed, vectors)
    insert_sec = time.time() - t1
    print(f"  Upserted {len(to_embed)} rows into Milvus in {insert_sec:.1f}s")

    return len(parents), len(children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_doc_representative(path: Path) -> Path:
    """If path is `content_list.json`, swap to the doc dir's representative
    file (source.pdf > pdf.txt > nlm.txt > web.txt). The structural chunker
    keys off `path.parent / content_list.json` either way; this just gives
    `source_file` a meaningful basename.
    """
    if path.name != "content_list.json":
        return path
    doc_dir = path.parent
    for fname in ("source.pdf", "pdf.txt", "nlm.txt", "web.txt"):
        cand = doc_dir / fname
        if cand.is_file():
            return cand
    # Fallback: keep content_list.json as the path; ext-dispatch will treat it
    # as text since we'll add it to _TEXT_EXTENSIONS via filename check.
    return path


def ingest_file(
    path: Path,
    notebook: str,
    collection: str,
    gemini_client: genai.Client,
    db_conn: sqlite3.Connection,
    enc: tiktoken.Encoding,
    *,
    modality_override: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Ingest one file into Milvus + parents.sqlite; returns a summary dict.
    Idempotent on the source.pdf hash unless ``force`` — the upgrade path
    re-parses an unchanged PDF, so the caller must delete stale chunks first
    (``storage.delete_source_chunks``). A ``content_list.json`` path is accepted
    and swapped to the doc's representative file (source.pdf preferred)."""
    path = _resolve_doc_representative(path)
    ext = path.suffix.lower()

    if ext not in (_TEXT_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS):
        print(f"  Skipping unsupported extension: {path.name}")
        return {
            "file": str(path),
            "skipped": True,
            "reason": f"unsupported extension {ext!r}",
            "n_parents": 0,
            "n_children": 0,
        }

    fhash = _file_hash(path)
    if not force and has_been_ingested_by_hash(db_conn, fhash, notebook):
        print(f"  Skipping {path.name} — already ingested (hash match).")
        return {
            "file": str(path),
            "skipped": True,
            "reason": "already_ingested",
            "n_parents": 0,
            "n_children": 0,
        }

    t_start = time.time()

    if ext in _TEXT_EXTENSIONS:
        n_parents, n_children = _ingest_text_file(
            path, notebook, collection, gemini_client, db_conn, enc,
            modality_override=modality_override, force=force,
        )
    elif ext in _IMAGE_EXTENSIONS:
        n_parents, n_children = _ingest_image_file(
            path, notebook, collection, gemini_client, db_conn, force=force
        )
    else:  # audio
        n_parents, n_children = _ingest_audio_file(
            path, notebook, collection, gemini_client, db_conn, enc, force=force
        )

    cid_meta = _load_source_meta(path.parent) or {"title": path.stem, "url": str(path)}
    cid, _ctype = canonical_id(cid_meta)
    mark_ingested_with_hash(db_conn, cid, fhash, notebook)

    elapsed = time.time() - t_start
    return {
        "file": str(path),
        "skipped": False,
        "n_parents": n_parents,
        "n_children": n_children,
        "total_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def _append_log(
    notebook: str,
    source_file: str,
    n_parents: int,
    n_children: int,
    embed_seconds: float,
    insert_seconds: float,
) -> None:
    """Append a single-line JSON summary to logs/ingest.jsonl."""
    log_path = Path("logs/ingest.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "notebook": notebook,
        "file": source_file,
        "chunks_parent": n_parents,
        "chunks_child": n_children,
        "embed_seconds": round(embed_seconds, 3),
        "insert_seconds": round(insert_seconds, 3),
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest source files into the Milvus RAG system."
    )
    parser.add_argument(
        "--notebook",
        default=None,
        help=(
            "Partition key. Defaults to the collection name when omitted "
            "(post-flatten layout uses one partition per collection)."
        ),
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection name (e.g. notes, trading, ecology)",
    )
    parser.add_argument(
        "--paths",
        required=True,
        nargs="+",
        help="One or more file paths or glob patterns (e.g. sources/example_topic/*)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes. 0 (default) auto-scales: "
            "min(8, max(1, N_paths // 4)). Pass 1 to force serial execution."
        ),
    )
    return parser.parse_args()


def _resolve_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns + literal paths to a sorted list of concrete files.

    Sorted output is the precondition for deterministic round-robin sharding —
    the same input arguments must produce the same shard membership across runs.
    """
    resolved: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            resolved.extend(Path(m) for m in matches if Path(m).is_file())
        else:
            candidate = Path(pattern)
            if candidate.is_file():
                resolved.append(candidate)
            else:
                print(f"Warning: no files matched pattern: {pattern!r}")
    return sorted(resolved)


def _auto_workers(n_paths: int, requested: int) -> int:
    """Pick worker count: explicit override > auto-scale heuristic.

    Explicit ``requested`` is capped by ``n_paths`` so we don't fork empty
    workers. Auto-scale (requested=0) is ``min(8, max(1, n_paths // 4))`` —
    single-doc dev calls stay serial, bulk runs scale to a cap of 8.
    """
    if requested > 0:
        return min(requested, n_paths)
    return max(1, min(8, n_paths // 4))


def _run_shard(
    shard_paths: list[str],
    collection: str,
    notebook: str,
    worker_id: int = 0,
) -> dict[str, int]:
    """Ingest one shard of files. Owns its own clients / sqlite handle.

    Used both as the in-process serial path (workers=1) and as the
    ProcessPoolExecutor worker entry (workers>1). Per-doc failures are
    absorbed at the loop so one bad doc doesn't kill the rest of the
    shard.

    Returns aggregate counts so the parent can sum across workers.
    """
    config.validate_api_key()
    ensure_collection(collection)
    ensure_partition(collection, notebook)

    gemini_client = _build_gemini_client()
    enc = _get_encoder()
    db_conn = _open_parents_db()

    total_parents = 0
    total_children = 0
    n_ok = 0
    n_fail = 0
    n_skip = 0

    prefix = f"[w{worker_id}] " if worker_id else ""

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    with StageLogger(notebook, "ingest") as sl:
        for path_str in shard_paths:
            path = Path(path_str)
            print(f"\n{prefix}[{path.name}]", flush=True)
            t0 = time.time()
            try:
                summary = ingest_file(
                    path=path,
                    notebook=notebook,
                    collection=collection,
                    gemini_client=gemini_client,
                    db_conn=db_conn,
                    enc=enc,
                )
            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - t0
                print(f"{prefix}  → FAIL {type(e).__name__}: {e}", flush=True)
                sl.item_fail(path.name, elapsed_s=elapsed,
                             reason=f"{type(e).__name__}: {e}")
                n_fail += 1
                continue

            if summary.get("skipped"):
                reason = str(summary.get("reason"))
                print(f"{prefix}  → skipped ({reason})", flush=True)
                sl.item_skip(path.name, reason=reason)
                n_skip += 1
                continue

            n_p = cast(int, summary["n_parents"])
            n_c = cast(int, summary["n_children"])
            total_parents += n_p
            total_children += n_c
            elapsed = time.time() - t0
            print(
                f"{prefix}  → done: {n_p} parents, {n_c} children in {elapsed:.1f}s",
                flush=True,
            )
            sl.item_ok(path.name, elapsed_s=elapsed,
                       n_parents=n_p, n_children=n_c)
            n_ok += 1

            _append_log(
                notebook=notebook,
                source_file=str(path),
                n_parents=n_p,
                n_children=n_c,
                embed_seconds=0.0,
                insert_seconds=0.0,
            )

    db_conn.close()
    return {
        "n_parents": total_parents,
        "n_children": total_children,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_skip": n_skip,
    }


def main() -> None:
    """CLI entry point: parse args, resolve paths, dispatch shards."""
    args = _parse_args()
    notebook = args.notebook or args.collection

    resolved = _resolve_paths(args.paths)
    if not resolved:
        raise RuntimeError("No files found to ingest. Check --paths argument.")

    print(f"Found {len(resolved)} file(s) to process.")
    print(f"Collection '{args.collection}'  notebook (partition key): '{notebook}'")

    n_workers = _auto_workers(len(resolved), args.workers)

    if n_workers == 1:
        print("Running serial (workers=1)")
        result = _run_shard(
            [str(p) for p in resolved], args.collection, notebook, worker_id=0,
        )
        print(
            f"\nIngest complete: {result['n_parents']} parents, "
            f"{result['n_children']} children "
            f"(ok={result['n_ok']} fail={result['n_fail']} skip={result['n_skip']})",
        )
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

    shards: list[list[str]] = [[] for _ in range(n_workers)]
    for i, p in enumerate(resolved):
        shards[i % n_workers].append(str(p))

    print(
        f"Running parallel: {n_workers} workers × ~{len(resolved) // n_workers} docs each",
    )
    for k, s in enumerate(shards):
        print(f"  shard {k}: {len(s)} docs")

    totals = {"n_parents": 0, "n_children": 0, "n_ok": 0, "n_fail": 0, "n_skip": 0}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _run_shard, shard, args.collection, notebook, k + 1,
            ): k
            for k, shard in enumerate(shards) if shard
        }
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"\nworker {k + 1} CRASHED: {type(e).__name__}: {e}", flush=True)
                continue
            print(
                f"\nworker {k + 1} done: {result['n_ok']} ok / "
                f"{result['n_fail']} fail / {result['n_skip']} skip "
                f"({result['n_parents']} parents, {result['n_children']} children)",
                flush=True,
            )
            for key in totals:
                totals[key] += result[key]

    print(
        f"\nIngest complete (parallel, {n_workers}w, {time.time() - t_start:.0f}s): "
        f"{totals['n_parents']} parents, {totals['n_children']} children "
        f"(ok={totals['n_ok']} fail={totals['n_fail']} skip={totals['n_skip']})",
    )


if __name__ == "__main__":
    main()
