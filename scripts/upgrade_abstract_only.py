"""Re-parse abstract-only monitor papers to full PDF and re-ingest.
Spawned by the DiscoverQueue worker (kind="upgrade"): acquires the ragctl
run-lock, starts the MinerU daemon once, re-parses each on-disk source.pdf,
deletes stale stub chunks, and re-ingests. Run-lock + single daemon owner
mean this never races a ragctl batch run."""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from google import genai

from src import config
from src.ingest.chunker import _get_encoder
from src.ingest.ingest import ingest_file
from src.ingest.storage import _open_parents_db, delete_source_chunks
from src.milvus_client import ensure_collection, ensure_partition
from src.pdf_parsers import mineru_daemon
from src.pdf_parsers.mineru import parse_pdf
from src.pipeline.runner import RunLockHeldError, _acquire_run_lock, _release_run_lock

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_NOTEBOOK = "arxiv_monitor"


def _find_abstract_only_docs(collection: str) -> list[Path]:
    """Doc dirs under sources/<collection>/ that are abstract-only with a PDF."""
    sources_dir = _PROJECT_ROOT / "sources" / collection
    if not sources_dir.is_dir():
        return []
    out: list[Path] = []
    for doc_dir in sorted(p for p in sources_dir.iterdir() if p.is_dir()):
        meta_path = doc_dir / "meta.json"
        if not meta_path.is_file() or not (doc_dir / "source.pdf").is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("extraction_quality") == "abstract_only":
            out.append(doc_dir)
    return out


def _upgrade_doc(
    doc_dir: Path,
    collection: str,
    gemini_client: genai.Client,
    db_conn: sqlite3.Connection,
    enc: object,
) -> None:
    """Re-parse one doc to full PDF and re-ingest, replacing the stub chunks."""
    pdf_path = doc_dir / "source.pdf"
    parse_pdf(pdf_path)  # overwrites content_list.json with structural output

    content_list = doc_dir / "content_list.json"
    if not content_list.is_file():
        raise RuntimeError(f"parse produced no content_list.json for {doc_dir.name}")

    delete_source_chunks(collection, str(pdf_path), db_conn, notebook=_NOTEBOOK)
    ingest_file(
        path=content_list,
        notebook=_NOTEBOOK,
        collection=collection,
        gemini_client=gemini_client,
        db_conn=db_conn,
        enc=enc,  # type: ignore[arg-type]
        force=True,
    )

    meta_path = doc_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["extraction_quality"] = "full_pdf"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(collection: str) -> dict[str, object]:
    docs = _find_abstract_only_docs(collection)
    if not docs:
        return {"status": "ok", "scanned": 0, "upgraded": 0, "failed": 0}

    try:
        lock_fd = _acquire_run_lock()
    except RunLockHeldError:
        logger.info("run lock held (ragctl active) — deferring upgrade to next poll")
        return {"status": "ok", "scanned": len(docs), "upgraded": 0, "failed": 0,
                "note": "run_lock_held"}

    upgraded = failed = 0
    try:
        config.validate_api_key()
        ensure_collection(collection)
        ensure_partition(collection, _NOTEBOOK)
        gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        enc = _get_encoder()
        db_conn = _open_parents_db()
        mineru_daemon.start_pipeline_daemon()
        try:
            for doc_dir in docs:
                try:
                    _upgrade_doc(doc_dir, collection, gemini_client, db_conn, enc)
                    upgraded += 1
                    logger.info("upgraded %s to full_pdf", doc_dir.name)
                except Exception:  # noqa: BLE001
                    failed += 1
                    logger.error("upgrade failed for %s", doc_dir.name, exc_info=True)
        finally:
            db_conn.close()
            mineru_daemon.stop_pipeline_daemon()
    finally:
        _release_run_lock(lock_fd)

    return {"status": "ok", "scanned": len(docs), "upgraded": upgraded, "failed": failed}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="trading")
    ap.add_argument("--result-json", type=Path, default=None)
    args = ap.parse_args()
    collection: str = args.collection
    result_json: Path | None = args.result_json

    try:
        result = _run(collection)
    except Exception as exc:  # noqa: BLE001
        logger.error("upgrade run crashed", exc_info=True)
        result = {"status": "crashed", "note": str(exc)}

    if result_json is not None:
        result_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = result_json.with_suffix(result_json.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(result_json)

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
