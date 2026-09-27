"""Quarantine deterministic-crasher PDFs (gh #17).

The MinerU pipeline / paddleocr / torch native heap-corruption SIGABRTs
are not getting fixed upstream. Local handling: detect the suspect PDF
correlated with two SIGABRTs and move it under
``sources/_quarantine/<collection>/<slug>/`` with a forensics sidecar.

Subsequent ragctl runs skip ``sources/_quarantine/`` because the
underscore-prefixed top-level dir is already filtered by the existing
walker (see ``orchestrate.select_doc_dirs`` and the source-discovery
guard in ``runner._discover_existing_pdfs``).

Fallback parsers (parse_vlm subprocess, pdfplumber) are deferred to a
follow-up — this module only owns the move + sidecar + lookup surface.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

_QUARANTINE_ROOT_NAME = "_quarantine"
_SIDECAR_NAME = "_quarantine.json"


@dataclass(frozen=True)
class QuarantineSidecar:
    """Forensics dropped next to a quarantined source.

    ``daemon_log_refs`` lists the pipeline-daemon log paths whose
    SIGABRTs implicated this PDF — operator can grep them for the
    actual glibc / paddleocr stack trace.
    """
    reason: str
    deaths_observed: int
    predecessor_pdf: str
    daemon_log_refs: list[str]
    url: str
    pdf_pages: int = 0
    pdf_size_bytes: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self, *, quarantined_at: str) -> dict[str, object]:
        d = asdict(self)
        d["quarantined_at"] = quarantined_at
        return d


def _quarantine_dir(sources_root: Path, collection: str, slug: str) -> Path:
    return sources_root / _QUARANTINE_ROOT_NAME / collection / slug


def _active_dir(sources_root: Path, collection: str, slug: str) -> Path:
    return sources_root / collection / slug


def quarantine_source(
    *,
    sources_root: Path,
    collection: str,
    slug: str,
    sidecar: QuarantineSidecar,
) -> Path:
    """Move the active source dir under ``_quarantine/`` and write the sidecar.

    Idempotent: re-quarantining an already-quarantined slug updates the
    sidecar in place. If the active dir exists alongside an existing
    quarantine entry (e.g. a re-fetch sneaked in before quarantine was
    applied), the active copy replaces the stale one — the freshest
    files are the ones the operator will want to inspect.

    Returns the new path under ``_quarantine/``.
    """
    src = _active_dir(sources_root, collection, slug)
    dst = _quarantine_dir(sources_root, collection, slug)

    # If the destination already exists (idempotent re-quarantine), nuke
    # it before move so shutil.move semantics stay deterministic across
    # platforms (POSIX overwrites, Windows raises).
    if dst.exists():
        shutil.rmtree(dst)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    else:
        # Active dir is missing — possibly a follow-up to an already-
        # quarantined slug. Recreate the destination so the sidecar
        # always lands somewhere readable.
        dst.mkdir(parents=True, exist_ok=True)

    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    (dst / _SIDECAR_NAME).write_text(
        json.dumps(sidecar.to_dict(quarantined_at=now), indent=2, ensure_ascii=False),
    )
    return dst


def is_quarantined(sources_root: Path, collection: str, slug: str) -> bool:
    return _quarantine_dir(sources_root, collection, slug).is_dir()


def list_quarantined(sources_root: Path) -> list[dict[str, object]]:
    """Return one row per quarantined slug across all collections.

    Each row carries ``collection``, ``slug``, ``path``, plus the
    sidecar's contents (or ``{}`` if missing/corrupt). Sorted by
    ``quarantined_at`` so the most recent quarantines come last.
    """
    qroot = sources_root / _QUARANTINE_ROOT_NAME
    if not qroot.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for coll_dir in sorted(qroot.iterdir()):
        if not coll_dir.is_dir():
            continue
        for slug_dir in sorted(coll_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            sc_path = slug_dir / _SIDECAR_NAME
            sidecar_data: dict[str, object] = {}
            if sc_path.exists():
                try:
                    sidecar_data = json.loads(sc_path.read_text())
                except (json.JSONDecodeError, OSError):
                    sidecar_data = {}
            row: dict[str, object] = {
                "collection": coll_dir.name,
                "slug": slug_dir.name,
                "path": str(slug_dir),
            }
            row.update(sidecar_data)
            rows.append(row)
    rows.sort(key=lambda r: str(r.get("quarantined_at", "")))
    return rows


def unquarantine_source(
    *,
    sources_root: Path,
    collection: str,
    slug: str,
) -> Path:
    """Move the slug back to active sources/. Drops the sidecar.

    Raises ``FileNotFoundError`` if the slug is not under quarantine.
    Returns the new active path.
    """
    src = _quarantine_dir(sources_root, collection, slug)
    if not src.is_dir():
        raise FileNotFoundError(
            f"not under quarantine: {collection}/{slug}",
        )
    dst = _active_dir(sources_root, collection, slug)

    # Remove the sidecar before moving so it doesn't survive in active.
    sidecar = src / _SIDECAR_NAME
    if sidecar.exists():
        sidecar.unlink()

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


__all__ = [
    "QuarantineSidecar",
    "is_quarantined",
    "list_quarantined",
    "quarantine_source",
    "unquarantine_source",
]
