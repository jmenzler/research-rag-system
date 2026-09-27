# long-ok-file
"""Pure helpers for the ``ragctl run`` pipelined orchestrator.

Subprocess execution, ProcessPool wiring, and arg parsing live in ``bin/ragctl``.
This module only owns: collection / doc-dir selection, stage selection, and
stage→subprocess-command mapping. Keeping it side-effect-free makes the moving
parts unit-testable without touching disk or running subprocesses.

Filesystem layout: ``sources/<collection>/<slug>/<leaf>``. The collection name
is also the Milvus partition value at ingest time. There is no per-NLM-notebook
addressability after fetch — once a doc lands in ``sources/<coll>/<slug>/`` the
NLM UUID it came from is a discardable batch handle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

VALID_COLLECTIONS: tuple[str, ...] = ("trading", "ecology", "notes", "system", "poker", "security")

ALL_STAGES: list[str] = [
    "fetch", "parse_pipeline", "parse_vlm", "stage", "audit", "contextualize", "ingest",
]

PRE_BARRIER_STAGES: tuple[str, ...] = (
    "fetch", "parse_pipeline", "parse_vlm", "stage",
)
POST_BARRIER_STAGES: tuple[str, ...] = ("contextualize", "ingest")
BARRIER_STAGE: str = "audit"


def select_doc_dirs(
    collection: str, sources_root: Path,
) -> list[dict[str, str]]:
    """Walk ``sources/<collection>/*/`` and return one row per doc dir.

    Each row carries ``collection`` and ``doc_dir`` (absolute path). Skips
    administrative subdirs (those whose name starts with ``_``) so quarantine
    and dedup buckets stay out of the iteration.
    """
    if collection not in VALID_COLLECTIONS:
        raise SystemExit(
            f"unknown collection: {collection!r}. "
            f"Expected one of {VALID_COLLECTIONS}.",
        )
    coll_root = sources_root / collection
    if not coll_root.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for d in sorted(coll_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        rows.append({"collection": collection, "doc_dir": str(d)})
    return rows


def select_stages(args: argparse.Namespace) -> list[str]:
    """Resolve --stages / --skip-stages → ordered subset of ALL_STAGES.

    The "parse" pseudo-alias expands to ["parse_pipeline", "parse_vlm"] in order.
    """
    stages = getattr(args, "stages", None)
    skip = getattr(args, "skip_stages", None)

    if stages and skip:
        raise SystemExit("--stages and --skip-stages are mutually exclusive")
    if stages:
        if "all" in stages:
            return list(ALL_STAGES)
        if "parse" in stages:
            stages = [
                s for s in stages if s != "parse"
            ] + ["parse_pipeline", "parse_vlm"]
        invalid = set(stages) - set(ALL_STAGES)
        if invalid:
            raise SystemExit(f"unknown stage(s): {sorted(invalid)}")
        return [s for s in ALL_STAGES if s in stages]
    if skip:
        if "parse" in skip:
            skip = [
                s for s in skip if s != "parse"
            ] + ["parse_pipeline", "parse_vlm"]
        invalid = set(skip) - set(ALL_STAGES)
        if invalid:
            raise SystemExit(f"unknown stage(s): {sorted(invalid)}")
        return [s for s in ALL_STAGES if s not in skip]
    return list(ALL_STAGES)


def build_fetch_command(
    notebook_id: str, collection: str, *, no_cffi: bool,
) -> list[str]:
    """Stage-1 fetch for one NLM UUID into ``sources/<collection>/``."""
    cmd = ["uv", "run", "python", "-m", "src.fetch.cli",
           "--notebook-id", notebook_id, "--collection", collection]
    if no_cffi:
        cmd.append("--no-cffi")
    return cmd


def build_parse_command(collection: str, *, pass_mode: str) -> list[str]:
    """Stage-2 parse over every doc-dir in ``sources/<collection>/*/``."""
    return ["uv", "run", "python", "-m", "src.pdf_parsers.cli",
            "--collection", collection, "--pass", pass_mode]


def build_ingest_command(collection: str) -> list[str]:
    """Ingest every doc in ``sources/<collection>/*/``."""
    return [
        "uv", "run", "python", "-m", "src.ingest.ingest",
        "--collection", collection,
        "--notebook", collection,
        "--paths", f"sources/{collection}/*/content_list.json",
    ]


def build_contextualize_command(collection: str) -> list[str]:
    """Contextualize every doc under ``sources/<collection>/``.

    Keys off ``content_list.json`` (the structural chunker's input) so the
    sidecar's (parent_idx, child_idx) align with what ingest emits.
    """
    return [
        "uv", "run", "python", "-m", "src.ingest.contextualize_corpus",
        "--paths", f"sources/{collection}/*/content_list.json",
        "--slug", collection,
    ]
