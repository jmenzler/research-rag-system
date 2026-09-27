# long-ok-file
"""Unit tests for src/pipeline/orchestrate.py — pure helpers used by `ragctl run`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.pipeline import orchestrate


def _ns(**kw: object) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ---- select_doc_dirs -------------------------------------------------------


def test_select_doc_dirs_walks_collection_root(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    (sources / "trading" / "alpha").mkdir(parents=True)
    (sources / "trading" / "beta").mkdir()
    (sources / "trading" / "_quarantine_x").mkdir()  # skipped
    (sources / "ecology" / "gamma").mkdir(parents=True)

    rows = orchestrate.select_doc_dirs("trading", sources)
    slugs = [Path(r["doc_dir"]).name for r in rows]
    assert slugs == ["alpha", "beta"]
    assert all(r["collection"] == "trading" for r in rows)


def test_select_doc_dirs_missing_collection_returns_empty(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    assert orchestrate.select_doc_dirs("trading", sources) == []


def test_select_doc_dirs_unknown_collection_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown collection"):
        orchestrate.select_doc_dirs("not-a-collection", tmp_path)


# ---- select_stages ---------------------------------------------------------


def test_select_stages_default_is_all() -> None:
    got = orchestrate.select_stages(_ns(stages=None, skip_stages=None))
    assert got == orchestrate.ALL_STAGES


def test_select_stages_explicit_subset_preserves_canonical_order() -> None:
    got = orchestrate.select_stages(_ns(stages=["ingest", "fetch", "audit"], skip_stages=None))
    assert got == ["fetch", "audit", "ingest"]


def test_select_stages_all_keyword_expands() -> None:
    got = orchestrate.select_stages(_ns(stages=["all"], skip_stages=None))
    assert got == orchestrate.ALL_STAGES


def test_select_stages_skip_stages_inverse() -> None:
    got = orchestrate.select_stages(_ns(stages=None, skip_stages=["ingest", "audit"]))
    assert "ingest" not in got and "audit" not in got
    assert "fetch" in got and "parse_pipeline" in got


def test_select_stages_mutex_raises() -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        orchestrate.select_stages(_ns(stages=["fetch"], skip_stages=["ingest"]))


def test_select_stages_unknown_stage_raises() -> None:
    with pytest.raises(SystemExit, match="unknown stage"):
        orchestrate.select_stages(_ns(stages=["bogus"], skip_stages=None))


# ---- stage command builders ------------------------------------------------


def test_build_fetch_command_includes_uuid_and_collection() -> None:
    cmd = orchestrate.build_fetch_command("uuid-x", "trading", no_cffi=False)
    assert "src.fetch.cli" in " ".join(cmd)
    assert "--notebook-id" in cmd and "uuid-x" in cmd
    assert "--collection" in cmd and "trading" in cmd
    assert "--no-cffi" not in cmd


def test_build_fetch_command_no_cffi() -> None:
    cmd = orchestrate.build_fetch_command("uuid-x", "trading", no_cffi=True)
    assert "--no-cffi" in cmd


def test_build_parse_command_uses_collection() -> None:
    cmd = orchestrate.build_parse_command("trading", pass_mode="pipeline-only")
    assert "src.pdf_parsers.cli" in " ".join(cmd)
    assert "--collection" in cmd and "trading" in cmd
    assert "--pass" in cmd and "pipeline-only" in cmd


def test_build_ingest_command_targets_collection_glob() -> None:
    cmd = orchestrate.build_ingest_command("trading")
    assert "--collection" in cmd and "trading" in cmd
    assert "sources/trading/*/content_list.json" in cmd


def test_build_contextualize_command_targets_collection_glob() -> None:
    cmd = orchestrate.build_contextualize_command("trading")
    assert "src.ingest.contextualize_corpus" in " ".join(cmd)
    assert "sources/trading/*/content_list.json" in cmd
