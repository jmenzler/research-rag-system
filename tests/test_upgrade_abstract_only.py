"""Worker that re-parses abstract-only monitor papers to full PDF.
Heavy deps (MinerU, Milvus, Gemini, run-lock, daemon) are mocked; the tests
pin the scan filter, the lock+daemon bracketing, and the per-doc upgrade
(re-parse → delete stub chunks → re-ingest → flip meta to full_pdf).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.upgrade_abstract_only as uao
from src.pipeline.runner import RunLockHeldError


def _make_doc(root: Path, arxiv_id: str, quality: str, *, with_pdf: bool = True) -> Path:
    doc = root / "sources" / "trading" / arxiv_id
    doc.mkdir(parents=True)
    (doc / "meta.json").write_text(
        json.dumps({"title": arxiv_id, "extraction_quality": quality}), encoding="utf-8"
    )
    if with_pdf:
        (doc / "source.pdf").write_bytes(b"%PDF-1.4")
    return doc


def test_find_abstract_only_docs_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    keep = _make_doc(tmp_path, "2401.1", "abstract_only")
    _make_doc(tmp_path, "2401.2", "full_pdf")  # already full — skip
    _make_doc(tmp_path, "2401.3", "abstract_only", with_pdf=False)  # no pdf — skip

    found = uao._find_abstract_only_docs("trading")

    assert found == [keep]


def test_run_no_docs_skips_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    with patch.object(uao, "_acquire_run_lock") as acquire:
        result = uao._run("trading")
    assert result == {"status": "ok", "scanned": 0, "upgraded": 0, "failed": 0}
    acquire.assert_not_called()


def test_run_defers_when_run_lock_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    _make_doc(tmp_path, "2401.1", "abstract_only")
    with (
        patch.object(uao, "_acquire_run_lock", side_effect=RunLockHeldError("busy")),
        patch("scripts.upgrade_abstract_only.mineru_daemon.start_pipeline_daemon") as start,
    ):
        result = uao._run("trading")
    assert result["status"] == "ok"
    assert result["upgraded"] == 0
    assert result["note"] == "run_lock_held"
    start.assert_not_called()


def test_run_upgrades_each_doc_with_daemon_bracket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    _make_doc(tmp_path, "2401.1", "abstract_only")
    _make_doc(tmp_path, "2401.2", "abstract_only")

    with (
        patch.object(uao, "_acquire_run_lock", return_value=7) as acquire,
        patch.object(uao, "_release_run_lock") as release,
        patch("scripts.upgrade_abstract_only.config.validate_api_key"),
        patch.object(uao, "ensure_collection"),
        patch.object(uao, "ensure_partition"),
        patch("scripts.upgrade_abstract_only.genai.Client"),
        patch.object(uao, "_get_encoder"),
        patch.object(uao, "_open_parents_db"),
        patch("scripts.upgrade_abstract_only.mineru_daemon.start_pipeline_daemon") as start,
        patch("scripts.upgrade_abstract_only.mineru_daemon.stop_pipeline_daemon") as stop,
        patch.object(uao, "_upgrade_doc") as upgrade,
    ):
        result = uao._run("trading")

    assert result == {"status": "ok", "scanned": 2, "upgraded": 2, "failed": 0}
    acquire.assert_called_once()
    release.assert_called_once_with(7)
    start.assert_called_once()
    stop.assert_called_once()
    assert upgrade.call_count == 2


def test_run_counts_per_doc_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    _make_doc(tmp_path, "2401.1", "abstract_only")
    _make_doc(tmp_path, "2401.2", "abstract_only")

    def _fail_second(doc_dir: Path, *a: object, **k: object) -> None:
        if doc_dir.name == "2401.2":
            raise RuntimeError("parse boom")

    with (
        patch.object(uao, "_acquire_run_lock", return_value=1),
        patch.object(uao, "_release_run_lock"),
        patch("scripts.upgrade_abstract_only.config.validate_api_key"),
        patch.object(uao, "ensure_collection"),
        patch.object(uao, "ensure_partition"),
        patch("scripts.upgrade_abstract_only.genai.Client"),
        patch.object(uao, "_get_encoder"),
        patch.object(uao, "_open_parents_db"),
        patch("scripts.upgrade_abstract_only.mineru_daemon.start_pipeline_daemon"),
        patch("scripts.upgrade_abstract_only.mineru_daemon.stop_pipeline_daemon") as stop,
        patch.object(uao, "_upgrade_doc", side_effect=_fail_second),
    ):
        result = uao._run("trading")

    assert result == {"status": "ok", "scanned": 2, "upgraded": 1, "failed": 1}
    stop.assert_called_once()  # daemon still torn down despite a failure


def test_upgrade_doc_reparses_and_flips_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uao, "_PROJECT_ROOT", tmp_path)
    doc = _make_doc(tmp_path, "2401.1", "abstract_only")

    def _fake_parse(pdf_path: Path) -> str:
        (pdf_path.parent / "content_list.json").write_text("[]", encoding="utf-8")
        return "ok"

    with (
        patch.object(uao, "parse_pdf", side_effect=_fake_parse) as parse,
        patch.object(uao, "delete_source_chunks") as delete,
        patch.object(uao, "ingest_file") as ingest,
    ):
        uao._upgrade_doc(doc, "trading", MagicMock(), MagicMock(), MagicMock())

    parse.assert_called_once()
    delete.assert_called_once()
    assert delete.call_args.kwargs["notebook"] == "arxiv_monitor"
    assert ingest.call_args.kwargs["force"] is True
    meta = json.loads((doc / "meta.json").read_text())
    assert meta["extraction_quality"] == "full_pdf"
