"""Wiring test for the `quality` CLI command against tiny on-disk stores.

Validates the SQL lookups (paper_quality / fwci_baseline / venue_prestige /
author_authority), JSON parsing of stored list fields, and the score render path —
without the real multi-GB stores.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.citations.cli import main

ARXIV = "2208.06046"
CID = 251554626


@pytest.fixture()
def stores(tmp_path: Path) -> Path:
    base = tmp_path / "2026-05-12"
    base.mkdir()

    meta = sqlite3.connect(base / "paper_meta.sqlite")
    meta.execute(
        "CREATE TABLE paper_meta(corpusid BIGINT, arxiv_id VARCHAR, doi VARCHAR, "
        "title VARCHAR, year BIGINT, citationcount BIGINT)"
    )
    meta.execute(
        "INSERT INTO paper_meta VALUES (?,?,?,?,?,?)",
        (CID, ARXIV, None, "Automated Market Making and LVR", 2022, 90),
    )
    meta.commit()
    meta.close()

    q = sqlite3.connect(base / "paper_quality.sqlite")
    q.execute(
        "CREATE TABLE paper_quality(corpusid INTEGER PRIMARY KEY, infl_cites INTEGER, "
        "ref_count INTEGER, pub_types TEXT, venue_id TEXT, is_oa INTEGER, field TEXT, "
        "year INTEGER, pub_date TEXT, author_ids TEXT)"
    )
    q.execute(
        "INSERT INTO paper_quality VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            CID,
            40,
            45,
            json.dumps(["JournalArticle"]),
            "v1",
            1,
            "Economics",
            2022,
            "2022-08-12",
            json.dumps([11, 22]),
        ),
    )
    q.execute(
        "CREATE TABLE fwci_baseline(field TEXT, pub_tier TEXT, age_bucket TEXT, "
        "mean_infl REAL, n INTEGER, PRIMARY KEY(field, pub_tier, age_bucket))"
    )
    q.execute(
        "INSERT INTO fwci_baseline VALUES (?,?,?,?,?)", ("Economics", "reviewed", "3-5y", 10.0, 500)
    )
    q.execute("CREATE TABLE venue_prestige(venue_id TEXT PRIMARY KEY, mean_infl REAL, n INTEGER)")
    q.execute("INSERT INTO venue_prestige VALUES (?,?,?)", ("v1", 8.0, 1000))
    q.commit()
    q.close()

    a = sqlite3.connect(base / "author_authority.sqlite")
    a.execute(
        "CREATE TABLE author_authority(authorid INTEGER PRIMARY KEY, hindex INTEGER, "
        "citationcount INTEGER, papercount INTEGER, name TEXT)"
    )
    a.executemany(
        "INSERT INTO author_authority VALUES (?,?,?,?,?)",
        [(11, 40, 9000, 120, "A"), (22, 18, 1200, 60, "B")],
    )
    a.commit()
    a.close()
    return tmp_path


def test_quality_cli_end_to_end(stores: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "--base",
            str(stores),
            "--release",
            "2026-05-12",
            "--json",
            "quality",
            ARXIV,
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["corpus_id"] == CID
    # FWCI = 40 influential / 10 expected = 4.0 → impact ~0.8; peer-reviewed venue.
    assert out["impact_fwci"] == pytest.approx(4.0)
    assert out["venue_label"] == "peer-reviewed"
    assert out["max_hindex"] == 40
    assert out["n_authors"] == 2
    assert out["composite"] > 60  # strong on all dimensions
    assert out["impact_confidence"] == "ok"
