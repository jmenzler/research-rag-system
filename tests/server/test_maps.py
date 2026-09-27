"""RED tests for src/server/maps_store.py — GREEN after Plan 05-02 lands.

Covers INGEST-06 (maps CRUD), INGEST-07 (coverage scorecard), D-10 (scorecard shape),
and security guards (snapshot size bound V5, SQL injection).

All tests skip at collection time until src.server.maps_store exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("src.server.maps_store")

from src.server.maps_store import (  # noqa: E402
    compute_coverage,
    create_map,
    delete_map,
    get_map,
    list_maps,
    open_maps_conn,
)

# ---------------------------------------------------------------------------
# INGEST-06: create + get round-trip
# ---------------------------------------------------------------------------


def test_create_and_get_map(tmp_chats_db: Path) -> None:
    """INGEST-06: create_map → get_map returns row with matching name + snapshot round-trips."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        snapshot = json.dumps(
            {
                "nodes": [{"corpus_id": 12345, "title": "Paper A", "in_corpus": True}],
                "edges": [],
                "view_params": {"depth": 2, "direction": "both"},
                "corpus_ids": [12345],
            }
        )
        map_id = create_map(conn, name="M1", collection="trading", snapshot=snapshot)
        assert isinstance(map_id, str) and map_id

        row = get_map(conn, map_id)
        assert row is not None
        assert row["name"] == "M1"
        loaded = json.loads(row["snapshot"])
        assert loaded["nodes"][0]["corpus_id"] == 12345
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# INGEST-06: snapshot nodes intact — corpus_id + edges + view_params + corpus_ids
# ---------------------------------------------------------------------------


def test_snapshot_preserves_resolved_nodes(tmp_chats_db: Path) -> None:
    """INGEST-06: snapshot JSON round-trips nodes[].corpus_id, edges, view_params, corpus_ids."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        snap = {
            "nodes": [{"corpus_id": 99, "title": "T", "in_corpus": False}],
            "edges": [{"from_id": 99, "to_id": 100}],
            "view_params": {"depth": 3, "direction": "references", "year_from": 2020},
            "corpus_ids": [99, 100],
        }
        map_id = create_map(conn, name="M2", collection="ecology", snapshot=json.dumps(snap))
        row = get_map(conn, map_id)
        assert row is not None
        loaded = json.loads(row["snapshot"])
        assert loaded["nodes"][0]["corpus_id"] == 99
        assert loaded["edges"][0]["from_id"] == 99
        assert loaded["view_params"]["direction"] == "references"
        assert loaded["corpus_ids"] == [99, 100]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# INGEST-06: list_maps filters by collection
# ---------------------------------------------------------------------------


def test_list_maps_by_collection(tmp_chats_db: Path) -> None:
    """INGEST-06: two maps in 'trading', one in 'ecology' → list_maps(conn,'trading') len 2."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        snap = json.dumps({"nodes": [], "edges": [], "view_params": {}, "corpus_ids": []})
        create_map(conn, name="T1", collection="trading", snapshot=snap)
        create_map(conn, name="T2", collection="trading", snapshot=snap)
        create_map(conn, name="S1", collection="ecology", snapshot=snap)

        trading_maps = list_maps(conn, "trading")
        assert len(trading_maps) == 2, f"expected 2 trading maps, got {len(trading_maps)}"

        ecology_maps = list_maps(conn, "ecology")
        assert len(ecology_maps) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# INGEST-06: delete_map
# ---------------------------------------------------------------------------


def test_delete_map(tmp_chats_db: Path) -> None:
    """INGEST-06: delete_map then get_map returns None."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        snap = json.dumps({"nodes": [], "edges": [], "view_params": {}, "corpus_ids": []})
        map_id = create_map(conn, name="Del", collection="trading", snapshot=snap)
        assert get_map(conn, map_id) is not None
        deleted = delete_map(conn, map_id)
        assert deleted is True
        assert get_map(conn, map_id) is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Security DoS: snapshot > 5MB → ValueError
# ---------------------------------------------------------------------------


def test_snapshot_size_bound(tmp_chats_db: Path) -> None:
    """Security V5/DoS: snapshot > 5MB must raise ValueError."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        big_snap = "x" * (5 * 1024 * 1024 + 1)
        with pytest.raises(ValueError):
            create_map(conn, name="Big", collection="trading", snapshot=big_snap)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Security tampering: SQL injection via name field
# ---------------------------------------------------------------------------


def test_parameterized_sql_no_injection(tmp_chats_db: Path) -> None:
    """SQL-like names remain literal data and leave the table intact."""
    conn = open_maps_conn(tmp_chats_db)
    try:
        malicious_name = "'; DROP TABLE maps;--"
        snap = json.dumps({"nodes": [], "edges": [], "view_params": {}, "corpus_ids": []})
        map_id = create_map(conn, name=malicious_name, collection="trading", snapshot=snap)
        row = get_map(conn, map_id)
        assert row is not None
        assert row["name"] == malicious_name

        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "maps" in tables, "maps table must still exist after injection attempt"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# INGEST-07 / D-10: coverage scorecard shape
# ---------------------------------------------------------------------------


def test_coverage_scorecard_shape() -> None:
    """INGEST-07/D-10: compute_coverage returns {in_corpus:N, total:M, missing_ids:[...]}."""
    snapshot_corpus_ids = [1, 2, 3, 4, 5]
    in_corpus_map = {
        1: {"in_corpus": True},
        2: {"in_corpus": False},
        3: {"in_corpus": True},
        4: {"in_corpus": False},
        5: {"in_corpus": True},
    }
    result = compute_coverage(snapshot_corpus_ids, in_corpus_map)
    assert result["in_corpus"] == 3, f"expected in_corpus=3, got {result['in_corpus']}"
    assert result["total"] == 5, f"expected total=5, got {result['total']}"
    assert sorted(result["missing_ids"]) == [2, 4], (
        f"expected missing_ids=[2,4], got {result['missing_ids']}"
    )
