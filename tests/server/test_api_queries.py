"""Wave-0 RED tests for src/server/api_queries.py.

Implementation lands in Plan 03 (D-14a — three per-id endpoints backing the
Audit Detail Sheet — CHAT-12 / D-13).
Collection succeeds because every import of `src.server.api_queries` is deferred
inside the test body.

Analogs:
  - tests/server/test_api_meta.py:10-22   TestClient + APIRouter pattern
  - tests/server/test_audit_smoke.py:22-23   PROJECT_ROOT / logs / queries idiom

Threat model (01-01-PLAN.md §threat_model T-01-WAVE0-01):
The path-traversal test deliberately sends `..%2F..%2Fetc%2Fpasswd`.  The server
MUST reject (400 or 404) — NEVER 200.  Regex constraints encoded:
  - query_id   : ^[A-Z0-9_]+$
  - filename   : ^[0-9]{2}_[a-z0-9_-]+\\.txt$
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_query_fixture(root: Path, query_id: str = "ABCDEF_1234_pid") -> Path:
    """Synthesize logs/queries/<date>/<query_id>/ with meta.json, report.md,
    one stage file, and one chunk file."""
    qdir = root / "2026-05-18" / query_id
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "meta.json").write_text(
        json.dumps({"totals": {"cost_usd": 0.01, "total_latency_ms": 4000}})
    )
    (qdir / "report.md").write_text("# Test report\nbody")
    (qdir / "03_milvus.json").write_text(
        json.dumps({"stage": "milvus", "latency_ms": 120, "n_hits": 25})
    )
    chunks_dir = qdir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    (chunks_dir / "01_abc.txt").write_text("hello chunk")
    return qdir


def test_get_query_returns_meta_and_report_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-13/D-14a endpoint #1: GET /api/queries/{qid} returns meta + report_md."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_query_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries/ABCDEF_1234_pid")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("meta"), dict)
    assert body["meta"]["totals"]["cost_usd"] == 0.01
    assert isinstance(body.get("report_md"), str)
    assert body["report_md"].lstrip().startswith("# Test report")


def test_get_query_stage_returns_payload_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-13/D-14a endpoint #2: GET /api/queries/{qid}/stages/{name}."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_query_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries/ABCDEF_1234_pid/stages/milvus")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"] == "milvus"
    assert body["latency_ms"] == 120
    assert body["n_hits"] == 25


def test_get_query_chunk_returns_raw_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-13/D-14a endpoint #3: GET /api/queries/{qid}/chunks/{filename}."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_query_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries/ABCDEF_1234_pid/chunks/01_abc.txt")
    assert r.status_code == 200, r.text
    assert r.text == "hello chunk"


def test_path_traversal_rejected_with_400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-01-WAVE0-01: path-traversal payloads MUST be rejected (400 or 404).

    Validates query_id regex (^[A-Z0-9_]+$) and filename regex
    (^[0-9]{2}_[a-z0-9_-]+\\.txt$).  Never 200 with a /etc/passwd payload.
    """
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_query_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r1 = client.get("/api/queries/..%2F..%2Fetc%2Fpasswd")
    assert r1.status_code in (400, 404), f"path traversal on query_id reached body=200: {r1.text!r}"

    r2 = client.get("/api/queries/ABCDEF_1234_pid/chunks/..%2F..%2Fetc%2Fpasswd")
    assert r2.status_code in (400, 404), f"path traversal on chunks/* reached body=200: {r2.text!r}"


# ---------------------------------------------------------------------------
# RED — 03-02/03-03: list/facet/cross-link endpoints
# ---------------------------------------------------------------------------


def _make_queries_fixture(root: Path) -> dict[str, str]:
    """Synthesize multiple query dirs under logs/queries/<date>/ with varied
    meta.json contents for list/facet testing.

    Returns a dict mapping query_id -> its dir path for cross-link assertions.
    """
    import json as _json

    qids: dict[str, str] = {}

    configs = [
        ("QF_000001_pid", "milvus", "success", 0.01, 4000, 100, 50, 150),
        ("QF_000002_pid", "paperqa", "success", 0.05, 15000, 500, 200, 700),
        ("QF_000003_pid", "milvus", "synthesis_skipped", 0.01, 3000, 100, 50, 150),
        ("QF_000004_pid", "fused", "success", 0.08, 25000, 800, 300, 1100),
    ]
    for qid, retriever, outcome, cost, latency, inp, outpt, total in configs:
        qdir = root / "2026-05-18" / qid
        qdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": {"collection": "trading", "gen_model": "gemini-2.5-flash"},
            "totals": {
                "total_latency_ms": latency,
                "tokens": {"input": inp, "output": outpt, "total": total},
                "cost_usd": cost,
            },
            "outcome": {
                "synthesis_skipped": outcome == "synthesis_skipped",
                "crag_retried": False,
                "crag_chose": False,
            },
        }
        (qdir / "meta.json").write_text(_json.dumps(meta))
        (qdir / "report.md").write_text(f"# {qid}\nretriever={retriever}")
        chunks_dir = qdir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        (chunks_dir / "01_abc.txt").write_text("chunk text")
        qids[qid] = str(qdir)
    return qids


def test_queries_list_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: GET /api/queries?limit=200 returns per-row totals."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?limit=200")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("queries"), list)
    assert len(body["queries"]) >= 1
    row = body["queries"][0]
    for key in (
        "query_id",
        "query",
        "ts_started",
        "model",
        "tokens",
        "cost_usd",
        "latency_ms",
        "outcome",
        "retriever",
    ):
        assert key in row, f"missing key {key!r} in query row"


def test_queries_list_facet_retriever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?retriever=milvus narrows to milvus queries."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?retriever=milvus")
    assert r.status_code == 200, r.text
    body = r.json()
    for q in body.get("queries", []):
        assert q.get("retriever") == "milvus"


def test_queries_list_facet_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?outcome=synthesis_skipped narrows."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?outcome=synthesis_skipped")
    assert r.status_code == 200, r.text
    body = r.json()
    for q in body.get("queries", []):
        assert "synthesis_skipped" in str(q.get("outcome", "")), (
            f"filtered row without synthesis_skipped outcome: {q}"
        )


def test_queries_list_facet_cost_min(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?costMin=0.03 filters by minimum cost."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?costMin=0.03")
    assert r.status_code == 200, r.text
    body = r.json()
    for q in body.get("queries", []):
        assert q.get("cost_usd", 0) >= 0.03, (
            f"costMin filter returned row with cost {q.get('cost_usd')}"
        )


def test_queries_list_facet_latency_min(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?latencyMin=10000 filters by minimum latency."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?latencyMin=10000")
    assert r.status_code == 200, r.text
    body = r.json()
    for q in body.get("queries", []):
        assert q.get("latency_ms", 0) >= 10000, (
            f"latencyMin filter returned row with latency {q.get('latency_ms')}"
        )


def test_queries_cross_link_paper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?paper=<paper_id> returns queries citing that paper."""
    pytest.importorskip("src.server.api_queries")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_queries
    from src.server.api_queries import router

    queries_root = tmp_path / "queries"
    _make_queries_fixture(queries_root)
    monkeypatch.setattr(api_queries, "QUERIES_LOG_DIR", queries_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/queries?paper=paper_abc123")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("queries"), list)
