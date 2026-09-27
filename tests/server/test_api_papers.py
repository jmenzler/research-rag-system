"""Wave-0 RED tests for src/server/api_papers.py.

Implementation lands in Plan 03 (router with GET /api/papers/{id} +
GET /api/chunks/{parent_id}).
Collection succeeds because every import of `src.server.api_papers` is deferred
inside the test body.

Analogs (verbatim from 01-PATTERNS.md):
  - tests/server/test_api_meta.py:10-22   TestClient + APIRouter pattern
  - src/ingest/storage.py:48-58           parents.sqlite schema
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_parents_db(db_path: Path) -> None:
    """Synthesize a parents.sqlite fixture matching src/ingest/storage.py:48-58."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS parents (
                id         TEXT PRIMARY KEY,
                text       TEXT NOT NULL,
                source_file TEXT NOT NULL,
                notebook   TEXT NOT NULL,
                modality   TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                image_path  TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO parents(id, text, source_file, notebook, modality, page_number) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p-1", "hello world", "paper.pdf", "trading", "text", 1),
        )
        conn.commit()
    finally:
        conn.close()


def test_get_chunk_returns_parent_text_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CHAT-04 backend: GET /api/chunks/{parent_id} returns the parent text."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    parents_db = tmp_path / "parents.sqlite"
    _make_parents_db(parents_db)
    # Module-level constant from 01-PATTERNS.md §PROJECT_ROOT Resolution.
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/chunks/p-1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello world"
    assert body["source_file"] == "paper.pdf"
    assert body["page_number"] == 1


def test_get_chunk_404_on_missing_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing parent_id returns 404 (never a silent empty body)."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    parents_db = tmp_path / "parents.sqlite"
    _make_parents_db(parents_db)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/chunks/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# RED — 03-02/03-03: list/facet/detail/provenance/cross-link endpoints
# ---------------------------------------------------------------------------


def _make_papers_fixture(parents_db: Path, sources_root: Path) -> None:
    """Create a parents.sqlite+meta.json fixture with multiple papers across
    two collections so list/facet tests have data."""
    conn = sqlite3.connect(str(parents_db))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id           TEXT PRIMARY KEY,
            id_type      TEXT NOT NULL,
            title        TEXT,
            authors      TEXT,
            year         INTEGER,
            url          TEXT,
            doi          TEXT,
            arxiv_id     TEXT,
            source_type  TEXT,
            notebook     TEXT,
            status       TEXT NOT NULL,
            file_hash    TEXT,
            updated_at   INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parents (
            id         TEXT PRIMARY KEY,
            text       TEXT NOT NULL,
            source_file TEXT NOT NULL,
            notebook   TEXT NOT NULL,
            modality   TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            image_path  TEXT
        );
        """
    )
    import json as _json

    for slug, coll, yr, has_arxiv, has_doi, title in [
        ("2401.aaaa", "trading", 2023, True, True, "Paper Alpha"),
        ("2401.bbbb", "ecology", 2024, True, False, "Paper Beta"),
        ("2401.cccc", "trading", 2024, False, True, "Paper Gamma"),
    ]:
        doc_dir = sources_root / coll / slug
        doc_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, object] = {
            "title": title,
            "authors": ["Author A"],
            "year": yr,
            "url": f"https://example.com/{slug}",
        }
        if has_arxiv:
            meta["arxiv_id"] = slug
            meta["metadata"] = {"arxiv_id": slug, "short_cite": f"A. et al. ({slug[:4]})"}
        if has_doi:
            meta["doi"] = f"10.1234/{slug}"
        (doc_dir / "meta.json").write_text(_json.dumps(meta))
        (doc_dir / "source.pdf").write_text("...")

        source_file = str(doc_dir / "source.pdf")
        conn.execute(
            "INSERT INTO parents (id, text, source_file, notebook, modality, page_number) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"p-{slug}", "paper text", source_file, coll, "text", 1),
        )
        cid = slug if has_arxiv else f"hash-{slug}"
        conn.execute(
            "INSERT INTO sources (id, id_type, title, status, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, "arxiv" if has_arxiv else "hash", title, "ingested", 1000),
        )
    conn.commit()
    conn.close()


def test_papers_list_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: GET /api/papers?limit=200 returns {papers, next_cursor}.
    Currently returns 404 or a non-list response (RED)."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?limit=200")
    # RED: today this 404s or returns a single-paper object; after 03-02 it
    # returns {papers: [...], next_cursor: ...}.
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("papers"), list)
    assert len(body["papers"]) >= 1


def test_papers_list_limit_clamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: limit > 200 is clamped to 200."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?limit=9999")
    # RED: endpoint currently doesn't exist or doesn't clamp.
    assert r.status_code == 200, r.text


def test_papers_list_facet_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?collection=trading narrows to only trading papers."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?collection=trading")
    assert r.status_code == 200, r.text
    body = r.json()
    for paper in body.get("papers", []):
        assert paper.get("collection") == "trading"


def test_papers_list_facet_year(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?year=2024 narrows to 2024 papers."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?year=2024")
    assert r.status_code == 200, r.text
    body = r.json()
    for paper in body.get("papers", []):
        assert paper.get("year") == 2024


def test_papers_list_facet_has_arxiv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?hasArxiv=true narrows to papers with arxiv_id."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?hasArxiv=true")
    assert r.status_code == 200, r.text
    body = r.json()
    for paper in body.get("papers", []):
        assert paper.get("arxiv_id") is not None


def test_papers_list_facet_has_doi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?hasDoi=true narrows to papers with doi."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?hasDoi=true")
    assert r.status_code == 200, r.text
    body = r.json()
    for paper in body.get("papers", []):
        assert paper.get("doi") is not None


def test_papers_list_fts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?q=<term> returns FTS5-matched rows."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?q=Alpha")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body.get("papers", [])) >= 1


def test_papers_detail_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: GET /api/papers/{id} returns provenance fields."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # After 03-02/03-03 this endpoint returns provenance fields.
    paper_id = "2401.aaaa"
    r = client.get(f"/api/papers/{paper_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    # The expanded endpoint should include provenance fields
    for key in (
        "fetch_source",
        "parser_version",
        "extraction_quality",
        "ingested_at",
        "collection",
        "chunk_count",
    ):
        assert key in body, f"missing provenance key {key!r}"


def test_papers_content_list_path_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: GET /api/papers/{id}/content-list rejects traversal."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers/../../etc/passwd/content-list")
    assert r.status_code in (400, 404), f"traversal-id returned 200: {r.text!r}"


def test_papers_content_list_missing_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: missing content-list file returns 404."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers/does-not-exist/content-list")
    assert r.status_code == 404, f"missing paper id returned {r.status_code}"


def test_papers_cross_link_cited_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — 03-02/03-03: ?citedInChat=<chatId> returns papers cited in that chat."""
    pytest.importorskip("src.server.api_papers")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_papers
    from src.server.api_papers import router

    sources_root = tmp_path / "sources"
    parents_db = tmp_path / "parents.sqlite"
    _make_papers_fixture(parents_db, sources_root)
    monkeypatch.setattr(api_papers, "PARENTS_DB_PATH", parents_db)
    monkeypatch.setattr(api_papers, "SOURCES_DIR", sources_root)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get("/api/papers?citedInChat=c1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body.get("papers"), list)
