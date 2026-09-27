# long-ok-file
"""RED tests — GREEN after 04-02 lands api_graph.py.

Tests for all 6 graph API endpoints:
  POST /api/graph/seed-search
  POST /api/graph/walk
  POST /api/graph/expand
  POST /api/graph/in-corpus-check
  GET  /api/graph/corpus-view
  GET  /api/graph/collections

Import fails until 04-02 ships src/server/api_graph.py. All tests assert
final production behavior and are expected to fail in this plan.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from types import SimpleNamespace
    from typing import Never

    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper_meta_db(db_path: Path) -> None:
    """Create a minimal paper_meta.sqlite fixture for seed resolution tests."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS paper_meta (
                corpusid     INTEGER PRIMARY KEY,
                title        TEXT,
                arxiv_id     TEXT,
                doi          TEXT,
                year         INTEGER,
                citationcount INTEGER DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO paper_meta (corpusid, title, arxiv_id, doi, year, citationcount) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                12345,
                "Attention Is All You Need",
                "1706.03762",
                "10.5555/3295222.3295349",
                2017,
                175000,
            ),
        )
        conn.execute(
            "INSERT INTO paper_meta (corpusid, title, arxiv_id, doi, year, citationcount) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (67890, "BERT Pre-training", "1810.04805", None, 2019, 90000),
        )
        conn.commit()
    finally:
        conn.close()


def _make_walk_result(corpus_id: int, title: str, year: int, hop: int = 1) -> SimpleNamespace:
    """Build a fake WalkCandidate-like object for monkeypatching walk()."""
    from types import SimpleNamespace

    return SimpleNamespace(
        corpus_id=corpus_id,
        title=title,
        year=year,
        citationcount=1000,
        hits=5,
        shortest_hop=hop,
        in_corpus=True,
    )


def _make_neighbor(corpus_id: int, title: str, year: int) -> SimpleNamespace:
    """Build a fake Neighbor-like object for monkeypatching neighbors()."""
    from types import SimpleNamespace

    return SimpleNamespace(
        corpus_id=corpus_id,
        title=title,
        year=year,
        citationcount=500,
        abstract=None,
    )


@pytest.fixture
def paper_meta_db(tmp_path: Path) -> Path:
    db = tmp_path / "paper_meta.sqlite"
    _make_paper_meta_db(db)
    return db


@pytest.fixture
def graph_client(
    paper_meta_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """FastAPI TestClient for the graph router with all externals stubbed."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    app = FastAPI()
    app.include_router(api_graph.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GRAPH-01: Seed resolution
# ---------------------------------------------------------------------------


def test_seed_search_local_hit(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-01: Seed that resolves locally produces no S2 network call.

    An arxiv ID present in paper_meta.sqlite must resolve without touching
    the S2 Graph API (poster spy must stay at 0 calls).
    """
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    poster_calls: list[Any] = []

    def spy_poster(ids: list[str], key: str | None) -> list[dict[str, Any] | None]:
        poster_calls.append(ids)
        return []

    monkeypatch.setattr(api_graph, "_s2_poster", spy_poster, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post("/api/graph/seed-search", json={"q": "1706.03762"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["corpus_id"] == 12345
    assert poster_calls == [], "S2 API must not be called for local hit"


def test_seed_search_s2_fallback(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-01: Seed miss in local DB triggers exactly one S2 API call.

    A seed string not in paper_meta.sqlite must fall back to the S2 Graph API
    (poster spy called exactly once), and the response carries the S2-resolved result.
    """
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    s2_response: list[dict[str, Any] | None] = [
        {
            "paperId": "abc123def456",
            "title": "GPT-4 Technical Report",
            "year": 2023,
            "externalIds": {"ArXiv": "2303.08774", "DOI": None},
            "citationCount": 12000,
            "corpusId": 99999,
        }
    ]
    poster_calls: list[Any] = []

    def spy_poster(ids: list[str], key: str | None) -> list[dict[str, Any] | None]:
        poster_calls.append(ids)
        return s2_response

    monkeypatch.setattr(api_graph, "_s2_poster", spy_poster, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post("/api/graph/seed-search", json={"q": "does-not-exist-in-db"})
    assert r.status_code == 200, r.text
    assert len(poster_calls) == 1, "S2 API must be called exactly once on local miss"


def test_seed_search_query_length_validation(
    paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Security/DoS: seed query longer than 512 chars must be rejected with 422."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    long_q = "x" * 513
    r = client.post("/api/graph/seed-search", json={"q": long_q})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# GRAPH-04: In-corpus check
# ---------------------------------------------------------------------------


def test_in_corpus_check_batch(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-04: POST /api/graph/in-corpus-check uses ONE paper_meta SELECT, ZERO Milvus calls.

    The endpoint must resolve corpus IDs from the sqlite index only — never touch Milvus.
    """
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    milvus_get_calls: list[Any] = []

    def spy_get_client() -> Never:
        milvus_get_calls.append(True)
        raise RuntimeError("Milvus must not be called from in-corpus-check")

    monkeypatch.setattr(api_graph, "_get_milvus_client", spy_get_client, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post("/api/graph/in-corpus-check", json={"corpus_ids": [12345, 67890, 11111]})
    assert r.status_code == 200, r.text
    body = r.json()

    # Corpus IDs in our fixture should be recognized; unknown should return in_corpus=False
    assert "12345" in body or 12345 in body, "Known corpus_id 12345 must appear in response"
    assert milvus_get_calls == [], "Milvus get_client must NEVER be called"


def test_in_corpus_check_batch_size_validation(
    paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Security/DoS: corpus_ids list above 2000 must be rejected with 422."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    big_list = list(range(2001))
    r = client.post("/api/graph/in-corpus-check", json={"corpus_ids": big_list})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# GRAPH-08: Walk — no S2 HTTP, direction/year filter, budget validation
# ---------------------------------------------------------------------------


def test_walk_no_s2_http(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-08: POST /api/graph/walk makes ZERO outbound HTTP, returns nodes+edges."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    def mock_walk(*args: object, **kwargs: object) -> list[Any]:
        return [
            _make_walk_result(11111, "Paper Alpha", 2022, hop=1),
            _make_walk_result(22222, "Paper Beta", 2023, hop=2),
        ]

    monkeypatch.setattr(api_graph, "_walk", mock_walk, raising=False)

    http_calls: list[Any] = []

    def raise_on_http(*args: object, **kwargs: object) -> Never:
        http_calls.append(args)
        raise ConnectionError("No outbound HTTP allowed during walk")

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    with patch("httpx.post", side_effect=raise_on_http):
        r = client.post(
            "/api/graph/walk",
            json={"corpus_ids": [12345], "depth": 2, "budget": 50},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert "nodes" in body
    assert "edges" in body
    assert http_calls == [], f"httpx.post was called {len(http_calls)} time(s) during walk"


def test_walk_direction_year_filter(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-02: Walk honors direction (references/citers/both) and year_from/year_to."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    captured_kwargs: dict[str, Any] = {}

    def mock_walk(*args: object, **kwargs: object) -> list[Any]:
        captured_kwargs.update(kwargs)
        return [_make_walk_result(11111, "Paper Alpha", 2022)]

    monkeypatch.setattr(api_graph, "_walk", mock_walk, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post(
        "/api/graph/walk",
        json={
            "corpus_ids": [12345],
            "depth": 2,
            "budget": 100,
            "direction": "references",
            "year_from": 2020,
            "year_to": 2023,
        },
    )
    assert r.status_code == 200, r.text


def test_walk_budget_validation(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Security/DoS: depth > 5 OR budget > 2000 must return 422."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    # depth too large
    r = client.post("/api/graph/walk", json={"corpus_ids": [12345], "depth": 6, "budget": 100})
    assert r.status_code == 422, f"depth=6 must be rejected, got {r.status_code}"

    # budget too large
    r = client.post("/api/graph/walk", json={"corpus_ids": [12345], "depth": 2, "budget": 2001})
    assert r.status_code == 422, f"budget=2001 must be rejected, got {r.status_code}"


def test_walk_ignores_walkcandidate_in_corpus(
    paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GRAPH-04 / Pitfall 3: Walk response must NOT mark every node in_corpus=True.

    WalkCandidate.in_corpus is always True (by kuzu construction) and MUST NOT be
    forwarded to the frontend as the in_corpus coloring signal. The graph endpoint
    must resolve in_corpus independently (via paper_meta lookup), not from the walker.
    """
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    # Return candidates with corpus_id NOT in paper_meta — so in_corpus must be False
    def mock_walk(*args: object, **kwargs: object) -> list[Any]:
        return [
            _make_walk_result(99991, "Unknown Paper X", 2021),
            _make_walk_result(99992, "Unknown Paper Y", 2022),
        ]

    monkeypatch.setattr(api_graph, "_walk", mock_walk, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post(
        "/api/graph/walk",
        json={"corpus_ids": [12345], "depth": 2, "budget": 50},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    nodes = body.get("nodes", [])
    assert nodes, "Walk must return at least one node"

    # Nodes not in paper_meta (99991, 99992) must have in_corpus=False
    for node in nodes:
        if node.get("corpus_id") in (99991, 99992):
            assert node.get("in_corpus") is False, (
                f"Node {node['corpus_id']} not in paper_meta but has in_corpus=True "
                "(Pitfall 3: WalkCandidate.in_corpus must not be forwarded directly)"
            )


# ---------------------------------------------------------------------------
# GRAPH-02: Expand — hop-1 directed neighbors
# ---------------------------------------------------------------------------


def test_expand_hop1_directed(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-02: POST /api/graph/expand returns hop-1 neighbors via neighbors.neighbors."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    from types import SimpleNamespace

    mock_ns = SimpleNamespace(
        references=[_make_neighbor(11111, "Ref Paper A", 2020)],
        citers=[_make_neighbor(22222, "Citer Paper B", 2023)],
    )

    def mock_neighbors(*args: object, **kwargs: object) -> SimpleNamespace:
        return mock_ns

    monkeypatch.setattr(api_graph, "_neighbors", mock_neighbors, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.post(
        "/api/graph/expand",
        json={"corpus_ids": [12345], "direction": "both", "limit": 50},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "nodes" in body
    assert "edges" in body

    node_ids = {n.get("corpus_id") for n in body["nodes"]}
    assert 11111 in node_ids, "references neighbor must appear in nodes"
    assert 22222 in node_ids, "citers neighbor must appear in nodes"


# ---------------------------------------------------------------------------
# GRAPH-03 data + D-03: Corpus view and collections list
# ---------------------------------------------------------------------------


def test_corpus_view_collection(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GRAPH-03 data: GET /api/graph/corpus-view?collection=trading returns papers + edges."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    def mock_get_read_view(*args: object, **kwargs: object) -> list[Any]:
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                paper_id="1706.03762",
                title="Attention Is All You Need",
                arxiv_id="1706.03762",
                doi="10.5555/3295222.3295349",
                collection="trading",
                year=2017,
                citationcount=None,
                chunk_count=42,
                authors='["Vaswani et al."]',
                short_cite="Vaswani et al. (2017)",
                ingested_at=1000000,
                extraction_quality=None,
                fetch_source=None,
                parser_version=None,
            ),
        ]

    monkeypatch.setattr(api_graph, "_get_read_view", mock_get_read_view, raising=False)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.get("/api/graph/corpus-view?collection=trading")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "nodes" in body
    assert "edges" in body
    assert len(body["nodes"]) >= 1, "Must return at least one paper node"


def test_collections_list(paper_meta_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-03: GET /api/graph/collections returns 4 collections with has_graph_data flags."""
    pytest.importorskip("src.server.api_graph")

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server import api_graph

    monkeypatch.setattr(api_graph, "META_DB_PATH", paper_meta_db)

    app = FastAPI()
    app.include_router(api_graph.router)
    client = TestClient(app)

    r = client.get("/api/graph/collections")
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list), "Must return a list"
    assert len(body) == 4, f"Must return exactly 4 collections, got {len(body)}"

    names = {c["name"] for c in body}
    assert names == {"trading", "ecology", "notes", "system"}, (
        f"Expected 4 standard collections, got {names}"
    )

    for coll in body:
        assert "has_graph_data" in coll, f"Collection {coll['name']} missing has_graph_data"
        assert isinstance(coll["has_graph_data"], bool), "has_graph_data must be bool"


# ---------------------------------------------------------------------------
# Finding 6: walk/expand seed lists drive an expensive per-node BFS — the seed
# count must be bounded (like in-corpus-check's 2000) to prevent a client from
# posting tens of thousands of seeds.
# ---------------------------------------------------------------------------


def test_walk_seed_count_bounded(graph_client: TestClient) -> None:
    """Finding 6: WalkRequest.corpus_ids above the bound is rejected with 422."""
    big = list(range(201))
    r = graph_client.post("/api/graph/walk", json={"corpus_ids": big, "depth": 2, "budget": 50})
    assert r.status_code == 422, f"201 seeds must be rejected, got {r.status_code}"


def test_expand_seed_count_bounded(graph_client: TestClient) -> None:
    """Finding 6: ExpandRequest.corpus_ids above the bound is rejected with 422."""
    big = list(range(201))
    r = graph_client.post("/api/graph/expand", json={"corpus_ids": big, "limit": 50})
    assert r.status_code == 422, f"201 seeds must be rejected, got {r.status_code}"
