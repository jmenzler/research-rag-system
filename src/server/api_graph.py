# long-ok-file
"""Read-only graph API: wraps the frozen src/citations/ engine with HTTP.

Six endpoints over Kuzu + paper_meta.sqlite + abstracts.sqlite + the
papers_view cache. ADDITIVE-ONLY — never modifies the citations engine.

Security:
  - Input bounds enforced via Pydantic Field constraints → 422 on violation.
  - All SQL uses parameterised ``?`` placeholders (no f-string injection).
  - All Cypher params are bind params (no user values interpolated into query string).
  - S2 API called ONLY in seed-search and node-detail handlers (CRIT-6).
  - Zero Milvus calls in any handler.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.citations import DEFAULT_BASE, DEFAULT_RELEASE, seed_resolver
from src.citations.seed_resolver import SeedUnresolvableError
from src.server.papers_view import PaperRow, get_read_view

logger = logging.getLogger(__name__)


def _kuzu_rows(con: Any, query: str, params: dict[str, Any]) -> list[list[Any]]:  # noqa: ANN401
    """Lazy wrapper around _kuzu_query.rows — avoids top-level kuzu import."""
    from src.citations._kuzu_query import rows  # kuzu lazily imported at call time
    return rows(con, query, params)

# ---------------------------------------------------------------------------
# Config constants (D-02 — never hardcode paths inline)
# ---------------------------------------------------------------------------

KUZU_PATH = Path(DEFAULT_BASE) / DEFAULT_RELEASE / "citations.kuz"
META_DB_PATH = Path(DEFAULT_BASE) / DEFAULT_RELEASE / "paper_meta.sqlite"
ABS_DB_PATH = Path(DEFAULT_BASE) / DEFAULT_RELEASE / "abstracts.sqlite"

# Kuzu defaults its buffer pool to ~80% of *free* RAM at open. The PC runs this
# server (Qwen3-8b resident, ~20GB RSS) next to the trading stack + Milvus on a
# 31GB box that sits near 0 free, so the auto-sized pool is either tiny (walk
# OOMs) or large enough to trip the OOM-killer onto trading-paper. Pin an
# explicit modest cap instead — it is a lazy max, pages fault in on demand.
KUZU_BUFFER_POOL_GB = float(os.environ.get("GRAPH_KUZU_BUFFER_POOL_GB", "2.0"))
KUZU_NUM_THREADS = int(os.environ.get("GRAPH_KUZU_NUM_THREADS", "4"))

# A node above this citation count is recorded as a walk candidate but never
# expanded — traversing its 10^5 citers per hop dominates latency without
# surfacing relevant lineage. Mirrors walker.HUB_THRESHOLD.
_HUB_THRESHOLD = 50_000

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/graph")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SeedSearchRequest(BaseModel):
    q: str = Field(..., max_length=512)


class SeedSearchHit(BaseModel):
    corpus_id: int
    title: str
    arxiv_id: str | None = None
    doi: str | None = None
    confidence: float
    source: str


class GraphNode(BaseModel):
    corpus_id: int
    title: str | None = None
    year: int | None = None
    citation_count: int | None = None
    abstract: str | None = None
    authors: list[str] | None = None
    short_cite: str | None = None
    is_seed: bool = False
    in_corpus: bool = False


class NodeDetailResponse(BaseModel):
    corpus_id: int
    title: str | None = None
    year: int | None = None
    citation_count: int | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None


class GraphEdge(BaseModel):
    from_id: int
    to_id: int


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class CollectionInfo(BaseModel):
    name: str
    has_graph_data: bool


class WalkRequest(BaseModel):
    corpus_ids: list[int] = Field(..., max_length=200)
    depth: int = Field(2, ge=1, le=5)
    direction: str = Field("both", pattern="^(references|citers|both)$")
    year_from: int | None = None
    year_to: int | None = None
    min_citations: int = Field(0, ge=0)
    budget: int = Field(500, ge=1, le=2000)
    per_hop_beam: int = Field(400, ge=1, le=5000)


class ExpandRequest(BaseModel):
    corpus_ids: list[int] = Field(..., max_length=200)
    direction: str = Field("both", pattern="^(references|citers|both)$")
    limit: int = Field(50, ge=1, le=500)
    min_citations: int = Field(0, ge=0)


class InCorpusCheckRequest(BaseModel):
    corpus_ids: list[int] = Field(..., max_length=2000)


class InCorpusResult(BaseModel):
    in_corpus: bool
    collection: str | None = None


# ---------------------------------------------------------------------------
# Kuzu plumbing (Pitfall 4 — Database is process-singleton, Connection per-request)
# ---------------------------------------------------------------------------

_db: Any = None


def _get_db() -> Any:  # noqa: ANN401
    global _db
    if _db is None:
        try:
            import kuzu
            # read-only: a query API must not take Kuzu's exclusive write lock,
            # or the citations CLI (and other readers) can't open the graph.
            _db = kuzu.Database(
                str(KUZU_PATH),
                read_only=True,
                buffer_pool_size=int(KUZU_BUFFER_POOL_GB * 1024**3),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="Kuzu citation graph unavailable"
            ) from exc
    return _db


def _open_kuzu_con() -> Any:  # noqa: ANN401
    import kuzu
    return kuzu.Connection(_get_db(), num_threads=KUZU_NUM_THREADS)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _open_meta_readonly() -> sqlite3.Connection:
    uri = f"file:{META_DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="paper_meta store unavailable") from exc


def _open_abs_readonly() -> sqlite3.Connection:
    uri = f"file:{ABS_DB_PATH}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail="abstracts store unavailable") from exc


# ---------------------------------------------------------------------------
# S2 API helpers (injectable poster for test isolation)
# ---------------------------------------------------------------------------

_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _s2_poster(ids: list[str], api_key: str | None) -> list[dict[str, Any] | None]:
    """Default S2 poster — injectable so tests skip network.

    For seed-search: called with ``["query:<q>"]``; maps to the S2 paper/search
    endpoint and wraps result in the batch-response shape.
    For node-detail: called with ``["CorpusId:<id>"]``; maps to S2 paper/batch.
    """
    if not ids:
        return []
    api_key_header = {"x-api-key": api_key} if api_key else {}

    if ids[0].startswith("query:"):
        q = ids[0][len("query:"):]
        try:
            resp = httpx.get(
                _S2_SEARCH_URL,
                params={"query": q, "fields": "corpusId,title,year,externalIds,citationCount"},
                headers=api_key_header,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        results = data.get("data") or []
        return [results[0]] if results else [None]

    resp = httpx.post(
        _S2_BATCH_URL,
        params={"fields": "corpusId,title,year,externalIds,citationCount,authors,abstract,tldr"},
        json={"ids": ids},
        headers=api_key_header,
        timeout=30.0,
    )
    resp.raise_for_status()
    return list(resp.json())


def _resolve_via_s2(q: str, poster: Any = None) -> SeedSearchHit:  # noqa: ANN401
    """Resolve a seed string via S2 paper/search when local DB misses.

    Uses the injected ``poster`` so tests can stub the network call.
    Raises HTTPException(404) if S2 also has no match.
    """
    if poster is None:
        import src.server.api_graph as _self
        poster = _self._s2_poster
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    records = poster([f"query:{q}"], api_key)
    if not records or records[0] is None:
        raise HTTPException(status_code=404, detail=f"No paper found for seed: {q!r}")

    rec = records[0]
    ext = rec.get("externalIds") or {}
    corpus_id = rec.get("corpusId")
    if corpus_id is None:
        raise HTTPException(status_code=404, detail="S2 result missing corpusId")

    return SeedSearchHit(
        corpus_id=int(corpus_id),
        title=rec.get("title") or "",
        arxiv_id=ext.get("ArXiv"),
        doi=ext.get("DOI"),
        confidence=0.7,
        source="s2_search",
    )


def _fetch_node_via_s2(
    corpus_id: int, poster: Any = None  # noqa: ANN401
) -> tuple[list[str] | None, str | None]:
    """Fetch authors + abstract from S2 paper batch for a single corpus_id.

    Returns (authors, abstract) — either may be None if S2 has no data.
    poster is injected so tests skip network.
    """
    if poster is None:
        poster = _s2_poster
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    try:
        records = poster([f"CorpusId:{corpus_id}"], api_key)
    except Exception as exc:
        logger.warning("S2 batch fetch failed for %d: %s", corpus_id, exc)
        return None, None

    if not records or records[0] is None:
        return None, None

    rec = records[0]
    raw_authors = rec.get("authors")
    authors: list[str] | None = None
    if isinstance(raw_authors, list):
        authors = [
            str(a.get("name")) for a in raw_authors
            if isinstance(a, dict) and a.get("name") is not None
        ]
        if not authors:
            authors = None

    abstract: str | None = rec.get("abstract")
    if not abstract:
        tldr = rec.get("tldr")
        abstract = tldr.get("text") if isinstance(tldr, dict) else None

    return authors, abstract


# ---------------------------------------------------------------------------
# Walk helpers (injectable for test isolation)
# ---------------------------------------------------------------------------


def _walk(
    seeds: list[int],
    *,
    direction: str = "both",
    max_hop: int = 2,
    budget: int = 500,
    min_citations: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    per_hop_beam: int = 400,
) -> list[Any]:
    """Direction-aware per-hop BFS walk — injectable so tests skip kuzu.

    Expansion goes through :func:`expand_seeds`, which runs one literal-PK
    indexed query per frontier node — the ``WHERE corpusid IN $list`` shape this
    used to send forced a full 34M-row Paper scan that exhausted the buffer pool.
    Reach (``hits``) and the degree-normalised per-hop beam are computed in
    Python; the beam bounds the surviving frontier so a hub's many low-signal
    citers don't dominate the next hop. Called by walk_graph; monkeypatched by
    tests.

    Hubs (citationcount > ``_HUB_THRESHOLD``) are recorded as candidates but not
    themselves expanded — traversing a 100k-citer node returns 100k rows per hop
    and dominates latency (hub depth-3 was ~8 min before this), without adding
    relevant lineage.
    """
    from collections import Counter
    from math import log
    from types import SimpleNamespace

    from src.citations._kuzu_query import expand_seeds

    con = _open_kuzu_con()
    min_cit = int(min_citations)
    beam = int(per_hop_beam)

    candidates: dict[int, tuple[int, int]] = {}
    meta_by_id: dict[int, tuple[Any, Any, Any]] = {}
    visited: set[int] = set(seeds)
    frontier: list[int] = list(seeds)

    for hop in range(1, int(max_hop) + 1):
        if not frontier:
            break
        hop_hits: Counter[int] = Counter()
        for row in expand_seeds(
            con, frontier, direction=direction, min_citationcount=min_cit,
            year_from=year_from, year_to=year_to,
        ):
            cid = int(row[0])
            if cid in visited:
                continue
            hop_hits[cid] += 1
            if cid not in meta_by_id:
                meta_by_id[cid] = (row[1], row[2], row[3])

        if not hop_hits:
            break

        # Degree-normalised beam: keep the per_hop_beam highest-reach neighbours
        # so a hub's thousands of low-signal citers don't dominate the next hop.
        def _beam_key(item: tuple[int, int]) -> float:
            cit = meta_by_id[item[0]][2]
            return item[1] / log(2.0 + (float(cit) if cit is not None else 0.0))

        next_frontier: list[int] = []
        for cid, hits in sorted(hop_hits.items(), key=_beam_key, reverse=True)[:beam]:
            candidates[cid] = (hits, hop)
            visited.add(cid)
            cit = meta_by_id[cid][2]
            if cit is None or int(cit) <= _HUB_THRESHOLD:
                next_frontier.append(cid)
        frontier = next_frontier

    for s in seeds:
        candidates.pop(s, None)

    # Budget truncation must preserve connectivity: order by hop (closest to the
    # seed first), then by hits. Keeping any hop-h node then guarantees every
    # hop-(h-1) node — including its discovery parent — is also kept, so no node
    # is returned without an edge back toward the seed. Sorting by hits alone
    # dropped low-hits connector nodes while keeping their high-hits descendants,
    # orphaning those descendants in the returned graph.
    top = sorted(candidates.items(), key=lambda kv: (kv[1][1], -kv[1][0]))[: int(budget)]
    out: list[Any] = []
    for cid, (hits, _hop) in top:
        m = meta_by_id.get(cid, (None, None, None))
        out.append(SimpleNamespace(
            corpus_id=cid,
            title=m[0],
            year=int(m[1]) if m[1] is not None else None,
            citationcount=int(m[2]) if m[2] is not None else None,
            hits=hits,
            in_corpus=True,
        ))
    return out


def _neighbors(seeds: list[int], **kwargs: Any) -> Any:  # noqa: ANN401
    """Hop-1 neighbors — injectable so tests skip kuzu.

    Opens its own per-request kuzu connection.
    """
    con = _open_kuzu_con()
    from src.citations.neighbors import neighbors
    return neighbors(con, seeds, **kwargs)


def _kuzu_query(query: str, params: dict[str, Any]) -> list[list[Any]]:
    """Run a raw Cypher query — injectable so tests skip kuzu."""
    con = _open_kuzu_con()
    return _kuzu_rows(con, query, params)


def _get_read_view(*args: Any, **kwargs: Any) -> list[PaperRow]:  # noqa: ANN401
    """Thin wrapper over get_read_view for monkeypatching in tests."""
    return get_read_view(*args, **kwargs)


def _get_milvus_client() -> Any:  # noqa: ANN401
    """Placeholder — api_graph NEVER calls Milvus. Exists only to satisfy test spy."""
    raise RuntimeError("api_graph must never call Milvus")


# ---------------------------------------------------------------------------
# In-corpus helper
# ---------------------------------------------------------------------------


def _resolve_in_corpus(
    corpus_ids: list[int],
    meta_conn: sqlite3.Connection,
) -> dict[int, bool]:
    """Resolve which corpus_ids are present in the local papers corpus.

    Uses ONE batched SELECT from paper_meta then matches via the papers_view
    TTL cache (arxiv_id / doi cross-reference). Zero Milvus calls.
    """
    if not corpus_ids:
        return {}

    placeholders = ",".join("?" * len(corpus_ids))
    rows_meta = meta_conn.execute(
        f"SELECT corpusid, arxiv_id, doi FROM paper_meta WHERE corpusid IN ({placeholders})",
        corpus_ids,
    ).fetchall()

    view = _get_read_view()
    arxiv_index = {r["arxiv_id"]: r for r in view if r.get("arxiv_id")}
    doi_index = {r["doi"]: r for r in view if r.get("doi")}

    result: dict[int, bool] = {}
    for row in rows_meta:
        cid = int(row[0])
        arxiv_id = row[1]
        doi = row[2]
        found = (arxiv_id and arxiv_index.get(arxiv_id)) or (doi and doi_index.get(doi))
        result[cid] = bool(found)

    for cid in corpus_ids:
        if cid not in result:
            result[cid] = False

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/collections")
def list_collections() -> list[CollectionInfo]:
    """Return the 4 standard collections with has_graph_data flags.

    In v1, only 'trading' has graph data loaded into Kuzu (D-03).
    """
    graph_enabled = {"trading"}
    return [
        CollectionInfo(name=c, has_graph_data=(c in graph_enabled))
        for c in ("trading", "ecology", "notes", "system")
    ]


@router.post("/seed-search")
def seed_search(body: SeedSearchRequest) -> SeedSearchHit:
    """Resolve a seed string to a corpus_id; S2 fallback on local miss.

    Validates locally first via paper_meta.sqlite + seed_resolver.resolve().
    Calls S2 paper/search ONLY on SeedUnresolvableError (D-05, CRIT-6).
    Query length bounded to 512 chars by the request model (T-04-03).
    """
    q = body.q

    try:
        conn = _open_meta_readonly()
    except HTTPException:
        raise
    try:
        result = seed_resolver.resolve(conn, q)
    except SeedUnresolvableError:
        conn.close()
        return _resolve_via_s2(q)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return SeedSearchHit(
        corpus_id=result.corpus_id,
        title=result.title,
        arxiv_id=result.arxiv_id,
        doi=result.doi,
        confidence=result.confidence,
        source=result.source,
    )


@router.get("/corpus-view")
def corpus_view(collection: str = "trading") -> GraphResponse:
    """Return ingested papers and inter-citation edges for a collection.

    Loads the papers_view TTL cache to get corpus_ids for the collection,
    then runs one Kuzu MATCH to find inter-citation edges among those nodes.
    Empty result for non-trading collections in v1.
    """
    view = _get_read_view()

    def _row_get(row: Any, key: str) -> Any:  # noqa: ANN401
        if hasattr(row, "get"):
            return row.get(key)
        return getattr(row, key, None)

    coll_rows = [r for r in view if _row_get(r, "collection") == collection]

    if not coll_rows:
        return GraphResponse(nodes=[], edges=[])

    node_map: dict[int, GraphNode] = {}

    try:
        meta_conn = _open_meta_readonly()
    except HTTPException:
        return GraphResponse(nodes=[], edges=[])

    try:
        for row in coll_rows:
            arxiv_id = _row_get(row, "arxiv_id")
            doi = _row_get(row, "doi")
            title_hint: str | None = _row_get(row, "title")
            if not arxiv_id and not doi:
                continue

            sql_parts: list[str] = []
            sql_params: list[Any] = []
            if arxiv_id:
                sql_parts.append("arxiv_id = ?")
                sql_params.append(arxiv_id)
            if doi:
                sql_parts.append("lower(doi) = lower(?)")
                sql_params.append(doi)

            where_clause = " OR ".join(sql_parts)
            meta_row = meta_conn.execute(
                f"SELECT corpusid, title, year, citationcount "
                f"FROM paper_meta WHERE {where_clause} LIMIT 1",
                sql_params,
            ).fetchone()
            if meta_row is None:
                continue

            cid = int(meta_row[0])
            node_map[cid] = GraphNode(
                corpus_id=cid,
                title=meta_row[1] or title_hint,
                year=int(meta_row[2]) if meta_row[2] is not None else None,
                citation_count=int(meta_row[3]) if meta_row[3] is not None else None,
                short_cite=_row_get(row, "short_cite") or None,
                in_corpus=True,
            )
    finally:
        meta_conn.close()

    if not node_map:
        return GraphResponse(nodes=list(node_map.values()), edges=[])

    corpus_id_list = list(node_map.keys())
    corpus_id_set = set(corpus_id_list)
    edges: list[GraphEdge] = []

    try:
        edge_q = (
            "MATCH (s:Paper)-[:Cites]->(t:Paper) "
            "WHERE s.corpusid IN $ids AND t.corpusid IN $ids "
            "RETURN s.corpusid, t.corpusid"
        )
        edge_rows = _kuzu_query(edge_q, {"ids": corpus_id_list})
        for edge_row in edge_rows:
            from_id = int(edge_row[0])
            to_id = int(edge_row[1])
            if from_id in corpus_id_set and to_id in corpus_id_set:
                edges.append(GraphEdge(from_id=from_id, to_id=to_id))
    except Exception as exc:
        logger.warning("Kuzu corpus-view query failed: %s", exc)

    return GraphResponse(nodes=list(node_map.values()), edges=edges)


@router.post("/walk")
def walk_graph(body: WalkRequest) -> GraphResponse:
    """Direction-aware BFS walk over the Kuzu citation graph.

    Delegates to the injectable ``_walk`` function which implements
    direction-aware Cypher BFS (RESEARCH Pattern 4, Option A). Zero S2 HTTP
    (CRIT-6, GRAPH-08). in_corpus resolved from paper_meta — NOT from
    WalkCandidate.in_corpus (always True in Kuzu; Pitfall 3).
    """
    seed_ids = list({int(c) for c in body.corpus_ids})
    if not seed_ids:
        return GraphResponse(nodes=[], edges=[])

    import src.server.api_graph as _self

    try:
        candidates = _self._walk(
            seed_ids,
            direction=body.direction,
            max_hop=body.depth,
            budget=body.budget,
            min_citations=body.min_citations,
            year_from=body.year_from,
            year_to=body.year_to,
            per_hop_beam=body.per_hop_beam,
        )
    except Exception as exc:
        logger.error("Walk failed: %s", exc)
        raise HTTPException(status_code=503, detail="Citation graph unavailable") from exc

    cand_ids = [int(c.corpus_id) for c in candidates]
    all_corpus_ids = seed_ids + cand_ids

    in_corpus_map: dict[int, bool] = {}
    try:
        meta_conn = _open_meta_readonly()
        try:
            in_corpus_map = _resolve_in_corpus(all_corpus_ids, meta_conn)
        finally:
            meta_conn.close()
    except HTTPException:
        pass

    nodes: list[GraphNode] = []
    for s in seed_ids:
        nodes.append(GraphNode(
            corpus_id=s,
            is_seed=True,
            in_corpus=in_corpus_map.get(s, False),
        ))

    for c in candidates:
        cid = int(c.corpus_id)
        nodes.append(GraphNode(
            corpus_id=cid,
            title=getattr(c, "title", None),
            year=getattr(c, "year", None),
            citation_count=getattr(c, "citationcount", None),
            is_seed=False,
            in_corpus=in_corpus_map.get(cid, False),
        ))

    node_id_list = [n.corpus_id for n in nodes]
    node_id_set = set(node_id_list)
    edges: list[GraphEdge] = []
    if node_id_list:
        try:
            edge_q = (
                "MATCH (s:Paper)-[:Cites]->(t:Paper) "
                "WHERE s.corpusid IN $ids AND t.corpusid IN $ids "
                "RETURN s.corpusid, t.corpusid"
            )
            edge_rows = _self._kuzu_query(edge_q, {"ids": node_id_list})
            for row in edge_rows:
                from_id = int(row[0])
                to_id = int(row[1])
                if from_id in node_id_set and to_id in node_id_set:
                    edges.append(GraphEdge(from_id=from_id, to_id=to_id))
        except Exception as exc:
            logger.warning("Walk edge query failed: %s", exc)

    return GraphResponse(nodes=nodes, edges=edges)


@router.post("/expand")
def expand_graph(body: ExpandRequest) -> GraphResponse:
    """Return hop-1 directed neighbors of the seed corpus_ids.

    Never calls S2 (CRIT-6). Takes corpus_id ints only.
    """
    seed_ids = list({int(c) for c in body.corpus_ids})
    if not seed_ids:
        return GraphResponse(nodes=[], edges=[])

    import src.server.api_graph as _self

    try:
        ns = _self._neighbors(
            seed_ids,
            direction=body.direction,
            limit=body.limit,
            min_citationcount=body.min_citations,
        )
    except Exception as exc:
        logger.error("Expand failed: %s", exc)
        raise HTTPException(status_code=503, detail="Citation graph unavailable") from exc

    neighbor_nodes: dict[int, GraphNode] = {}
    for n in ns.references:
        if n.corpus_id not in neighbor_nodes:
            neighbor_nodes[n.corpus_id] = GraphNode(
                corpus_id=n.corpus_id,
                title=n.title,
                year=n.year,
                citation_count=n.citationcount,
            )
    for n in ns.citers:
        if n.corpus_id not in neighbor_nodes:
            neighbor_nodes[n.corpus_id] = GraphNode(
                corpus_id=n.corpus_id,
                title=n.title,
                year=n.year,
                citation_count=n.citationcount,
            )

    all_cids = seed_ids + list(neighbor_nodes.keys())
    in_corpus_map: dict[int, bool] = {}
    try:
        meta_conn = _open_meta_readonly()
        try:
            in_corpus_map = _resolve_in_corpus(all_cids, meta_conn)
        finally:
            meta_conn.close()
    except HTTPException:
        pass

    nodes: list[GraphNode] = []
    for s in seed_ids:
        nodes.append(GraphNode(
            corpus_id=s,
            is_seed=True,
            in_corpus=in_corpus_map.get(s, False),
        ))
    for cid, node in neighbor_nodes.items():
        node.in_corpus = in_corpus_map.get(cid, False)
        nodes.append(node)

    node_id_list = [n.corpus_id for n in nodes]
    node_id_set = set(node_id_list)
    edges: list[GraphEdge] = []
    if node_id_list:
        try:
            edge_q = (
                "MATCH (s:Paper)-[:Cites]->(t:Paper) "
                "WHERE s.corpusid IN $ids AND t.corpusid IN $ids "
                "RETURN s.corpusid, t.corpusid"
            )
            edge_rows = _self._kuzu_query(edge_q, {"ids": node_id_list})
            for row in edge_rows:
                from_id = int(row[0])
                to_id = int(row[1])
                if from_id in node_id_set and to_id in node_id_set:
                    edges.append(GraphEdge(from_id=from_id, to_id=to_id))
        except Exception as exc:
            logger.warning("Expand edge query failed: %s", exc)

    return GraphResponse(nodes=nodes, edges=edges)


@router.post("/in-corpus-check")
def in_corpus_check(body: InCorpusCheckRequest) -> dict[str, InCorpusResult]:
    """Batch check which corpus_ids are in the local papers corpus.

    One batched SELECT from paper_meta.sqlite, then matched against the
    papers_view TTL cache (arxiv_id / doi cross-reference). ZERO Milvus calls.
    Batch size bounded to 2000 by the request model (T-04-02).
    """
    corpus_ids = list({int(c) for c in body.corpus_ids})

    try:
        meta_conn = _open_meta_readonly()
    except HTTPException:
        raise

    try:
        if not corpus_ids:
            return {}

        placeholders = ",".join("?" * len(corpus_ids))
        rows_meta = meta_conn.execute(
            f"SELECT corpusid, arxiv_id, doi FROM paper_meta WHERE corpusid IN ({placeholders})",
            corpus_ids,
        ).fetchall()
    finally:
        meta_conn.close()

    view = _get_read_view()
    arxiv_index: dict[str, PaperRow] = {key: r for r in view if (key := r.get("arxiv_id"))}
    doi_index: dict[str, PaperRow] = {key: r for r in view if (key := r.get("doi"))}

    result: dict[str, InCorpusResult] = {}
    for row in rows_meta:
        cid = int(row[0])
        arxiv_id = row[1]
        doi = row[2]
        paper_row = (arxiv_id and arxiv_index.get(arxiv_id)) or (doi and doi_index.get(doi))
        if paper_row:
            if hasattr(paper_row, "get"):
                coll = paper_row.get("collection")
            else:
                coll = getattr(paper_row, "collection", None)
            result[str(cid)] = InCorpusResult(in_corpus=True, collection=coll)
        else:
            result[str(cid)] = InCorpusResult(in_corpus=False, collection=None)

    for cid in corpus_ids:
        if str(cid) not in result:
            result[str(cid)] = InCorpusResult(in_corpus=False, collection=None)

    return result


@router.get("/node/{corpus_id}")
def node_detail(corpus_id: int) -> NodeDetailResponse:
    """Return title, year, citations, authors, abstract for the node detail panel.

    Sources on-demand (ONLY on node click, never during walk/expand — GRAPH-08):
      1. title/year/citations/arxiv_id/doi from paper_meta.sqlite.
      2. abstract from abstracts.sqlite.
      3. If abstract missing OR authors needed → S2 paper batch (authors + abstract).
    Absent values return null (frontend renders '—' / 'No abstract available').
    """
    try:
        meta_conn = _open_meta_readonly()
    except HTTPException:
        raise

    try:
        row = meta_conn.execute(
            "SELECT corpusid, title, year, citationcount, arxiv_id, doi "
            "FROM paper_meta WHERE corpusid = ? LIMIT 1",
            (corpus_id,),
        ).fetchone()
    finally:
        meta_conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"corpus_id {corpus_id} not found")

    title = row[1]
    year = int(row[2]) if row[2] is not None else None
    citation_count = int(row[3]) if row[3] is not None else None
    arxiv_id: str | None = row[4]
    doi: str | None = row[5]

    abstract: str | None = None
    try:
        abs_conn = _open_abs_readonly()
        try:
            from src.citations import abstracts as abstracts_mod
            local_abstracts = abstracts_mod.fetch(abs_conn, [corpus_id])
            abstract = local_abstracts.get(corpus_id)
        finally:
            abs_conn.close()
    except (HTTPException, Exception) as exc:
        logger.warning("abstracts.sqlite lookup failed for %d: %s", corpus_id, exc)

    authors: list[str] | None = None
    if abstract is None:
        import src.server.api_graph as _self
        fetched_authors, fetched_abstract = _fetch_node_via_s2(corpus_id, poster=_self._s2_poster)
        authors = fetched_authors
        abstract = fetched_abstract

    return NodeDetailResponse(
        corpus_id=corpus_id,
        title=title,
        year=year,
        citation_count=citation_count,
        authors=authors,
        abstract=abstract,
        arxiv_id=arxiv_id,
        doi=doi,
    )
