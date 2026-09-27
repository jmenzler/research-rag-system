# long-ok-file
"""Citation-graph walker: iterative per-hop BFS over the KuzuDB graph.

KuzuDB stores citations as CSR adjacency lists keyed by node, so a hop-2
expansion on 200 frontier papers reads ~10 MB instead of scanning the full
edge table. Cold walks finish in seconds rather than minutes.

BFS is driven from Python with one single-hop Cypher MATCH per iteration. An
earlier version used a single variable-length match ``(s)-[r:Cites*1..N]-(c)``
which Kuzu materialises as full-path enumeration — hop-3 on hub-adjacent seeds
(e.g. Attention is All You Need, 175k citers) OOMs the engine. The per-hop
loop bounds memory deterministically via ``per_hop_beam`` and lets
``pass_through_hubs`` traverse popular papers without polluting the output.

Layer 1 (this module) returns reach-ranked candidates; Layer 2
(:mod:`src.citations.scoring`) re-ranks them with the multi-signal score.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import log

import kuzu

from src.citations._kuzu_query import expand_seeds, rows

HUB_THRESHOLD = 50_000


def hop1_counts(
    con: kuzu.Connection, seeds: list[int], *, min_citationcount: int = 0
) -> tuple[int, int]:
    """Return (references, citers) hop-1 degree of the seed, honouring min-cite.

    A cheap pre-flight: lets a caller size the neighbourhood (and decide whether to
    raise --min-cite / lower --niche-bias) before a full walk.
    """
    ids = list({int(s) for s in seeds})
    if not ids:
        return (0, 0)
    refs = {
        int(row[0]) for row in expand_seeds(
            con, ids, direction="references", min_citationcount=min_citationcount)
    }
    cits = {
        int(row[0]) for row in expand_seeds(
            con, ids, direction="citers", min_citationcount=min_citationcount)
    }
    return (len(refs), len(cits))


def induced_edges(con: kuzu.Connection, node_ids: list[int]) -> list[tuple[int, int]]:
    """Return the directed ``Cites`` edges whose both endpoints are in ``node_ids``.

    Lets a multi-hop walk be returned as a subgraph (nodes + edges) so an LLM can
    reconstruct seed→…→candidate paths instead of reading a flat ranked list.
    """
    ids = list({int(n) for n in node_ids})
    if len(ids) < 2:
        return []
    edge_rows = rows(
        con,
        "MATCH (a:Paper)-[:Cites]->(b:Paper) "
        "WHERE a.corpusid IN $ids AND b.corpusid IN $ids "
        "RETURN a.corpusid, b.corpusid",
        {"ids": ids},
    )
    return [(int(r[0]), int(r[1])) for r in edge_rows]


@dataclass(frozen=True)
class WalkCandidate:
    corpus_id: int
    title: str | None
    year: int | None
    citationcount: int | None
    hits: int
    shortest_hop: int
    in_corpus: bool | None = None
    abstract: str | None = None
    reliability: float | None = None
    # True for a pass-through hub retained only to keep its bridge edges in the
    # path-DAG (graph/tree/mermaid). Unranked: never in the plain walk() output.
    is_hub: bool = False


_LAYER_EDGE_Q = """
    MATCH (a:Paper)-[:Cites]->(b:Paper)
    WHERE (a.corpusid IN $frontier AND b.corpusid IN $children)
       OR (a.corpusid IN $children AND b.corpusid IN $frontier)
    RETURN a.corpusid, b.corpusid
"""


def _bridge_hubs(
    edges: set[tuple[int, int]], keep: set[int], hubs: set[int]
) -> set[int]:
    """Hubs on a kept→…→kept path through hub nodes only.

    A hub is a bridge iff it is forward-reachable from a kept node and can also
    backward-reach a kept node, in both cases travelling only through hubs at
    the intermediate steps. Re-admitting these keeps seed→hub→gem (and longer
    hub chains) connected in the path-DAG instead of orphaning the bridged gem.
    """
    if not hubs:
        return set()

    fwd_adj: dict[int, list[int]] = {}
    bwd_adj: dict[int, list[int]] = {}
    for a, b in edges:
        fwd_adj.setdefault(a, []).append(b)
        bwd_adj.setdefault(b, []).append(a)

    def _reach(adj: dict[int, list[int]]) -> set[int]:
        # BFS from keep; only step *into* a hub (kept nodes are terminal anchors).
        seen: set[int] = set()
        stack = list(keep)
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt in hubs and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    return _reach(fwd_adj) & _reach(bwd_adj)


def _walk_core(
    con: kuzu.Connection,
    seeds: Iterable[int],
    *,
    max_hop: int,
    budget: int,
    skip_seeds: bool,
    min_citationcount: int,
    per_hop_beam: int | None,
    pass_through_hubs: bool,
    collect_edges: bool,
) -> tuple[list[WalkCandidate], list[tuple[int, int]]]:
    seed_list = list({int(s) for s in seeds})
    if not seed_list:
        return [], []

    min_cit = int(min_citationcount)
    visited: set[int] = set(seed_list)
    candidates: dict[int, tuple[int, int]] = {}
    meta: dict[int, tuple[str | None, int | None, int | None]] = {}
    frontier: list[int] = list(seed_list)
    edges: set[tuple[int, int]] = set()
    # Pass-through hubs (traversed but unranked); keyed to their first-seen hop so
    # the graph path can re-attach them as bridge nodes without ranking them.
    hub_hops: dict[int, int] = {}

    for hop in range(1, int(max_hop) + 1):
        if not frontier:
            break
        # One literal-PK indexed query per frontier node (see expand_seeds): the
        # old WHERE corpusid IN $frontier shape scanned the whole 34M-row Paper
        # table and OOM'd under a modest buffer pool. Reach and the beam are now
        # aggregated in Python.
        hop_hits: Counter[int] = Counter()
        for row in expand_seeds(
            con, frontier, direction="both", min_citationcount=min_cit
        ):
            cid = int(row[0])
            if cid in visited:
                continue
            hop_hits[cid] += 1
            if cid not in meta:
                meta[cid] = (row[1], row[2], row[3])
        if not hop_hits:
            frontier = []
            break

        items: Iterable[tuple[int, int]] = hop_hits.items()
        if per_hop_beam:
            # Degree-normalised reach so hubs don't dominate the surviving frontier.
            def _beam_key(item: tuple[int, int]) -> float:
                cit = meta[item[0]][2]
                return item[1] / log(2.0 + (float(cit) if cit is not None else 0.0))

            items = sorted(hop_hits.items(), key=_beam_key, reverse=True)[
                : int(per_hop_beam)
            ]

        next_frontier: list[int] = []
        for cid, hits_inc in items:
            cit = meta[cid][2]
            is_pass_through_hub = (
                pass_through_hubs and cit is not None and int(cit) > HUB_THRESHOLD
            )
            if not is_pass_through_hub:
                candidates[cid] = (hits_inc, hop)
            elif cid not in hub_hops:
                hub_hops[cid] = hop
            next_frontier.append(cid)
            visited.add(cid)
        # Capture the directed Cites edges between this frontier and the layer it
        # just discovered, so the walk can be returned as a path-DAG (the bridge
        # nodes that connect seed→…→candidate would be lost in a top-N induced
        # subgraph). Both endpoints are bounded sets here, so the IN match is
        # cheap (it never scans the full table).
        if collect_edges and next_frontier:
            for er in rows(
                con, _LAYER_EDGE_Q,
                {"frontier": frontier, "children": next_frontier},
            ):
                edges.add((int(er[0]), int(er[1])))
        frontier = next_frontier

    if skip_seeds:
        for s in seed_list:
            candidates.pop(s, None)

    if not candidates:
        return [], []

    out: list[WalkCandidate] = []
    for cid, (hits, shortest_hop) in candidates.items():
        m = meta.get(cid, (None, None, None))
        out.append(
            WalkCandidate(
                corpus_id=cid,
                title=m[0],
                year=int(m[1]) if m[1] is not None else None,
                citationcount=int(m[2]) if m[2] is not None else None,
                hits=hits,
                shortest_hop=shortest_hop,
            )
        )
    out.sort(key=lambda c: (-c.hits, -(c.citationcount or 0)))
    out = out[: int(budget)]

    if collect_edges:
        keep = {c.corpus_id for c in out} | set(seed_list)
        # A gem reachable only through a pass-through hub has no incoming edge in
        # the path-DAG once the hub is filtered out — the renderers then orphan
        # it. Re-admit hubs that lie on a kept→…→kept path through hubs, so
        # seed→hub→gem (and seed→hub→hub→gem chains) survive the edge filter.
        bridges = _bridge_hubs(edges, keep, set(hub_hops))
        for h in bridges:
            m = meta.get(h, (None, None, None))
            out.append(
                WalkCandidate(
                    corpus_id=h,
                    title=m[0],
                    year=int(m[1]) if m[1] is not None else None,
                    citationcount=int(m[2]) if m[2] is not None else None,
                    hits=0,
                    shortest_hop=hub_hops[h],
                    is_hub=True,
                )
            )
        keep |= bridges
        edges = {(a, b) for a, b in edges if a in keep and b in keep}
    return out, sorted(edges)


def walk(
    con: kuzu.Connection,
    seeds: Iterable[int],
    *,
    max_hop: int = 2,
    budget: int = 500,
    skip_seeds: bool = True,
    min_citationcount: int = 0,
    per_hop_beam: int | None = None,
    pass_through_hubs: bool = False,
) -> list[WalkCandidate]:
    """Iterative per-hop BFS over the Kuzu citation graph; reach-ranked candidates.

    ``per_hop_beam`` caps the frontier per hop by degree-normalised reach;
    ``pass_through_hubs`` traverses hubs without recording them. See
    :func:`walk_graph` for the variant that also returns traversal edges.
    """
    out, _ = _walk_core(
        con, seeds, max_hop=max_hop, budget=budget, skip_seeds=skip_seeds,
        min_citationcount=min_citationcount, per_hop_beam=per_hop_beam,
        pass_through_hubs=pass_through_hubs, collect_edges=False,
    )
    return out


def walk_graph(
    con: kuzu.Connection,
    seeds: Iterable[int],
    *,
    max_hop: int = 2,
    budget: int = 500,
    skip_seeds: bool = True,
    min_citationcount: int = 0,
    per_hop_beam: int | None = None,
    pass_through_hubs: bool = False,
) -> tuple[list[WalkCandidate], list[tuple[int, int]]]:
    """Like :func:`walk` but also returns the directed BFS edges (incl. seed→hop1).

    Edges are restricted to the returned candidate set ∪ seeds, so the result is
    a self-contained path-DAG: seed → bridge → … → candidate.
    """
    return _walk_core(
        con, seeds, max_hop=max_hop, budget=budget, skip_seeds=skip_seeds,
        min_citationcount=min_citationcount, per_hop_beam=per_hop_beam,
        pass_through_hubs=pass_through_hubs, collect_edges=True,
    )
