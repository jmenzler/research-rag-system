# long-ok-file
"""Unit tests for the citation walker (Layer 1) and scoring (Layer 2).

The synthetic graph is small (~10 papers / ~14 edges) and built in an
in-memory KuzuDB so tests run in milliseconds. Topology covers the cases we
care about:

  - direct edges (seed → cand or cand → seed)
  - 2-hop reachability + shortest_hop tracking
  - shared citer (co-citation signal)
  - shared reference (bibliographic coupling signal)
  - a hub paper (citationcount well above HUB_THRESHOLD) to validate
    pass_through_hubs and IDF damping
  - an unreachable paper (must never appear)

Integration smoke against the real KuzuDB on backend host is gated by the
``RAG_INTEGRATION_KUZU`` env var so CI doesn't try to read a mounted snapshot.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from typing import cast

import kuzu
import pytest

from src.citations import abstracts as abstracts_mod
from src.citations import s2_client
from src.citations.abstracts import zstd
from src.citations.neighbors import neighbors
from src.citations.scoring import score_candidates
from src.citations.walker import HUB_THRESHOLD, hop1_counts, induced_edges, walk, walk_graph

# Paper ids — explicit numbers make assertions readable.
SEED = 1
CITED_BY_SEED = 2  # SEED -> 2
CITES_SEED = 3  # 3 -> SEED
BIBCOUPLE_PARTNER = 4  # 4 -> 11 (and SEED -> 11) so 4 ~ SEED via shared ref
COCITE_PARTNER = 5  # 12 -> 5 (and 12 -> SEED) so 5 ~ SEED via shared citer
HOP2_VIA_2 = 6  # 2 -> 6 (reaches via the direct neighbor)
HUB = 10  # very high citationcount; touched by many papers
SHARED_REF = 11  # both SEED and BIBCOUPLE_PARTNER cite this
SHARED_CITER = 12  # cites both SEED and COCITE_PARTNER
UNREACHABLE = 99

_NODES = [
    (SEED, "seed paper", 2024, 50),
    (CITED_BY_SEED, "cited-by-seed", 2023, 30),
    (CITES_SEED, "cites-seed", 2024, 25),
    (BIBCOUPLE_PARTNER, "bibcouple partner", 2024, 12),
    (COCITE_PARTNER, "cocite partner", 2024, 14),
    (HOP2_VIA_2, "hop-2 via cited-by-seed", 2023, 8),
    (HUB, "hub paper", 2017, HUB_THRESHOLD + 50_000),
    (SHARED_REF, "shared reference", 2010, 200),
    (SHARED_CITER, "shared citer", 2024, 9),
    (UNREACHABLE, "unreachable paper", 2024, 5),
]
_EDGES = [
    (SEED, CITED_BY_SEED),
    (CITES_SEED, SEED),
    (CITED_BY_SEED, HOP2_VIA_2),
    # Bibliographic coupling: SEED and BIBCOUPLE_PARTNER both cite SHARED_REF.
    (SEED, SHARED_REF),
    (BIBCOUPLE_PARTNER, SHARED_REF),
    # Co-citation: SHARED_CITER cites both SEED and COCITE_PARTNER.
    (SHARED_CITER, SEED),
    (SHARED_CITER, COCITE_PARTNER),
    # Hub: many papers cite the hub, including SEED and several candidates.
    (SEED, HUB),
    (CITED_BY_SEED, HUB),
    (CITES_SEED, HUB),
    (BIBCOUPLE_PARTNER, HUB),
    (COCITE_PARTNER, HUB),
    (SHARED_CITER, HUB),
    (HOP2_VIA_2, HUB),
]


def _build_graph() -> kuzu.Connection:
    """Build a tiny in-memory Paper + Cites graph in KuzuDB."""
    db = kuzu.Database()  # empty path → in-memory
    con = kuzu.Connection(db)
    con.execute(
        "CREATE NODE TABLE Paper("
        "corpusid INT64, title STRING, year INT64, citationcount INT64, "
        "PRIMARY KEY(corpusid))"
    )
    con.execute("CREATE REL TABLE Cites(FROM Paper TO Paper)")
    for cid, title, year, cit in _NODES:
        con.execute(
            "CREATE (:Paper {corpusid: $cid, title: $title, year: $year, citationcount: $cit})",
            {"cid": cid, "title": title, "year": year, "cit": cit},
        )
    for src, dst in _EDGES:
        con.execute(
            "MATCH (a:Paper {corpusid: $src}), (b:Paper {corpusid: $dst}) CREATE (a)-[:Cites]->(b)",
            {"src": src, "dst": dst},
        )
    return con


@pytest.fixture()
def graph() -> Iterator[kuzu.Connection]:
    con = _build_graph()
    yield con


# -------------------------- walker (Layer 1) ---------------------------------


def test_walker_hop1_direct_neighbors(graph: kuzu.Connection) -> None:
    cands = walk(graph, [SEED], max_hop=1)
    ids = {c.corpus_id for c in cands}
    assert CITED_BY_SEED in ids
    assert CITES_SEED in ids
    assert SHARED_REF in ids
    assert HUB in ids
    assert SHARED_CITER in ids
    assert UNREACHABLE not in ids
    assert HOP2_VIA_2 not in ids  # only reachable at hop 2


def test_walker_shortest_hop_distinguishes_layers(graph: kuzu.Connection) -> None:
    cands = walk(graph, [SEED], max_hop=2)
    by_id = {c.corpus_id: c for c in cands}
    assert by_id[CITED_BY_SEED].shortest_hop == 1
    assert by_id[HOP2_VIA_2].shortest_hop == 2


def test_walker_skips_seeds_by_default(graph: kuzu.Connection) -> None:
    cands = walk(graph, [SEED], max_hop=2)
    assert SEED not in {c.corpus_id for c in cands}


def test_walker_min_citation_threshold_filters(graph: kuzu.Connection) -> None:
    # SHARED_CITER has citationcount=9; threshold 20 should drop it.
    cands = walk(graph, [SEED], max_hop=2, min_citationcount=20)
    ids = {c.corpus_id for c in cands}
    assert SHARED_CITER not in ids
    assert HOP2_VIA_2 not in ids  # cit=8 also filtered
    assert CITES_SEED in ids  # cit=25 survives


def test_walker_per_hop_beam_caps_frontier(graph: kuzu.Connection) -> None:
    cands = walk(graph, [SEED], max_hop=2, per_hop_beam=2)
    # With beam=2 per hop and seed-skipping, we should never exceed 2 hops *
    # 2 entries = 4 candidates (modulo dedupe).
    assert len(cands) <= 4


def test_walker_pass_through_hubs_excludes_hub(graph: kuzu.Connection) -> None:
    cands = walk(graph, [SEED], max_hop=2, pass_through_hubs=True)
    ids = {c.corpus_id for c in cands}
    assert HUB not in ids
    # HUB still traversable, so HOP2_VIA_2 (which the hub also reaches) stays.
    assert HOP2_VIA_2 in ids


def test_walker_empty_seeds_returns_empty(graph: kuzu.Connection) -> None:
    assert walk(graph, [], max_hop=2) == []


def _build_hub_bridge_graph() -> kuzu.Connection:
    """seed → hub → gem, where the gem is reachable ONLY through the hub.

    Topology to exercise pass-through-hub bridging: gem (corpusid 5) has exactly
    one incoming edge, from the hub (10). Without re-admitting the hub, the gem
    is orphaned in the path-DAG.
    """
    db = kuzu.Database()
    con = kuzu.Connection(db)
    con.execute(
        "CREATE NODE TABLE Paper("
        "corpusid INT64, title STRING, year INT64, citationcount INT64, "
        "PRIMARY KEY(corpusid))"
    )
    con.execute("CREATE REL TABLE Cites(FROM Paper TO Paper)")
    nodes = [
        (1, "seed", 2024, 50),
        (10, "hub", 2017, HUB_THRESHOLD + 50_000),
        (5, "gem behind hub", 2024, 30),
    ]
    edges = [(1, 10), (10, 5)]
    for cid, title, year, cit in nodes:
        con.execute(
            "CREATE (:Paper {corpusid: $cid, title: $title, year: $year, citationcount: $cit})",
            {"cid": cid, "title": title, "year": year, "cit": cit},
        )
    for src, dst in edges:
        con.execute(
            "MATCH (a:Paper {corpusid: $src}), (b:Paper {corpusid: $dst}) CREATE (a)-[:Cites]->(b)",
            {"src": src, "dst": dst},
        )
    return con


def test_pass_through_hub_keeps_bridge_for_gem_behind_hub() -> None:
    """A gem reachable only via a pass-through hub must not be orphaned.

    The hub stays out of the ranked candidates but is re-admitted (flagged
    is_hub) so seed→hub→gem survives the path-DAG edge filter.
    """
    con = _build_hub_bridge_graph()
    cands, edges = walk_graph(con, [1], max_hop=2, pass_through_hubs=True)
    by_id = {c.corpus_id: c for c in cands}

    # Gem behind the hub is a real candidate.
    assert 5 in by_id
    assert by_id[5].is_hub is False

    # Hub is re-admitted as a flagged, unranked bridge node.
    assert 10 in by_id
    assert by_id[10].is_hub is True

    # Both bridge edges survive so the gem can be attached under the seed.
    eset = set(edges)
    assert (1, 10) in eset
    assert (10, 5) in eset


def test_pass_through_hub_bridge_renders_in_tree() -> None:
    """The bridged gem is reachable from the seed in the emitted tree."""
    from src.citations.cli import _emit_walk_tree

    con = _build_hub_bridge_graph()
    cands, edges = walk_graph(con, [1], max_hop=2, pass_through_hubs=True)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_walk_tree(1, "seed", cands, edges)
    tree = json.loads(buf.getvalue())

    # Collect every corpus_id reachable from the root.
    reached: set[int] = set()

    def _collect(node: dict[str, object]) -> None:
        cid = node.get("corpus_id")
        if isinstance(cid, int):
            reached.add(cid)
        kids = node.get("children", [])
        assert isinstance(kids, list)
        for k in kids:
            _collect(k)

    _collect(tree)
    assert 5 in reached  # gem behind hub is no longer orphaned


def test_hop1_counts_directed_and_min_cite(graph: kuzu.Connection) -> None:
    refs, citers = hop1_counts(graph, [SEED])
    assert refs == 3  # SEED → CITED_BY_SEED, SHARED_REF, HUB
    assert citers == 2  # CITES_SEED, SHARED_CITER → SEED
    # min_cite drops the obscure citer (SHARED_CITER cit=9), keeps CITES_SEED (25).
    _, citers20 = hop1_counts(graph, [SEED], min_citationcount=20)
    assert citers20 == 1
    assert hop1_counts(graph, []) == (0, 0)


def test_walk_graph_returns_reconstructable_paths(graph: kuzu.Connection) -> None:
    cands, edges = walk_graph(graph, [SEED], max_hop=2)
    eset = set(edges)
    # seed → hop-1 edge captured (direction preserved).
    assert (SEED, CITED_BY_SEED) in eset
    # hop-1 → hop-2 bridge captured, so SEED→CITED_BY_SEED→HOP2_VIA_2 is walkable.
    assert (CITED_BY_SEED, HOP2_VIA_2) in eset
    # Every edge endpoint is a returned node or the seed (self-contained DAG).
    node_ids = {c.corpus_id for c in cands} | {SEED}
    assert all(a in node_ids and b in node_ids for a, b in edges)


def test_walk_matches_walk_graph_candidates(graph: kuzu.Connection) -> None:
    # walk() and walk_graph() share the BFS core; candidate sets must match.
    plain = {c.corpus_id for c in walk(graph, [SEED], max_hop=2)}
    graphed = {c.corpus_id for c in walk_graph(graph, [SEED], max_hop=2)[0]}
    assert plain == graphed


def test_walk_tree_nests_children(
    graph: kuzu.Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.citations.cli import _emit_walk_tree

    cands, edges = walk_graph(graph, [SEED], max_hop=2)
    _emit_walk_tree(SEED, "seed paper", cands, edges)
    tree = json.loads(capsys.readouterr().out)

    assert tree["corpus_id"] == SEED
    assert "children" in tree  # hop-1 nodes nested under seed

    # SEED → CITED_BY_SEED → HOP2_VIA_2 means some hop-1 child nests a hop-2 child.
    def _has_grandchild(node: dict[str, object]) -> bool:
        kids = node.get("children", [])
        assert isinstance(kids, list)
        return any("children" in c for c in kids)

    assert _has_grandchild(tree)


def test_walk_mermaid_emits_graph_td(
    graph: kuzu.Connection, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.citations.cli import _emit_walk_mermaid

    cands, edges = walk_graph(graph, [SEED], max_hop=2)
    _emit_walk_mermaid(SEED, "seed paper", cands, edges)
    out = capsys.readouterr().out
    assert out.startswith("graph TD")
    assert f'N{SEED}["seed paper"]' in out
    assert "-->" in out  # at least one edge rendered


def test_expand_seeds_both_dedupes_mutual_citation() -> None:
    """direction='both' must emit a mutually-citing neighbour once per seed.

    When seed→C (reference) and C→seed (citer) both exist, the old concat of
    refs+citers rows emitted C twice, double-counting its per-seed reach in the
    walker's hop_hits.
    """
    from src.citations._kuzu_query import expand_seeds

    db = kuzu.Database()
    con = kuzu.Connection(db)
    con.execute(
        "CREATE NODE TABLE Paper("
        "corpusid INT64, title STRING, year INT64, citationcount INT64, "
        "PRIMARY KEY(corpusid))"
    )
    con.execute("CREATE REL TABLE Cites(FROM Paper TO Paper)")
    for cid in (1, 2, 3):
        con.execute(
            "CREATE (:Paper {corpusid: $cid, title: 't', year: 2024, citationcount: 10})",
            {"cid": cid},
        )
    # Mutual citation 1<->2, plus a one-way reference 1->3.
    for src, dst in [(1, 2), (2, 1), (1, 3)]:
        con.execute(
            "MATCH (a:Paper {corpusid: $src}), (b:Paper {corpusid: $dst}) CREATE (a)-[:Cites]->(b)",
            {"src": src, "dst": dst},
        )

    rows_out = expand_seeds(con, [1], direction="both")
    ids = sorted(int(r[0]) for r in rows_out)
    # 2 (mutual) emitted once, 3 (one-way) once → no double-count.
    assert ids == [2, 3]


def test_induced_edges_only_within_node_set(graph: kuzu.Connection) -> None:
    edges = set(induced_edges(graph, [SEED, CITED_BY_SEED, CITES_SEED]))
    assert (SEED, CITED_BY_SEED) in edges  # SEED -> CITED_BY_SEED
    assert (CITES_SEED, SEED) in edges  # CITES_SEED -> SEED
    # Edges to nodes outside the set are excluded (SEED -> SHARED_REF / HUB).
    assert (SEED, SHARED_REF) not in edges
    assert all(a in {SEED, CITED_BY_SEED, CITES_SEED} for a, _ in edges)
    assert induced_edges(graph, [SEED]) == []  # need >=2 nodes for an edge


# -------------------------- scoring (Layer 2) --------------------------------


def _by_id(scored: list) -> dict[int, object]:  # type: ignore[type-arg]
    return {s.corpus_id: s for s in scored}


def test_scoring_direct_citation_counted(graph: kuzu.Connection) -> None:
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED, CITES_SEED, UNREACHABLE],
        norm="none",
    )
    by = _by_id(scored)
    # Each direct edge counts once (seed -> cand or cand -> seed).
    assert by[CITED_BY_SEED].direct == 1.0  # type: ignore[attr-defined]
    assert by[CITES_SEED].direct == 1.0  # type: ignore[attr-defined]
    assert by[UNREACHABLE].direct == 0.0  # type: ignore[attr-defined]


def test_scoring_cocitation_picks_up_shared_citer(graph: kuzu.Connection) -> None:
    # SHARED_CITER cites both SEED and COCITE_PARTNER, so COCITE_PARTNER gets
    # a non-zero co-citation score even though it has no direct edge to SEED.
    scored = score_candidates(graph, [SEED], [COCITE_PARTNER], norm="none")
    s = scored[0]
    assert s.corpus_id == COCITE_PARTNER
    assert s.cocite >= 1.0
    assert s.direct == 0.0


def test_scoring_bibcouple_picks_up_shared_reference(graph: kuzu.Connection) -> None:
    # SEED and BIBCOUPLE_PARTNER both cite SHARED_REF → bibcouple signal.
    scored = score_candidates(graph, [SEED], [BIBCOUPLE_PARTNER], norm="none")
    s = scored[0]
    assert s.corpus_id == BIBCOUPLE_PARTNER
    assert s.bibcouple >= 1.0
    assert s.direct == 0.0


def test_scoring_idf_damping_suppresses_hub(graph: kuzu.Connection) -> None:
    # Many candidates share the HUB as a citation. Without damping, the hub
    # inflates every candidate's bibcouple score. With damping enabled, that
    # contribution shrinks toward zero (1 / log(2 + HUB_THRESHOLD + 50k)).
    cands = [CITED_BY_SEED, CITES_SEED, BIBCOUPLE_PARTNER, COCITE_PARTNER]
    plain = score_candidates(graph, [SEED], cands, norm="none")
    damped = score_candidates(graph, [SEED], cands, norm="logidf")

    plain_by = _by_id(plain)
    damped_by = _by_id(damped)
    for cid in cands:
        damped_score = cast(float, getattr(damped_by[cid], "bibcouple"))
        plain_score = cast(float, getattr(plain_by[cid], "bibcouple"))
        if plain_score > 0:
            assert damped_score < plain_score


def test_scoring_combined_weights_apply(graph: kuzu.Connection) -> None:
    # Disable recency + reach so the test isolates the topical-signal weights.
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED],
        alpha=2.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        norm="none",
        recency_tau=0.0,
    )
    s = scored[0]
    assert s.score == pytest.approx(2.0 * s.direct)
    assert s.reach == 0.0
    assert s.recency == 1.0


def test_scoring_shortest_hop_embedded(graph: kuzu.Connection) -> None:
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED, HOP2_VIA_2],
        shortest_hop_by_id={CITED_BY_SEED: 1, HOP2_VIA_2: 2},
    )
    by = _by_id(scored)
    assert by[CITED_BY_SEED].shortest_hop == 1  # type: ignore[attr-defined]
    assert by[HOP2_VIA_2].shortest_hop == 2  # type: ignore[attr-defined]


def test_scoring_reach_uses_hits_and_hop(graph: kuzu.Connection) -> None:
    # hits=10 at hop=2 → reach 5.0; hits=4 at hop=1 → reach 4.0.
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED, HOP2_VIA_2],
        shortest_hop_by_id={CITED_BY_SEED: 1, HOP2_VIA_2: 2},
        hits_by_id={CITED_BY_SEED: 4, HOP2_VIA_2: 10},
        recency_tau=0.0,
    )
    by = _by_id(scored)
    assert by[CITED_BY_SEED].reach == pytest.approx(4.0)  # type: ignore[attr-defined]
    assert by[HOP2_VIA_2].reach == pytest.approx(5.0)  # type: ignore[attr-defined]


def test_scoring_recency_boosts_recent(graph: kuzu.Connection) -> None:
    # Anchor at 2026: CITED_BY_SEED is 2023.
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED],
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        norm="none",
        recency_tau=5.0,
        recency_floor=0.35,
        current_year=2026,
    )
    s = scored[0]
    # age = 2026 - 2023 = 3; recency = 0.35 + 0.65 * exp(-3/5) ≈ 0.7068
    assert 0.69 < s.recency < 0.72  # type: ignore[attr-defined]
    assert s.score == pytest.approx(s.direct * s.recency)


def test_scoring_recency_disabled_when_tau_zero(graph: kuzu.Connection) -> None:
    scored = score_candidates(
        graph,
        [SEED],
        [CITED_BY_SEED],
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        delta=0.0,
        norm="none",
        recency_tau=0.0,
    )
    s = scored[0]
    assert s.recency == 1.0
    assert s.score == pytest.approx(s.direct)


def test_scoring_empty_inputs(graph: kuzu.Connection) -> None:
    assert score_candidates(graph, [], [1, 2]) == []
    assert score_candidates(graph, [SEED], []) == []


# -------------------- niche scoring (assoc norm + bias) ----------------------


def test_scoring_assoc_norm_divides_by_citationcount(graph: kuzu.Connection) -> None:
    # BIBCOUPLE_PARTNER (cit=12) shares two references with SEED (SHARED_REF and
    # HUB): raw overlap 2. assoc norm divides that by the candidate citationcount.
    none = score_candidates(graph, [SEED], [BIBCOUPLE_PARTNER], norm="none")[0]
    assoc = score_candidates(graph, [SEED], [BIBCOUPLE_PARTNER], norm="assoc")[0]
    assert none.bibcouple == pytest.approx(2.0)
    assert assoc.bibcouple == pytest.approx(2.0 / 12)


def test_scoring_niche_equals_score_when_bias_zero(graph: kuzu.Connection) -> None:
    plain = score_candidates(graph, [SEED], [CITES_SEED, COCITE_PARTNER], norm="none")
    assert all(s.niche == pytest.approx(s.score) for s in plain)


def test_scoring_niche_bias_demotes_high_citation(graph: kuzu.Connection) -> None:
    biased = score_candidates(
        graph,
        [SEED],
        [CITES_SEED, COCITE_PARTNER],
        norm="none",
        niche_bias=1.0,
    )
    for s in biased:
        if s.citationcount and s.score > 0:
            assert s.niche < s.score


def test_scoring_niche_reorders_toward_obscure(graph: kuzu.Connection) -> None:
    # CITES_SEED (cit=25) and SHARED_CITER (cit=9) both link directly to SEED.
    # High bias should rank the obscure SHARED_CITER above CITES_SEED.
    biased = score_candidates(
        graph,
        [SEED],
        [CITES_SEED, SHARED_CITER],
        norm="none",
        niche_bias=1.0,
    )
    order = [s.corpus_id for s in biased if s.niche > 0]
    assert order and order[0] == SHARED_CITER


# ----------------------------- neighbors -------------------------------------


def test_neighbors_splits_references_and_citers(graph: kuzu.Connection) -> None:
    nset = neighbors(graph, [SEED])
    ref_ids = {n.corpus_id for n in nset.references}
    citer_ids = {n.corpus_id for n in nset.citers}
    assert CITED_BY_SEED in ref_ids
    assert SHARED_REF in ref_ids
    assert CITES_SEED in citer_ids
    assert SHARED_CITER in citer_ids
    assert CITED_BY_SEED not in citer_ids  # directed split is real


def test_neighbors_direction_filter(graph: kuzu.Connection) -> None:
    refs_only = neighbors(graph, [SEED], direction="references")
    assert refs_only.citers == []
    assert refs_only.references
    citers_only = neighbors(graph, [SEED], direction="citers")
    assert citers_only.references == []
    assert citers_only.citers


def test_neighbors_min_cite_and_limit(graph: kuzu.Connection) -> None:
    capped = neighbors(graph, [SEED], direction="references", limit=1)
    assert len(capped.references) == 1  # citationcount desc → HUB first
    assert capped.references[0].corpus_id == HUB
    filtered = neighbors(graph, [SEED], direction="citers", min_citationcount=20)
    ids = {n.corpus_id for n in filtered.citers}
    assert CITES_SEED in ids  # cit=25 survives
    assert SHARED_CITER not in ids  # cit=9 dropped


def test_neighbors_empty_seed(graph: kuzu.Connection) -> None:
    ns = neighbors(graph, [])
    assert ns.references == [] and ns.citers == []


def test_neighbors_rerank_attaches_niche_and_reorders(graph: kuzu.Connection) -> None:
    # Plain citers are fame-sorted (CITES_SEED cit=25 before SHARED_CITER cit=9).
    plain = neighbors(graph, [SEED], direction="citers")
    plain_order = [n.corpus_id for n in plain.citers]
    assert plain_order.index(CITES_SEED) < plain_order.index(SHARED_CITER)

    # With niche rerank, the obscure citer outranks the famous one and carries
    # score/niche fields.
    ranked = neighbors(graph, [SEED], direction="citers", rerank=True, niche_bias=1.0)
    assert all(n.niche is not None and n.score is not None for n in ranked.citers)
    ranked_ids = [n.corpus_id for n in ranked.citers]
    assert ranked_ids.index(SHARED_CITER) < ranked_ids.index(CITES_SEED)


# ----------------------------- abstracts -------------------------------------


def test_abstracts_fetch_decodes_zstd() -> None:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE abstracts(corpusid INTEGER PRIMARY KEY, abstract_zstd BLOB)")
    con.execute(
        "INSERT INTO abstracts VALUES (?, ?)",
        (1, zstd.compress(b"hello abstract", level=6)),
    )
    con.execute("INSERT INTO abstracts VALUES (?, ?)", (2, None))
    con.commit()
    got = abstracts_mod.fetch(con, [1, 2, 999])
    assert got == {1: "hello abstract"}  # null + missing absent


# ----------------------------- s2_client -------------------------------------


def test_s2_client_parses_batch_and_falls_back_to_tldr() -> None:
    captured: dict[str, object] = {}

    def fake_poster(ids: list[str], key: str | None) -> list[dict[str, object] | None]:
        captured["ids"] = ids
        return [
            {"abstract": "real abstract"},
            {"abstract": None, "tldr": {"text": "the tldr"}},
            None,
        ]

    out = s2_client.fetch_abstracts([1, 2, 3], api_key="k", poster=fake_poster)
    assert out == {1: "real abstract", 2: "the tldr"}
    assert captured["ids"] == ["CorpusId:1", "CorpusId:2", "CorpusId:3"]


# ----------------------------- cli parser ------------------------------------


def test_cli_locators_builds_urls() -> None:
    from src.citations.cli import _locators

    full = _locators(123, "2208.06046", "10.1145/x")
    assert full["s2"] == "https://www.semanticscholar.org/paper/123"
    assert full["arxiv"] == "https://arxiv.org/abs/2208.06046"
    assert full["doi"] == "https://doi.org/10.1145/x"
    # Missing external ids → only the always-available S2 link.
    assert _locators(123, None, None) == {"s2": "https://www.semanticscholar.org/paper/123"}


def test_cli_parser_builds_without_format_errors() -> None:
    # argparse formats help eagerly (3.14), so a literal % in any help string
    # crashes parser construction. Building every subparser guards against it.
    from src.citations.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


# ------------------------ integration (optional) -----------------------------


@pytest.mark.skipif(
    not os.environ.get("RAG_INTEGRATION_KUZU"),
    reason="set RAG_INTEGRATION_KUZU=/path/to/citations.kuz to enable",
)
def test_walker_integration_lvr() -> None:
    db = kuzu.Database(os.environ["RAG_INTEGRATION_KUZU"], read_only=True)
    con = kuzu.Connection(db)
    lvr = 251554626
    cands = walk(con, [lvr], max_hop=2, per_hop_beam=5000, min_citationcount=5)
    assert len(cands) > 50
    scored = score_candidates(
        con,
        [lvr],
        [c.corpus_id for c in cands],
        shortest_hop_by_id={c.corpus_id: c.shortest_hop for c in cands},
    )
    assert len(scored) == len(cands)
