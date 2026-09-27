"""Hop-1 directed neighbour view over the citation graph.

Splits the seed's immediate lineage: ``references`` (seed cites — foundational)
vs ``citers`` (cite the seed — descendants). Drives a stepwise frontier walk:
expand a node, read neighbours, pick the next node, repeat. ``rerank`` ranks each
side by the niche score (relevant-but-obscure first) instead of raw fame, so a
human or LLM steers toward gems. Abstracts attach via
:mod:`src.citations.abstracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import kuzu

from src.citations._kuzu_query import expand_seeds
from src.citations.scoring import NormKind, score_candidates

Direction = Literal["references", "citers", "both"]


@dataclass(frozen=True)
class Neighbor:
    corpus_id: int
    title: str | None
    year: int | None
    citationcount: int | None
    score: float | None = None
    niche: float | None = None
    abstract: str | None = None
    reliability: float | None = None
    in_corpus: bool | None = None


@dataclass(frozen=True)
class NeighborSet:
    references: list[Neighbor]
    citers: list[Neighbor]


def _to_neighbors(raw: list[list[Any]]) -> list[Neighbor]:
    out: list[Neighbor] = []
    for row in raw:
        out.append(
            Neighbor(
                corpus_id=int(row[0]),
                title=row[1],
                year=int(row[2]) if row[2] is not None else None,
                citationcount=int(row[3]) if row[3] is not None else None,
            )
        )
    return out


def _rank_hop1(raw: list[list[Any]], limit: int) -> list[Neighbor]:
    """Dedupe expand_seeds rows by corpusid, rank by citationcount DESC, cap.

    Replaces the in-Cypher ``DISTINCT ... ORDER BY citationcount DESC LIMIT`` —
    expand_seeds returns one row per (seed, neighbour), so the dedupe also folds
    a neighbour reached from several seeds into a single entry.
    """
    best: dict[int, list[Any]] = {}
    for row in raw:
        cid = int(row[0])
        if cid not in best:
            best[cid] = row
    ordered = sorted(
        best.values(),
        key=lambda r: -(int(r[3]) if r[3] is not None else 0),
    )
    return _to_neighbors(ordered[:limit])


def _scored_side(
    con: kuzu.Connection,
    direction: Direction,
    seed_list: list[int],
    anchor_list: list[int],
    *,
    min_citationcount: int,
    limit: int,
    pool: int,
    norm: NormKind,
    niche_bias: float,
) -> list[Neighbor]:
    # Indexed per-seed expansion (no full Paper scan), then take the lowest-cite
    # `pool` candidates so the niche rerank still sees obscure gems — the cap used
    # to be an in-Cypher ORDER BY citationcount ASC LIMIT.
    cit_by_id: dict[int, int] = {}
    for row in expand_seeds(
        con, seed_list, direction=direction, min_citationcount=int(min_citationcount)
    ):
        cid = int(row[0])
        if cid not in cit_by_id:
            cit_by_id[cid] = int(row[3]) if row[3] is not None else 0
    pool_ids = [
        cid for cid, _ in sorted(cit_by_id.items(), key=lambda kv: kv[1])[: int(pool)]
    ]
    if not pool_ids:
        return []
    scored = score_candidates(
        con, anchor_list, pool_ids,
        norm=norm, niche_bias=niche_bias,
        shortest_hop_by_id=dict.fromkeys(pool_ids, 1),
    )
    return [
        Neighbor(
            corpus_id=s.corpus_id,
            title=s.title,
            year=s.year,
            citationcount=s.citationcount,
            score=s.score,
            niche=s.niche,
        )
        for s in scored[:limit]
    ]


def neighbors(
    con: kuzu.Connection,
    seeds: list[int],
    *,
    direction: Direction = "both",
    limit: int = 50,
    min_citationcount: int = 0,
    rerank: bool = False,
    anchors: list[int] | None = None,
    norm: NormKind = "assoc",
    niche_bias: float = 0.0,
    pool: int = 2000,
) -> NeighborSet:
    """Return the seed's hop-1 references and/or citers.

    Default: citation-count ranked. With ``rerank``, each side is niche-scored
    against ``anchors`` (default the expanded ``seeds``) and ranked by that —
    pass the original seed as ``anchors`` to keep a multi-step walk on-topic.
    """
    seed_list = list({int(s) for s in seeds})
    if not seed_list:
        return NeighborSet(references=[], citers=[])

    want_refs = direction in ("references", "both")
    want_citers = direction in ("citers", "both")

    if rerank:
        anchor_list = list({int(a) for a in anchors}) if anchors else seed_list

        def side(side_direction: Direction) -> list[Neighbor]:
            return _scored_side(
                con, side_direction, seed_list, anchor_list,
                min_citationcount=min_citationcount, limit=limit,
                pool=pool, norm=norm, niche_bias=niche_bias,
            )

        refs = side("references") if want_refs else []
        citers = side("citers") if want_citers else []
        return NeighborSet(references=refs, citers=citers)

    refs = (
        _rank_hop1(
            expand_seeds(con, seed_list, direction="references",
                         min_citationcount=min_citationcount),
            int(limit),
        )
        if want_refs else []
    )
    citers = (
        _rank_hop1(
            expand_seeds(con, seed_list, direction="citers",
                         min_citationcount=min_citationcount),
            int(limit),
        )
        if want_citers else []
    )
    return NeighborSet(references=refs, citers=citers)
