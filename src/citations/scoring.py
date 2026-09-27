"""Multi-signal re-rank for walker candidates (Layer 2).

Fuses direct + co-citation + bibliographic-coupling + reach, times a recency
factor. ``norm`` sets the popularity correction (logidf damps the shared bridge
node; assoc/salton divide overlap by candidate citationcount). ``niche_bias``
b in [0,1] then applies ``score/(1+cit)^b`` to float relevant-but-obscure papers
up. Zero-overlap candidates fall back to walker reach, not the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import kuzu

from src.citations._kuzu_query import rows

NormKind = Literal["none", "logidf", "assoc", "salton"]


@dataclass(frozen=True)
class ScoredCandidate:
    corpus_id: int
    title: str | None
    year: int | None
    citationcount: int | None
    score: float
    niche: float
    direct: float
    cocite: float
    bibcouple: float
    reach: float
    recency: float
    shortest_hop: int
    in_corpus: bool | None = None
    abstract: str | None = None
    reliability: float | None = None


def _recency_factor(year: int | None, current_year: int, tau: float, floor: float) -> float:
    if year is None or tau <= 0:
        return 1.0
    age = max(0, current_year - int(year))
    return float(floor + (1.0 - floor) * math.exp(-age / tau))


def _normalize(raw: float, logw: float, cit_c: int, norm: NormKind) -> float:
    """Popularity-normalise an overlap count for one candidate.

    ``raw`` = shared-bridge count, ``logw`` = bridge-damped sum. assoc/salton
    divide raw by candidate citationcount; the seed count is ranking-invariant
    for a single seed, so dropped.
    """
    if norm == "none":
        return raw
    if norm == "logidf":
        return logw
    denom = max(1, cit_c)
    if norm == "assoc":
        return raw / denom
    return raw / math.sqrt(denom)  # salton


def score_candidates(
    con: kuzu.Connection,
    seeds: list[int],
    candidates: list[int],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.3,
    norm: NormKind = "logidf",
    niche_bias: float = 0.0,
    shortest_hop_by_id: dict[int, int] | None = None,
    hits_by_id: dict[int, int] | None = None,
    recency_tau: float = 5.0,
    recency_floor: float = 0.35,
    current_year: int | None = None,
) -> list[ScoredCandidate]:
    """Rerank ``candidates`` with combined bibliometric score via Cypher."""
    seed_list = list({int(s) for s in seeds})
    cand_list = list({int(c) for c in candidates})
    if not seed_list or not cand_list:
        return []

    hop_map = shortest_hop_by_id or {}
    hits_map = hits_by_id or {}
    anchor_year = current_year if current_year is not None else datetime.now().year

    direct_q = """
        MATCH (a:Paper)-[r:Cites]-(b:Paper)
        WHERE a.corpusid IN $seeds AND b.corpusid IN $cands
        WITH b.corpusid AS cid, count(r) AS s
        RETURN cid, s
    """
    # Both the raw overlap and the bridge-damped sum are returned so the chosen
    # norm is applied client-side without re-querying.
    cocite_q = """
        MATCH (other:Paper)-[:Cites]->(s:Paper),
              (other:Paper)-[:Cites]->(c:Paper)
        WHERE s.corpusid IN $seeds
          AND c.corpusid IN $cands
          AND NOT c.corpusid IN $seeds
        WITH c.corpusid AS cid, count(*) AS raw,
             sum(1.0 / log(2.0 + other.citationcount)) AS logw
        RETURN cid, raw, logw
    """
    bibcouple_q = """
        MATCH (s:Paper)-[:Cites]->(other:Paper),
              (c:Paper)-[:Cites]->(other:Paper)
        WHERE s.corpusid IN $seeds
          AND c.corpusid IN $cands
          AND NOT c.corpusid IN $seeds
        WITH c.corpusid AS cid, count(*) AS raw,
             sum(1.0 / log(2.0 + other.citationcount)) AS logw
        RETURN cid, raw, logw
    """
    meta_q = """
        MATCH (c:Paper)
        WHERE c.corpusid IN $cands
        RETURN c.corpusid, c.title, c.year, c.citationcount
    """

    # Kuzu rejects parameters that the query string does not reference, so each
    # statement gets exactly the params it uses (meta_q only filters on cands).
    signal_params = {"seeds": seed_list, "cands": cand_list}

    direct_by = {
        int(row[0]): float(row[1]) for row in rows(con, direct_q, signal_params)
    }

    def collect_overlap(query: str) -> dict[int, tuple[float, float]]:
        return {
            int(row[0]): (float(row[1]), float(row[2]))
            for row in rows(con, query, signal_params)
        }

    cocite_by = collect_overlap(cocite_q)
    bibcouple_by = collect_overlap(bibcouple_q)

    scored: list[ScoredCandidate] = []
    for row in rows(con, meta_q, {"cands": cand_list}):
        cid = int(row[0])
        year_val = int(row[2]) if row[2] is not None else None
        cit_c = int(row[3]) if row[3] is not None else 0
        d = direct_by.get(cid, 0.0)
        co_raw, co_log = cocite_by.get(cid, (0.0, 0.0))
        bc_raw, bc_log = bibcouple_by.get(cid, (0.0, 0.0))
        co = _normalize(co_raw, co_log, cit_c, norm)
        bc = _normalize(bc_raw, bc_log, cit_c, norm)
        shortest_hop = int(hop_map.get(cid, 0))
        hits = int(hits_map.get(cid, 0))
        reach = float(hits) / max(1, shortest_hop) if shortest_hop > 0 else 0.0
        base = alpha * d + beta * co + gamma * bc + delta * reach
        recency = _recency_factor(year_val, anchor_year, recency_tau, recency_floor)
        score = base * recency
        niche = score / (1.0 + cit_c) ** niche_bias if niche_bias else score
        scored.append(
            ScoredCandidate(
                corpus_id=cid,
                title=row[1],
                year=year_val,
                citationcount=cit_c if row[3] is not None else None,
                direct=d,
                cocite=co,
                bibcouple=bc,
                reach=reach,
                recency=recency,
                score=score,
                niche=niche,
                shortest_hop=shortest_hop,
            )
        )
    # Rank by niche-adjusted score (== score when bias=0); zero-overlap papers
    # fall back to reach so high-reach recent papers aren't buried.
    scored.sort(
        key=lambda s: (s.niche <= 0.0, -s.niche, -s.reach, -(s.citationcount or 0))
    )
    return scored
