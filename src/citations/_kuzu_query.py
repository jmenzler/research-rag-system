# long-ok-file
"""Shared helper for querying the KuzuDB citation graph.

Isolates the ``kuzu`` import (so :mod:`seed_resolver` and the package root stay
importable without kuzu installed) and narrows kuzu's loose return types —
``execute() -> QueryResult | list[QueryResult]`` and
``get_next() -> list[Any] | dict[str, Any]`` — to the single-statement,
list-row case the walker and scorer always use.
"""

from __future__ import annotations

from typing import Any, cast

import kuzu


def rows(con: kuzu.Connection, query: str, params: dict[str, Any]) -> list[list[Any]]:
    """Run a single-statement Cypher query and return its rows as lists."""
    result = cast(kuzu.QueryResult, con.execute(query, params))
    try:
        out: list[list[Any]] = []
        while result.has_next():
            out.append(cast("list[Any]", result.get_next()))
        return out
    finally:
        # Release the QueryResult's buffer-pool pages immediately. Callers that
        # run several queries on one connection (the BFS walk) otherwise pin
        # pages until GC, exhausting Kuzu's buffer pool ("no memory could be
        # freed").
        result.close()


def expand_seeds(
    con: kuzu.Connection,
    frontier: list[int],
    *,
    direction: str,
    min_citationcount: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[list[Any]]:
    """Index-driven hop-1 expansion of every node in ``frontier``.

    Kuzu 0.11.3 fires the ``corpusid`` primary-key index ONLY for a literal
    scalar equality. ``WHERE s.corpusid IN $list``, a bound ``= $param``, and
    ``UNWIND $list AS id MATCH (s {corpusid: id})`` all fall back to a full scan
    of the 34M-row Paper table, which pins the whole table in the buffer pool
    until it OOMs regardless of pool size. So each corpusid (an int — safe to
    interpolate) becomes its own literal-equality query (one PRIMARY_KEY_SCAN +
    CSR adjacency read), and the per-seed rows are unioned. The far-node filters
    stay as bind params; they apply after the traversal and keep the PK scan.

    Returns ``[corpusid, title, year, citationcount]`` rows for the neighbours;
    a neighbour recurs once per frontier node that reaches it, so callers
    aggregate (count = reach, dedupe for a flat set).
    """
    if not frontier:
        return []
    year_clause = (
        " AND (c.year IS NULL OR c.year >= $year_from)" if year_from is not None else ""
    ) + (
        " AND (c.year IS NULL OR c.year <= $year_to)" if year_to is not None else ""
    )
    params: dict[str, Any] = {"min_cit": int(min_citationcount)}
    if year_from is not None:
        params["year_from"] = int(year_from)
    if year_to is not None:
        params["year_to"] = int(year_to)

    want_refs = direction in ("references", "both")
    want_citers = direction in ("citers", "both")
    out: list[list[Any]] = []
    for raw_sid in frontier:
        sid = int(raw_sid)
        # For direction="both", a neighbour reachable via BOTH a ref edge (s->c)
        # and a citer edge (c->s) — a mutual citation — would otherwise be
        # emitted twice for this seed, double-counting its per-seed reach in the
        # walker's hop_hits. Union the two directions per seed by neighbour id.
        seen_nbr: set[int] = set()
        for want, query in (
            (want_refs,
             f"MATCH (s:Paper)-[:Cites]->(c:Paper) WHERE s.corpusid = {sid} "
             f"AND c.citationcount >= $min_cit{year_clause} "
             "RETURN c.corpusid, c.title, c.year, c.citationcount"),
            (want_citers,
             f"MATCH (c:Paper)-[:Cites]->(s:Paper) WHERE s.corpusid = {sid} "
             f"AND c.citationcount >= $min_cit{year_clause} "
             "RETURN c.corpusid, c.title, c.year, c.citationcount"),
        ):
            if not want:
                continue
            for row in rows(con, query, params):
                nbr = int(row[0])
                if nbr in seen_nbr:
                    continue
                seen_nbr.add(nbr)
                out.append(row)
    return out
