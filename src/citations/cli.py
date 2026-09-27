"""Thin CLI over the citation query core (resolve / neighbors / walk / rerank).

Resolution hits ``paper_meta.sqlite``; neighbours/walk/rerank run on KuzuDB;
abstracts attach from ``abstracts.sqlite`` (``--abstracts``) with optional S2 API
back-fill. ``--json`` emits structured stdout (informational lines → stderr). See
``--help`` for per-command flags.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any

import kuzu

from src.citations import DEFAULT_BASE, DEFAULT_RELEASE, s2_client
from src.citations import abstracts as abstracts_mod
from src.citations import corpus as corpus_mod
from src.citations.neighbors import neighbors
from src.citations.quality import (
    QualityScore,
    Weights,
    age_bucket,
    cohort_tier,
    paper_age_years,
    score_quality,
)
from src.citations.scoring import score_candidates
from src.citations.seed_resolver import SeedUnresolvableError, resolve
from src.citations.walker import hop1_counts, walk, walk_graph


def _meta_path(args: argparse.Namespace) -> str:
    if args.meta_path:
        return str(args.meta_path)
    return f"{args.base}/{args.release}/paper_meta.sqlite"


def _kuzu_path(args: argparse.Namespace) -> str:
    if args.kuzu_path:
        return str(args.kuzu_path)
    return f"{args.base}/{args.release}/citations.kuz"


def _abstracts_path(args: argparse.Namespace) -> str:
    if args.abstracts_path:
        return str(args.abstracts_path)
    return f"{args.base}/{args.release}/abstracts.sqlite"


def _meta_con(args: argparse.Namespace) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{_meta_path(args)}?mode=ro", uri=True)


def _kuzu_con(args: argparse.Namespace) -> kuzu.Connection:
    db = kuzu.Database(
        _kuzu_path(args),
        buffer_pool_size=int(args.buffer_pool_gb * 1024**3),
        read_only=True,
    )
    return kuzu.Connection(db, num_threads=args.num_threads)


def _abstract_map(args: argparse.Namespace, ids: list[int]) -> dict[int, str]:
    """Local ``abstracts.sqlite`` lookup, then optional S2 API back-fill.

    Empty when ``--abstracts`` is unset. Back-fill targets the misses, which skew
    toward the niche papers the bulk dump omits.
    """
    if not getattr(args, "abstracts", False) or not ids:
        return {}
    abs_con = sqlite3.connect(f"file:{_abstracts_path(args)}?mode=ro", uri=True)
    try:
        found = abstracts_mod.fetch(abs_con, ids)
    finally:
        abs_con.close()
    if getattr(args, "abstracts_fallback", False):
        missing = [i for i in ids if i not in found]
        if missing:
            found.update(s2_client.fetch_abstracts(missing))
    return found


def _dump_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _scope_marker(in_corpus: bool | None) -> str:
    """``✓`` already in RAG corpus · ``+`` new (not ingested) · ``?`` unknown."""
    if in_corpus is None:
        return "?"
    return "✓" if in_corpus else "+"


def _abstract_block(text: str | None) -> str:
    if not text:
        return ""
    return f"\n      {text}"


def _snippet(text: str, width: int = 108) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


def _emit_walk_graph(
    seed_id: int,
    seed_title: str,
    node_dicts: list[dict[str, Any]],
    edges: list[tuple[int, int]],
) -> None:
    """Dump the walk as a path-DAG: seed + walked candidates + directed edges."""
    _dump_json(
        {
            "seed": {"corpus_id": seed_id, "title": seed_title},
            "nodes": node_dicts,
            "edges": [{"src": a, "dst": b} for a, b in edges],
        }
    )


def _emit_walk_tree(
    seed_id: int,
    seed_title: str,
    node_objs: list[Any],
    edges: list[tuple[int, int]],
) -> None:
    """Dump the walk as a seed-rooted nested tree (children embedded per node).

    A spanning tree over the path-DAG: each node is attached under one
    lower-hop parent so an LLM reads paths top-down without joining an edge list.
    Cross-links (alternate parents, same-hop edges) are dropped — use --graph for
    the complete edge set.
    """
    by_id = {o.corpus_id: o for o in node_objs}
    hop = {seed_id: 0}
    for o in node_objs:
        hop[o.corpus_id] = o.shortest_hop

    children: dict[int, list[int]] = {}
    placed: set[int] = set()
    for a, b in edges:
        ha, hb = hop.get(a), hop.get(b)
        if ha is None or hb is None or ha == hb:
            continue
        parent, child = (a, b) if ha < hb else (b, a)
        if child in placed:
            continue
        children.setdefault(parent, []).append(child)
        placed.add(child)

    def build(cid: int) -> dict[str, Any]:
        node = (
            {"corpus_id": seed_id, "title": seed_title}
            if cid == seed_id
            else asdict(by_id[cid])
        )
        kids = [build(c) for c in children.get(cid, [])]
        if kids:
            node["children"] = kids
        return node

    _dump_json(build(seed_id))


def _mm_label(text: str | None) -> str:
    # Mermaid quoted labels break on embedded double quotes; collapse + escape.
    return " ".join((text or "").split()).replace('"', "'")[:48]


def _emit_walk_mermaid(
    seed_id: int,
    seed_title: str,
    node_objs: list[Any],
    edges: list[tuple[int, int]],
) -> None:
    """Print a Mermaid ``graph TD`` of the walked DAG (seed + candidates + edges)."""
    lines = ["graph TD", f'  N{seed_id}["{_mm_label(seed_title)}"]']
    for o in node_objs:
        niche = getattr(o, "niche", None)
        tag = f" (n{niche:.1f})" if niche is not None else ""
        lines.append(f'  N{o.corpus_id}["{_mm_label(o.title)}{tag}"]')
    lines.extend(f"  N{a} --> N{b}" for a, b in edges)
    print("\n".join(lines))


def cmd_resolve(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    try:
        result = resolve(meta, args.seed)
    except SeedUnresolvableError as e:
        print(f"unresolvable: {e}", file=sys.stderr)
        return 2
    if args.json:
        _dump_json(asdict(result))
        return 0
    print(
        f"corpus_id={result.corpus_id}  conf={result.confidence:.2f}  src={result.source}\n"
        f"  title:  {result.title!r}\n"
        f"  arxiv:  {result.arxiv_id}\n"
        f"  doi:    {result.doi}"
    )
    return 0


def _locators(corpus_id: int, arxiv_id: str | None, doi: str | None) -> dict[str, str]:
    loc = {"s2": f"https://www.semanticscholar.org/paper/{corpus_id}"}
    if arxiv_id:
        loc["arxiv"] = f"https://arxiv.org/abs/{arxiv_id}"
    if doi:
        loc["doi"] = f"https://doi.org/{doi}"
    return loc


def cmd_show(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    resolved = []
    for s in args.seed:
        try:
            resolved.append(resolve(meta, s))
        except SeedUnresolvableError as e:
            print(f"unresolvable: {s}: {e}", file=sys.stderr)
    if not resolved:
        return 2
    cids = [r.corpus_id for r in resolved]

    abs_con = sqlite3.connect(f"file:{_abstracts_path(args)}?mode=ro", uri=True)
    try:
        abs_map = abstracts_mod.fetch(abs_con, cids)
    finally:
        abs_con.close()
    if args.abstracts_fallback:
        missing = [c for c in cids if c not in abs_map]
        if missing:
            abs_map.update(s2_client.fetch_abstracts(missing))

    records: list[dict[str, object]] = []
    for seed in resolved:
        row = meta.execute(
            "SELECT title, arxiv_id, doi, year, citationcount "
            "FROM paper_meta WHERE corpusid = ?",
            [seed.corpus_id],
        ).fetchone()
        title, arxiv_id, doi, year, cit = (
            row if row else (seed.title, seed.arxiv_id, seed.doi, None, None)
        )
        records.append(
            {
                "corpus_id": seed.corpus_id, "title": title, "year": year,
                "citationcount": cit, "arxiv_id": arxiv_id, "doi": doi,
                "urls": _locators(seed.corpus_id, arxiv_id, doi),
                "abstract": abs_map.get(seed.corpus_id),
            }
        )

    if args.json:
        _dump_json(records if len(records) > 1 else records[0])
        return 0
    for i, rec in enumerate(records):
        if i:
            print("\n" + "─" * 80)
        print(f"{rec['corpus_id']}  {rec['title']}")
        print(f"  year {rec['year']} · cit {rec['citationcount']}")
        urls = rec["urls"]
        assert isinstance(urls, dict)
        for label in ("arxiv", "doi", "s2"):
            if label in urls:
                print(f"  {label:<6} {urls[label]}")
        print()
        print(rec["abstract"] or "<no abstract stored>")
    return 0


def _quality_path(args: argparse.Namespace) -> str:
    return args.quality_path or f"{args.base}/{args.release}/paper_quality.sqlite"


def _authority_path(args: argparse.Namespace) -> str:
    return args.authority_path or f"{args.base}/{args.release}/author_authority.sqlite"


def _score_one(
    qcon: sqlite3.Connection,
    acon: sqlite3.Connection,
    snapshot: date,
    weights: Weights,
    corpus_id: int,
) -> QualityScore | None:
    """Compute one paper's reliability from the stores, or None if not present."""
    row = qcon.execute(
        "SELECT infl_cites, ref_count, pub_types, venue_id, is_oa, field, year, "
        "pub_date, author_ids FROM paper_quality WHERE corpusid = ?",
        [corpus_id],
    ).fetchone()
    if row is None:
        return None
    infl, refs, ptypes_j, venue_id, _oa, field, year, pub_date, aids_j = row
    ptypes = json.loads(ptypes_j) if ptypes_j else []
    aids = json.loads(aids_j) if aids_j else []
    age_y, date_known = paper_age_years(pub_date, year, snapshot)
    tier = cohort_tier(ptypes, venue_id is not None)

    b = qcon.execute(
        "SELECT mean_infl FROM fwci_baseline WHERE field = ? AND pub_tier = ? "
        "AND age_bucket = ?",
        [field or "NA", tier, age_bucket(age_y)],
    ).fetchone()
    expected = float(b[0]) if b else None

    venue_prestige = 0.0
    if venue_id:
        vp = qcon.execute(
            "SELECT mean_infl FROM venue_prestige WHERE venue_id = ?", [venue_id]
        ).fetchone()
        venue_prestige = float(vp[0]) if vp else 0.0

    hidx: list[int] = []
    if aids:
        ph = ",".join("?" * len(aids))
        hidx = [
            int(r[0]) for r in acon.execute(
                f"SELECT hindex FROM author_authority WHERE authorid IN ({ph})", aids
            )
        ]

    return score_quality(
        influential_cites=infl, reference_count=refs, publication_types=ptypes,
        has_venue=venue_id is not None, venue_prestige=venue_prestige,
        expected_infl_cites=expected, author_hindexes=hidx,
        age_years=age_y, date_known=date_known, weights=weights,
    )


def _quality_cons(args: argparse.Namespace) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    return (
        sqlite3.connect(f"file:{_quality_path(args)}?mode=ro", uri=True),
        sqlite3.connect(f"file:{_authority_path(args)}?mode=ro", uri=True),
    )


def _reliability_map(args: argparse.Namespace, corpus_ids: list[int]) -> dict[int, float]:
    """Map corpus_id → reliability composite, annotating results by default.

    Fail-soft: returns {} if disabled via --no-reliability or the quality store
    isn't present, so walk/neighbors still work without the quality build.
    """
    if getattr(args, "no_reliability", False) or not corpus_ids:
        return {}
    if not Path(_quality_path(args)).exists():
        return {}
    qcon, acon = _quality_cons(args)
    try:
        snapshot = date.fromisoformat(args.release)
        out: dict[int, float] = {}
        for cid in corpus_ids:
            q = _score_one(qcon, acon, snapshot, Weights(), cid)
            if q is not None:
                out[cid] = q.composite
        return out
    finally:
        qcon.close()
        acon.close()


def _corpus_db_path(args: argparse.Namespace) -> str | None:
    if args.corpus_db:
        return str(args.corpus_db)
    default = Path(__file__).resolve().parents[2] / "parents" / "parents.sqlite"
    return str(default) if default.exists() else None


def _in_corpus_map(args: argparse.Namespace, corpus_ids: list[int]) -> dict[int, bool | None]:
    """Map corpus_id → in-RAG-corpus (True/False/None-unknown).

    Fail-soft: returns {} (leaves in_corpus at its None default) when no corpus
    db is resolvable, so walk/neighbors still work without the RAG store.
    """
    if not corpus_ids:
        return {}
    path = _corpus_db_path(args)
    if not path:
        return {}
    ingested = corpus_mod.load_ingested_arxiv(path)
    meta = _meta_con(args)
    try:
        return corpus_mod.membership(meta, corpus_ids, ingested)
    finally:
        meta.close()


def cmd_quality(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    resolved = []
    for s in args.seed:
        try:
            resolved.append(resolve(meta, s))
        except SeedUnresolvableError as e:
            print(f"unresolvable: {s}: {e}", file=sys.stderr)
    if not resolved:
        return 2

    qcon, acon = _quality_cons(args)
    snapshot = date.fromisoformat(args.release)
    weights = Weights(
        venue=args.w_venue, impact=args.w_impact,
        author=args.w_author, grounding=args.w_grounding,
    )

    records: list[dict[str, Any]] = []
    for seed in resolved:
        q = _score_one(qcon, acon, snapshot, weights, seed.corpus_id)
        if q is None:
            print(f"no quality data: {seed.corpus_id}", file=sys.stderr)
            continue
        records.append({"corpus_id": seed.corpus_id, "title": seed.title, **asdict(q)})

    qcon.close()
    acon.close()

    if args.json:
        _dump_json(records if len(records) != 1 else records[0])
        return 0
    for i, r in enumerate(records):
        if i:
            print("\n" + "─" * 80)
        print(f"{r['corpus_id']}  {str(r['title'])[:72]}")
        print(f"  reliability {r['composite']}/100  ·  {r['verdict']}")
        fwci = r["impact_fwci"]
        flag = "  [low conf]" if r["impact_confidence"] == "low" else ""
        fwci_s = f"fwci={fwci}" if fwci is not None else "fwci=n/a"
        print(f"  venue     {r['venue']:.2f}  {r['venue_label']} (prestige {r['venue_prestige']})")
        print(f"  impact    {r['impact']:.2f}  {fwci_s} ({r['influential_cites']} infl-cites, "
              f"age {r['age_years']}y){flag}")
        print(f"  author    {r['author']:.2f}  max h={r['max_hindex']} mean={r['mean_hindex']} "
              f"(n={r['n_authors']})")
        print(f"  grounding {r['grounding']:.2f}  {r['reference_count']} refs")
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    try:
        seed = resolve(meta, args.seed)
    except SeedUnresolvableError as e:
        print(f"unresolvable seed: {e}", file=sys.stderr)
        return 2

    con = _kuzu_con(args)
    refs, citers = hop1_counts(con, [seed.corpus_id], min_citationcount=args.min_cite)

    by_hop: dict[int, int] = {}
    if args.max_hop > 1:
        cands = walk(
            con, [seed.corpus_id],
            max_hop=args.max_hop, budget=10_000_000,
            min_citationcount=args.min_cite, per_hop_beam=args.per_hop_beam,
            pass_through_hubs=args.pass_through_hubs,
        )
        for c in cands:
            by_hop[c.shortest_hop] = by_hop.get(c.shortest_hop, 0) + 1

    payload = {
        "corpus_id": seed.corpus_id, "title": seed.title, "min_cite": args.min_cite,
        "references": refs, "citers": citers, "hop1_total": refs + citers,
        "walk_by_hop": {str(k): by_hop[k] for k in sorted(by_hop)},
        "walk_total": sum(by_hop.values()),
    }
    if args.json:
        _dump_json(payload)
        return 0
    print(f"{seed.corpus_id}  {seed.title[:72]}")
    print(f"  hop-1 (min_cite={args.min_cite}):  references={refs}  citers={citers}")
    if by_hop:
        layers = "  ".join(f"hop{k}={by_hop[k]}" for k in sorted(by_hop))
        beam = args.per_hop_beam if args.per_hop_beam else "none"
        print(f"  walk (beam={beam}):  {layers}  total={sum(by_hop.values())}")
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    try:
        seed = resolve(meta, args.seed)
    except SeedUnresolvableError as e:
        print(f"unresolvable seed: {e}", file=sys.stderr)
        return 2
    print(f"seed: corpus_id={seed.corpus_id}  {seed.title[:80]!r}", file=sys.stderr)

    anchors: list[int] | None = None
    if args.relative_to:
        try:
            anchor = resolve(meta, args.relative_to)
        except SeedUnresolvableError as e:
            print(f"unresolvable --relative-to: {e}", file=sys.stderr)
            return 2
        anchors = [anchor.corpus_id]

    con = _kuzu_con(args)
    nset = neighbors(
        con,
        [seed.corpus_id],
        direction=args.direction,
        limit=args.limit,
        min_citationcount=args.min_cite,
        rerank=args.rerank,
        anchors=anchors,
        norm=args.norm,
        niche_bias=args.niche_bias,
        pool=args.pool,
    )
    all_ids = [n.corpus_id for n in nset.references] + [n.corpus_id for n in nset.citers]
    amap = _abstract_map(args, all_ids)
    rmap = _reliability_map(args, all_ids)
    cmap = _in_corpus_map(args, all_ids)
    refs = [
        replace(n, abstract=amap.get(n.corpus_id), reliability=rmap.get(n.corpus_id),
                in_corpus=cmap.get(n.corpus_id))
        for n in nset.references
    ]
    citers = [
        replace(n, abstract=amap.get(n.corpus_id), reliability=rmap.get(n.corpus_id),
                in_corpus=cmap.get(n.corpus_id))
        for n in nset.citers
    ]

    if args.json:
        _dump_json(
            {
                "seed": asdict(seed),
                "references": [asdict(n) for n in refs],
                "citers": [asdict(n) for n in citers],
            }
        )
        return 0

    print(f"seed {seed.corpus_id}  {(seed.title or '')[:72]}")
    idx = 1

    def _print_section(label: str, items: list[Any], start: int) -> int:
        print(f"\n{label} ({len(items)})")
        for i, n in enumerate(items, start=start):
            niche_s = f"n{n.niche:.2f} " if n.niche is not None else ""
            rel_s = f"r{int(n.reliability)} " if n.reliability is not None else ""
            meta = f"{niche_s}{rel_s}c{n.citationcount or 0} {n.year}"
            print(f"{i:>2}. [{meta}] {n.corpus_id}  {(n.title or '<no title>')[:72]}")
            if n.abstract:
                print(f"    {_snippet(n.abstract)}")
        return start + len(items)

    if args.direction in ("references", "both"):
        idx = _print_section("REFERENCES · seed cites", refs, idx)
    if args.direction in ("citers", "both"):
        _print_section("CITERS · cite seed", citers, idx)
    return 0


def _score_walk(
    con: kuzu.Connection, seed_id: int, candidates: list[Any], args: argparse.Namespace
) -> list[Any]:
    return score_candidates(
        con, [seed_id], [c.corpus_id for c in candidates],
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta,
        norm=args.norm, niche_bias=args.niche_bias,
        shortest_hop_by_id={c.corpus_id: c.shortest_hop for c in candidates},
        hits_by_id={c.corpus_id: c.hits for c in candidates},
        recency_tau=args.recency_tau, recency_floor=args.recency_floor,
    )


def cmd_walk(args: argparse.Namespace) -> int:
    meta = _meta_con(args)
    try:
        seed = resolve(meta, args.seed)
    except SeedUnresolvableError as e:
        print(f"unresolvable seed: {e}", file=sys.stderr)
        return 2
    print(f"seed: corpus_id={seed.corpus_id}  {seed.title[:80]!r}", file=sys.stderr)

    con = _kuzu_con(args)

    # Graph/tree mode returns the full walked path-DAG (all candidates + BFS
    # edges), not a top-N slice — the bridge nodes that connect seed→…→gem stay.
    if args.graph or args.tree or args.mermaid:
        cands, edges = walk_graph(
            con, [seed.corpus_id],
            max_hop=args.max_hop, budget=args.budget,
            min_citationcount=args.min_cite, per_hop_beam=args.per_hop_beam,
            pass_through_hubs=args.pass_through_hubs,
        )
        node_objs: list[Any] = (
            _score_walk(con, seed.corpus_id, cands, args) if args.rerank else cands
        )
        ids = [o.corpus_id for o in node_objs]
        amap = _abstract_map(args, ids)
        rmap = _reliability_map(args, ids)
        cmap = _in_corpus_map(args, ids)
        node_objs = [
            replace(o, abstract=amap.get(o.corpus_id), reliability=rmap.get(o.corpus_id),
                    in_corpus=cmap.get(o.corpus_id))
            for o in node_objs
        ]
        if args.mermaid:
            _emit_walk_mermaid(seed.corpus_id, seed.title, node_objs, edges)
        elif args.tree:
            _emit_walk_tree(seed.corpus_id, seed.title, node_objs, edges)
        else:
            _emit_walk_graph(seed.corpus_id, seed.title, [asdict(o) for o in node_objs], edges)
        return 0

    candidates = walk(
        con,
        [seed.corpus_id],
        max_hop=args.max_hop,
        budget=args.budget,
        min_citationcount=args.min_cite,
        per_hop_beam=args.per_hop_beam,
        pass_through_hubs=args.pass_through_hubs,
    )
    print(f"candidates: {len(candidates)}", file=sys.stderr)

    if not args.rerank:
        shown = candidates[: args.top]
        ids = [c.corpus_id for c in shown]
        amap = _abstract_map(args, ids)
        rmap = _reliability_map(args, ids)
        cmap = _in_corpus_map(args, ids)
        shown = [
            replace(c, abstract=amap.get(c.corpus_id), reliability=rmap.get(c.corpus_id),
                    in_corpus=cmap.get(c.corpus_id))
            for c in shown
        ]
        if args.json:
            _dump_json([asdict(c) for c in shown])
            return 0
        for c in shown:
            scope = _scope_marker(c.in_corpus)
            title = (c.title or "<no title>")[:80]
            rel = f"rel={int(c.reliability):>3} " if c.reliability is not None else ""
            print(
                f"  {scope} {rel}hop={c.shortest_hop} hits={c.hits:>4} "
                f"cit={c.citationcount or 0:>6} y={c.year}  {title}"
                f"{_abstract_block(c.abstract)}"
            )
        return 0

    scored = _score_walk(con, seed.corpus_id, candidates, args)
    print(f"reranked: {len(scored)}", file=sys.stderr)
    top_scored = scored[: args.top]
    ids = [s.corpus_id for s in top_scored]
    amap = _abstract_map(args, ids)
    rmap = _reliability_map(args, ids)
    cmap = _in_corpus_map(args, ids)
    top_scored = [
        replace(s, abstract=amap.get(s.corpus_id), reliability=rmap.get(s.corpus_id),
                in_corpus=cmap.get(s.corpus_id))
        for s in top_scored
    ]
    if args.json:
        _dump_json([asdict(s) for s in top_scored])
        return 0
    for s in top_scored:
        scope = _scope_marker(s.in_corpus)
        title = (s.title or "<no title>")[:60]
        rel = f"rel={int(s.reliability):>3} " if s.reliability is not None else ""
        print(
            f"  {scope} niche={s.niche:7.3f} {rel}hop={s.shortest_hop} "
            f"dir={s.direct:3.0f} co={s.cocite:5.2f} bc={s.bibcouple:5.2f} "
            f"r={s.reach:4.1f} age={s.recency:.2f} "
            f"cit={s.citationcount or 0:>5} y={s.year}  {title}"
            f"{_abstract_block(s.abstract)}"
        )
    return 0


def _add_abstract_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--abstracts", action="store_true",
        help="attach abstracts from local abstracts.sqlite (partial coverage)",
    )
    p.add_argument(
        "--abstracts-fallback", action="store_true",
        help="back-fill abstracts missing locally via the S2 Graph API (network)",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--kuzu-path", type=Path, default=None, help="path to citations.kuz")
    ap.add_argument("--meta-path", type=Path, default=None, help="path to paper_meta.sqlite")
    ap.add_argument("--abstracts-path", type=Path, default=None, help="path to abstracts.sqlite")
    ap.add_argument("--quality-path", type=Path, default=None, help="path to paper_quality.sqlite")
    ap.add_argument("--authority-path", type=Path, default=None,
                    help="path to author_authority.sqlite")
    ap.add_argument("--corpus-db", type=Path, default=None,
                    help="path to RAG parents.sqlite for real in_corpus membership; "
                         "defaults to <repo>/parents/parents.sqlite when present")
    ap.add_argument(
        "--buffer-pool-gb", type=float, default=4.0,
        help="Kuzu buffer pool size in GB for walk/rerank queries (default 4). "
             "Walk/neighbours use indexed per-seed expansion, so the working set "
             "is bounded — 4G plateaus latency and leaves headroom next to the "
             "resident embedding process on a shared box.",
    )
    ap.add_argument(
        "--num-threads", type=int, default=12,
        help="Kuzu query threads (default 12)",
    )
    ap.add_argument("--json", action="store_true", help="emit structured JSON on stdout")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_resolve = sub.add_parser("resolve", help="resolve a seed to corpus_id")
    p_resolve.add_argument("seed")
    p_resolve.set_defaults(fn=cmd_resolve)

    p_count = sub.add_parser(
        "count", help="neighbour/frontier counts for a seed (size a walk before running it)"
    )
    p_count.add_argument("seed")
    p_count.add_argument(
        "--min-cite", type=int, default=0,
        help="count only neighbours with citationcount >= this (default 0)",
    )
    p_count.add_argument(
        "--max-hop", type=int, default=1,
        help="if >1, also report walked frontier size per hop (default 1)",
    )
    p_count.add_argument(
        "--per-hop-beam", type=int, default=None,
        help="beam used for the multi-hop frontier count (mirror your walk)",
    )
    p_count.add_argument("--pass-through-hubs", action="store_true")
    p_count.set_defaults(fn=cmd_count)

    p_quality = sub.add_parser(
        "quality", help="reliability score + signal breakdown for one or more papers"
    )
    p_quality.add_argument("seed", nargs="+")
    p_quality.add_argument("--w-venue", type=float, default=0.30)
    p_quality.add_argument("--w-impact", type=float, default=0.35)
    p_quality.add_argument("--w-author", type=float, default=0.20)
    p_quality.add_argument("--w-grounding", type=float, default=0.15)
    p_quality.set_defaults(fn=cmd_quality)

    p_show = sub.add_parser("show", help="full abstract + locators for one or more papers")
    p_show.add_argument("seed", nargs="+", help="one or more seeds (id/arxiv/doi/url/title)")
    p_show.add_argument(
        "--abstracts-fallback", action="store_true",
        help="fetch the abstract from the S2 Graph API if not stored locally",
    )
    p_show.set_defaults(fn=cmd_show)

    p_neighbors = sub.add_parser("neighbors", help="hop-1 references + citers of a seed")
    p_neighbors.add_argument("seed")
    p_neighbors.add_argument(
        "--direction", choices=("references", "citers", "both"), default="both",
        help="references = seed cites; citers = cite seed (default both)",
    )
    p_neighbors.add_argument("--limit", type=int, default=50, help="max per direction")
    p_neighbors.add_argument(
        "--min-cite", type=int, default=0,
        help="drop neighbours with citationcount below this (default: 0)",
    )
    p_neighbors.add_argument(
        "--rerank", action="store_true",
        help="rank each side by niche score vs the node instead of citation count",
    )
    p_neighbors.add_argument(
        "--relative-to", default=None,
        help="anchor the niche score to this seed (keeps a multi-step walk on-topic)",
    )
    p_neighbors.add_argument(
        "--norm", choices=("none", "logidf", "assoc", "salton"), default="assoc",
        help="overlap popularity-normalisation for --rerank (default assoc)",
    )
    p_neighbors.add_argument(
        "--niche-bias", type=float, default=0.0,
        help="obscurity preference b in [0,1] for --rerank (default 0)",
    )
    p_neighbors.add_argument(
        "--pool", type=int, default=2000,
        help="candidate pool size per side before --rerank (citation-ASC; default 2000)",
    )
    p_neighbors.add_argument(
        "--no-reliability", action="store_true",
        help="skip the reliability annotation (on by default when the store exists)",
    )
    _add_abstract_flags(p_neighbors)
    p_neighbors.set_defaults(fn=cmd_neighbors)

    p_walk = sub.add_parser("walk", help="walk citation graph from seed")
    p_walk.add_argument("seed")
    p_walk.add_argument("--max-hop", type=int, default=2)
    p_walk.add_argument("--budget", type=int, default=500)
    p_walk.add_argument("--top", type=int, default=20)
    p_walk.add_argument(
        "--per-hop-beam", type=int, default=1500,
        help="retain only top-K candidates per hop by degree-normalised reach "
             "(default 1500). Bounds the frontier — with indexed per-seed "
             "expansion an unbounded frontier on a hub seed materialises "
             "millions of rows. Lower it (e.g. 400) for fast deep hub walks.",
    )
    p_walk.add_argument(
        "--min-cite", type=int, default=0,
        help="drop frontier candidates with citationcount below this (default: 0)",
    )
    p_walk.add_argument(
        "--pass-through-hubs", action="store_true",
        help="walk through hub papers but exclude them from the candidate set",
    )
    p_walk.add_argument(
        "--rerank", action="store_true",
        help="apply Layer-2 multi-signal rerank (direct + co-cit + bibcouple)",
    )
    p_walk.add_argument(
        "--graph", action="store_true",
        help="emit JSON path-DAG {seed, nodes, edges} of the walked subgraph so an "
             "LLM can reconstruct multi-hop paths",
    )
    p_walk.add_argument(
        "--tree", action="store_true",
        help="emit a seed-rooted nested JSON tree (children embedded per node); "
             "easier for an LLM than the flat --graph edge list",
    )
    p_walk.add_argument(
        "--mermaid", action="store_true",
        help="emit a Mermaid 'graph TD' of the walked DAG (use a small "
             "--per-hop-beam to keep it legible)",
    )
    p_walk.add_argument(
        "--no-reliability", action="store_true",
        help="skip the reliability annotation (on by default when the store exists)",
    )
    p_walk.add_argument("--alpha", type=float, default=1.0, help="direct-citation weight")
    p_walk.add_argument("--beta", type=float, default=1.0, help="co-citation weight")
    p_walk.add_argument("--gamma", type=float, default=1.0, help="bibcouple weight")
    p_walk.add_argument(
        "--delta", type=float, default=0.3,
        help="walker-reach weight (hits / shortest_hop)",
    )
    p_walk.add_argument(
        "--norm", choices=("none", "logidf", "assoc", "salton"), default="assoc",
        help="overlap popularity-normalisation: logidf damps the shared bridge "
             "node; assoc/salton damp by candidate citation count (default assoc)",
    )
    p_walk.add_argument(
        "--niche-bias", type=float, default=0.0,
        help="obscurity preference b in [0,1]: score/(1+cit)^b. 0=relevance only, "
             "1=float relevant-but-obscure papers to the top (default 0)",
    )
    p_walk.add_argument(
        "--recency-tau", type=float, default=5.0,
        help="time constant (yrs) for exp recency decay; 0 disables",
    )
    p_walk.add_argument(
        "--recency-floor", type=float, default=0.35,
        help="minimum recency multiplier (keeps classics from total decay)",
    )
    _add_abstract_flags(p_walk)
    p_walk.set_defaults(fn=cmd_walk)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
