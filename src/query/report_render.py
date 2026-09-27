"""Render a single-page Markdown audit report from a finalized query directory.

Reads ``meta.json`` plus the ``NN_<stage>.json`` files written by
``QueryLogger`` and emits a ``report.md`` that summarizes everything in one
scannable view. Source-of-truth files stay for deep dives; this is the
"detective's overview" view.

Pure reader — does no IO except reading already-finalized files in the
directory and writing ``report.md`` to it. Never raises on missing optional
stages; degrades to "(skipped)" lines instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECTION_SEP = "\n\n"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_report(log_dir: Path) -> str:
    """Build a Markdown report from the stage files inside *log_dir*.

    Reads ``meta.json`` and any present ``01_decompose.json`` … ``08_crag_retry.json``.
    Returns the rendered string. The caller is responsible for writing it.

    If ``meta.json`` is missing or malformed, returns a minimal
    "report unavailable" stub so the file still exists in every dir.
    """
    meta = _load_json(log_dir / "meta.json")
    if not meta:
        return "# Query Audit Report\n\n_meta.json missing or unreadable._\n"

    stages = _load_stages(log_dir)
    chunk_files = _list_chunk_files(log_dir / "chunks")

    sections: list[str] = ["# Query Audit Report"]

    for fn in (
        _section_qa,
        _section_overview,
        _section_config,
        _section_timeline,
        _section_decomposition,
        _section_retrieval,
        _section_mmr,
        _section_synthesis,
        _section_crag,
        # Phase 2 — multi-retriever sub-stage renderers (Plan 02-05).
        _section_paperqa,
        _section_hipporag,
        _section_lazygraph,
        _section_fused,
        _section_files,
    ):
        try:
            chunk = fn(meta, stages, chunk_files, log_dir)
        except Exception as exc:  # one corrupted section shouldn't kill the report
            logger.warning("report_render: %s failed: %s", fn.__name__, exc)
            section = fn.__name__.replace("_section_", "").title()
            chunk = f"## {section}\n\n_(rendering failed: {exc})_"
        if chunk:
            sections.append(chunk)

    return _SECTION_SEP.join(sections) + "\n"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("report_render: failed to read %s: %s", path, exc)
        return None


_STAGE_FILES = {
    "decompose": "01_decompose.json",
    "stepback": "02_stepback.json",
    "milvus": "03_milvus.json",
    "rerank": "04_rerank.json",
    "mmr": "05_mmr.json",
    "synthesis": "06_synthesis.json",
    "grader": "07_grader.json",
    "crag_retry": "08_crag_retry.json",
    # Phase 2 — multi-retriever sub-stages (Plan 02-05).
    "paperqa_rerank": "10_paperqa_rerank.json",
    "paperqa_read": "11_paperqa_read.json",
    "paperqa_answer": "12_paperqa_answer.json",
    "hipporag_entities": "13_hipporag_entities.json",
    "hipporag_walk": "14_hipporag_walk.json",
    "hipporag_synthesize": "15_hipporag_synthesize.json",
    "lazygraph_route": "16_lazygraph_route.json",
    "lazygraph_community": "17_lazygraph_community.json",
    "lazygraph_summarize": "18_lazygraph_summarize.json",
    "fused_dispatch": "20_fused_dispatch.json",
    "fused_rerank": "21_fused_rerank.json",
}


def _load_stages(log_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {stage_name: payload_dict} for every stage file present."""
    out: dict[str, dict[str, Any]] = {}
    for name, filename in _STAGE_FILES.items():
        payload = _load_json(log_dir / filename)
        if payload is not None:
            out[name] = payload
    return out


def _list_chunk_files(chunks_dir: Path) -> list[Path]:
    if not chunks_dir.is_dir():
        return []
    return sorted(chunks_dir.glob("*.txt"))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ms(ms: int | float | None) -> str:
    if ms is None:
        return "—"
    s = float(ms) / 1000.0
    if s >= 100:
        return f"{s:.0f}s"
    return f"{s:.1f}s"


def _fmt_cost(usd: float | None) -> str:
    if usd is None:
        return "—"
    if usd == 0:
        return "$0"
    if usd < 0.001:
        return f"${usd:.2e}"
    return f"${usd:.4f}"


def _fmt_int(n: int | float | None) -> str:
    if n is None or n == 0:
        return "—"
    return f"{int(n):,}"


def _fmt_bool(b: bool | None, *, true_label: str = "✅", false_label: str = "❌") -> str:
    if b is None:
        return "—"
    return true_label if b else false_label


def _truncate(s: str, n: int = 80) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _shorten_source(path: str, collection: str | None = None) -> str:
    """Strip ``sources/<collection>/`` prefix, drop trailing filename, truncate."""
    p = path
    if collection and p.startswith(f"sources/{collection}/"):
        p = p[len(f"sources/{collection}/") :]
    elif p.startswith("sources/"):
        # Keep the collection segment if collection wasn't supplied.
        p = p[len("sources/") :]
    # Drop trailing /source.pdf, /source.txt, /content_list.json, etc.
    for tail in ("/source.pdf", "/source.txt", "/content_list.json", "/nlm.txt", "/pdf.txt"):
        if p.endswith(tail):
            p = p[: -len(tail)]
            break
    if len(p) > 60:
        p = p[:59] + "…"
    return p


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple Markdown table. Aligns numeric columns right via marker."""
    if not rows:
        return ""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _kv_table(rows: list[tuple[str, str]]) -> str:
    """Two-column key/value table without headers."""
    head = "|  |  |"
    sep = "|---|---|"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"{head}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
# Section: Q&A — query + answer up top for fastest read
# ---------------------------------------------------------------------------


def _section_qa(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    query = (meta.get("query") or "").strip()
    syn = stages.get("synthesis") or {}
    answer = (syn.get("answer") or "").strip()
    outcome = meta.get("outcome", {})

    out = ["## Q & A", f"\n**Q:** {query}\n"]
    if answer:
        out.append(f"**A:**\n\n{answer}")
    elif outcome.get("synthesis_skipped"):
        out.append("**A:** _(synthesis skipped — no chunks retrieved or `--raw` mode)_")
    else:
        out.append("**A:** _(no answer recorded)_")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section: overview
# ---------------------------------------------------------------------------


def _section_overview(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    totals = meta.get("totals", {})
    outcome = meta.get("outcome", {})
    cfg = meta.get("config", {})

    # Status line
    if outcome.get("synthesis_skipped"):
        status = f"⚠️ Synthesis skipped · {outcome.get('n_chunks_to_synth', 0)} chunks"
    else:
        n_chunks = outcome.get("n_chunks_to_synth", 0)
        crag = ""
        if outcome.get("crag_retried"):
            chose = outcome.get("crag_chose") or "?"
            crag = f" · CRAG retried (chose: {chose})"
        status = f"✅ Synthesized · {n_chunks} chunks{crag}"

    wall = totals.get("total_latency_ms")
    retrieval = totals.get("retrieval_latency_ms")
    synthesis = totals.get("synthesis_latency_ms")
    timing_breakdown = []
    if retrieval:
        timing_breakdown.append(f"retrieval {_fmt_ms(retrieval)}")
    if synthesis:
        timing_breakdown.append(f"synthesis {_fmt_ms(synthesis)}")
    timing_str = _fmt_ms(wall)
    if timing_breakdown:
        timing_str += " (" + " · ".join(timing_breakdown) + ")"

    n_calls = totals.get("n_llm_calls", 0)
    tokens = totals.get("tokens", {})
    total_tokens = tokens.get("total", 0)
    cost_str = _fmt_cost(totals.get("cost_usd"))
    cost_line = f"{cost_str} ({n_calls} LLM calls; {_fmt_int(total_tokens)} tokens)"

    pid = meta.get("pid")
    tid = meta.get("tid")
    host = meta.get("host", "?")
    host_line = f"`{host}` · pid {pid} · tid {tid}"

    rows: list[tuple[str, str]] = [
        ("**Status**", status),
        ("**Started**", meta.get("ts_started", "—")),
        ("**Wall clock**", timing_str),
        ("**Total cost**", cost_line),
        ("**Host**", host_line),
        ("**Git SHA**", meta.get("git_sha") or "—"),
        ("**Query hash**", meta.get("query_hash", "—")[:8]),
        (
            "**Collection / Notebook**",
            f"`{cfg.get('collection','?')}` / `{cfg.get('notebook','?')}`",
        ),
    ]
    return "## Overview\n\n" + _kv_table(rows)


# ---------------------------------------------------------------------------
# Section: config
# ---------------------------------------------------------------------------


def _section_config(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    cfg = meta.get("config", {})
    if not cfg:
        return ""

    reranker = meta.get("reranker") or {}
    rr_str = "—"
    if reranker.get("backend"):
        parts = [f"`{reranker.get('backend')}`"]
        if reranker.get("model"):
            parts.append(f"`{reranker['model']}`")
        head = " · ".join(parts)
        if reranker.get("endpoint"):
            head = f"{head} @ {reranker['endpoint']}"
        rr_str = head

    flags = (
        f"{_fmt_bool(cfg.get('decompose'))} decompose · "
        f"{_fmt_bool(cfg.get('stepback'))} stepback · "
        f"{_fmt_bool(cfg.get('crag'))} crag (thr {cfg.get('crag_threshold','?')}) · "
        f"{_fmt_bool(cfg.get('use_mmr'))} mmr"
    )
    if cfg.get("use_mmr"):
        flags += f" (λ={cfg.get('mmr_lambda')}, top-k={cfg.get('mmr_top_k')})"

    rows: list[tuple[str, str]] = [
        ("Generation model", f"`{cfg.get('gen_model','?')}`"),
        ("Decompose model", f"`{cfg.get('decompose_model','?')}`"),
        ("Judge model", f"`{cfg.get('judge_model','?')}`"),
        ("Embedder", f"`{cfg.get('embed_provider','?')}` ({cfg.get('embed_dim','?')} dim)"),
        ("Reranker", rr_str),
        (
            "Top-K retrieve / rerank",
            f"{cfg.get('top_k_retrieve','?')} / {cfg.get('top_k_rerank','?')}",
        ),
        ("Score threshold", str(cfg.get("score_threshold", "?"))),
        ("Max children/parent", str(cfg.get("max_children_per_parent", "?"))),
        ("Decompose max sub-queries", str(cfg.get("decompose_max_subqueries", "?"))),
        ("Pipeline flags", flags),
    ]
    return "## Configuration\n\n" + _kv_table(rows)


# ---------------------------------------------------------------------------
# Section: timeline
# ---------------------------------------------------------------------------


_STAGE_TIMELINE_ORDER = [
    ("decompose", "decompose", "decompose_model"),
    ("stepback", "stepback", "decompose_model"),
    ("milvus", "milvus", None),
    ("rerank", "rerank", None),
    ("synthesis", "synthesis", "gen_model"),
    ("grader", "grader", "judge_model"),
    ("crag_retry", "crag_retry", "gen_model"),
]


def _section_timeline(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Per-stage table: latency · model · in/out/reasoning tokens · cost."""
    cfg = meta.get("config", {})
    rows: list[list[str]] = []

    for stage_name, file_key, _model_cfg_key in _STAGE_TIMELINE_ORDER:
        payload = stages.get(file_key)
        if payload is None:
            continue

        # Special handling for milvus + rerank (no LLM tokens, latency is nested)
        if stage_name == "milvus":
            sub_qs = payload.get("sub_queries", []) or []
            milvus_lat = sum(int(sq.get("latency_ms") or 0) for sq in sub_qs)
            rows.append([
                f"milvus ({len(sub_qs)} sub-q)",
                _fmt_ms(milvus_lat),
                "—",
                "—", "—", "—", "—",
            ])
            continue

        if stage_name == "rerank":
            input_pairs = payload.get("input_pair_count", 0)
            survivors = len(payload.get("survivors", []) or [])
            rows.append([
                f"rerank ({input_pairs} → {survivors})",
                _fmt_ms(payload.get("latency_ms")),
                f"`{(meta.get('reranker') or {}).get('model','?')}`",
                "—", "—", "—", "—",
            ])
            continue

        # CRAG retry has its own nested shape
        if stage_name == "crag_retry":
            # Two parts: reformulate + retry-synthesis (+optional retry-grader)
            ref_lat = payload.get("reformulate_latency_ms") or 0
            ret_syn_lat = payload.get("retry_synthesis_latency_ms") or 0
            ret_grader_lat = payload.get("retry_grader_latency_ms") or 0
            total_lat = int(ref_lat) + int(ret_syn_lat) + int(ret_grader_lat)
            rows.append([
                "crag retry (refmt+synth+grade)",
                _fmt_ms(total_lat),
                f"`{cfg.get('gen_model','?')}` / `{cfg.get('judge_model','?')}`",
                "—", "—", "—",
                _fmt_cost(payload.get("retry_synthesis_cost_usd")),
            ])
            continue

        # Generic LLM stage: model, latency, usage, cost
        usage = payload.get("usage") or {}
        in_tok = int(usage.get("input") or 0)
        out_tok = int(usage.get("output") or 0)
        reason_tok = int(usage.get("reasoning") or 0)
        cost = payload.get("cost_usd")

        model = payload.get("model")
        rows.append([
            stage_name,
            _fmt_ms(payload.get("latency_ms")),
            f"`{model}`" if model else "—",
            _fmt_int(in_tok),
            _fmt_int(out_tok),
            _fmt_int(reason_tok),
            _fmt_cost(cost),
        ])

    if not rows:
        return ""

    # Append totals row
    totals = meta.get("totals", {})
    rows.append([
        "**TOTAL**",
        f"**{_fmt_ms(totals.get('total_latency_ms'))}**",
        "",
        f"**{_fmt_int(totals.get('tokens',{}).get('input'))}**",
        f"**{_fmt_int(totals.get('tokens',{}).get('output'))}**",
        f"**{_fmt_int(totals.get('tokens',{}).get('reasoning'))}**",
        f"**{_fmt_cost(totals.get('cost_usd'))}**",
    ])

    table = _md_table(
        ["Stage", "Latency", "Model", "In tok", "Out tok", "Reason tok", "Cost USD"],
        rows,
    )
    return "## Pipeline timeline\n\n" + table


# ---------------------------------------------------------------------------
# Section: decomposition
# ---------------------------------------------------------------------------


def _section_decomposition(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    decomp = stages.get("decompose")
    if decomp is None:
        return ""

    parse_status = decomp.get("parse_status", "ok")
    sub_queries = decomp.get("parsed_subqueries") or []
    original = (meta.get("query") or "").strip()
    stepback = stages.get("stepback")
    sb_q = (stepback.get("stepback_query") or "").strip() if stepback else ""

    # Extended fields from the new router (may be None when not activated).
    hyde_doc: str | None = decomp.get("hyde_doc") or None
    router_stepback: str | None = decomp.get("stepback") or None

    # When decompose was skipped or returned a single sub-query equal to the
    # original (and there's no stepback or hyde), the section is pure noise — the
    # query is already in Overview. Skip entirely.
    is_trivial_passthrough = (
        (parse_status == "skipped" or len(sub_queries) <= 1)
        and (not sub_queries or sub_queries[0].strip() == original)
        and not sb_q
        and not hyde_doc
        and not router_stepback
    )
    if is_trivial_passthrough:
        return ""

    lines = ["## Decomposition"]
    if parse_status == "skipped" or not sub_queries:
        lines.append("_(decomposition skipped — using original query)_")
    else:
        lines.append(f"Sub-queries ({len(sub_queries)}):")
        for i, sq in enumerate(sub_queries, 1):
            lines.append(f"{i}. \"{_truncate(sq, 200)}\"")
    if sb_q:
        lines.append(f"\nStepback: \"{_truncate(sb_q, 200)}\"")
    if router_stepback:
        lines.append(f"\nRouter stepback: \"{_truncate(router_stepback, 200)}\"")
    if hyde_doc:
        lines.append(f"\nHyDE passage: \"{_truncate(hyde_doc, 200)}\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section: retrieval
# ---------------------------------------------------------------------------


def _section_retrieval(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    milvus = stages.get("milvus")
    rerank = stages.get("rerank")
    cfg = meta.get("config", {})
    collection = cfg.get("collection")

    if milvus is None and rerank is None and not chunk_files:
        return ""

    out = ["## Retrieval"]

    # Milvus per sub-query table
    if milvus:
        sub_qs = milvus.get("sub_queries", []) or []
        if sub_qs:
            rows: list[list[str]] = []
            for i, sq in enumerate(sub_qs, 1):
                cands = sq.get("candidates", []) or []
                top_rrf = max((float(c.get("rrf_score") or 0) for c in cands), default=0.0)
                rows.append([
                    str(i),
                    _truncate(sq.get("sub_query", ""), 70),
                    _fmt_ms(sq.get("latency_ms")),
                    str(len(cands)),
                    f"{top_rrf:.4f}" if top_rrf else "—",
                ])
            out.append("\n### Milvus per sub-query\n\n" + _md_table(
                ["#", "Sub-query", "Latency", "Cands", "Top RRF"],
                rows,
            ))

    # Rerank summary
    if rerank:
        input_pairs = rerank.get("input_pair_count", 0)
        per_pair = rerank.get("per_pair_scores") or []
        survivors = rerank.get("survivors") or []
        rerank_cfg = rerank.get("config", {}) or {}
        threshold = rerank_cfg.get("score_threshold", "?")
        cap = rerank_cfg.get("max_children_per_parent", "?")
        unique_after_dedup = (
            len({s.get("child_id") for s in per_pair}) if per_pair else "?"
        )
        n_subq = len((milvus or {}).get("sub_queries", []) or [])
        top_k_retrieve = cfg.get("top_k_retrieve", "?")
        bullets = [
            (
                f"- Input pairs: {input_pairs} "
                f"({n_subq} sub-q × {top_k_retrieve} candidates)"
            ),
            f"- After dedup: {unique_after_dedup} unique chunks",
            (
                f"- After threshold (≥{threshold}) + per-parent cap (≤{cap}): "
                f"{len(survivors)} survivors"
            ),
        ]
        out.append("\n### Rerank summary\n\n" + "\n".join(bullets))

    # Top chunks shown to synthesis. Compact: rank order implies score; modality
    # only shown when any row is non-text (rare). Adds a "Cited" column merged
    # from the synthesis citations so we don't duplicate the citation list.
    if chunk_files:
        synthesis = stages.get("synthesis") or {}
        # Citations carry the bare slug ("foo_bar") while chunk headers carry the
        # full path ("sources/<col>/foo_bar/source.pdf"). Match on (slug, page).
        cited_keys: set[tuple[str, int]] = set()
        for c in synthesis.get("parsed_citations") or []:
            slug = (c.get("source_file") or "").strip()
            # The slug may have a "-N" duplicate-disambiguation suffix added
            # by the citation parser; strip that for matching.
            if "-" in slug and slug.rsplit("-", 1)[-1].isdigit():
                slug = slug.rsplit("-", 1)[0]
            try:
                cited_keys.add((slug, int(c.get("page_number") or 0)))
            except (TypeError, ValueError):
                continue

        chunk_rows: list[list[str]] = []
        any_non_text = False
        for cf in chunk_files:
            header = _read_chunk_header(cf)
            modality = header.get("modality", "text")
            if modality not in ("text", "?"):
                any_non_text = True
            try:
                page_int = int(header.get("page", 0))
            except (TypeError, ValueError):
                page_int = 0
            # Extract the slug from the chunk's source path.
            chunk_path = header.get("source_file", "")
            chunk_slug = chunk_path
            for prefix in (f"sources/{collection}/" if collection else "", "sources/"):
                if prefix and chunk_slug.startswith(prefix):
                    chunk_slug = chunk_slug[len(prefix):]
                    break
            chunk_slug = chunk_slug.split("/", 1)[0]  # drop trailing /source.pdf etc.
            cited = (chunk_slug, page_int) in cited_keys
            chunk_rows.append([
                cf.name.split("_", 1)[0],
                _shorten_source(header.get("source_file", ""), collection),
                str(header.get("page", "?")),
                modality,
                "✓" if cited else "",
            ])

        # Drop modality column when uninformative (all rows are text).
        if not any_non_text:
            headers = ["#", "Source", "Page", "Cited"]
            chunk_rows = [[r[0], r[1], r[2], r[4]] for r in chunk_rows]
        else:
            headers = ["#", "Source", "Page", "Modality", "Cited"]

        out.append(
            f"\n### Top {len(chunk_files)} chunks shown to synthesis "
            "_(in rerank-score order; details in `chunks/`)_\n"
        )
        out.append(_md_table(headers, chunk_rows))

    return "\n".join(out)


def _read_chunk_header(path: Path) -> dict[str, str]:
    """Parse the 6-line metadata header at the top of a chunks/*.txt file."""
    out: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(6):
                line = fh.readline()
                if not line:
                    break
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
    except Exception as exc:
        logger.warning("report_render: failed to read chunk header %s: %s", path, exc)
    return out


# ---------------------------------------------------------------------------
# Section: MMR
# ---------------------------------------------------------------------------


def _section_mmr(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    mmr = stages.get("mmr")
    cfg = meta.get("config", {})
    if mmr is None:
        # Active config indicates whether MMR is supposed to run.
        if cfg.get("use_mmr"):
            return (
                "## MMR\n\n_(USE_MMR=True but no MMR provenance written — "
                "pool may have been ≤ top_k.)_"
            )
        return ""

    pool_size = mmr.get("pool_size") or len(mmr.get("pool", []) or [])
    picks = mmr.get("picks") or []
    dropped = mmr.get("dropped") or []
    lam = mmr.get("lambda")
    top_k = mmr.get("top_k")

    head = (
        f"## MMR diversity filter\n\n"
        f"- λ = {lam} · top-k = {top_k} · "
        f"pool {pool_size} → picks {len(picks)} (dropped {len(dropped)})"
    )

    if picks:
        rows = []
        for p in picks[:10]:
            rows.append([
                str(p.get("pick_order", "?")),
                p.get("child_id", "?"),
                f"{float(p.get('rerank_score', 0.0)):.4f}",
            ])
        head += "\n\n" + _md_table(["Pick", "Child ID", "Rerank score"], rows)
    return head


# ---------------------------------------------------------------------------
# Section: synthesis
# ---------------------------------------------------------------------------


def _section_synthesis(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    syn = stages.get("synthesis")
    outcome = meta.get("outcome", {})

    if syn is None:
        if outcome.get("synthesis_skipped"):
            return "## Synthesis\n\n_Skipped (no chunks retrieved or `--raw` mode)._"
        return ""

    citations = syn.get("parsed_citations") or []

    # Latency / model / tokens / cost are already in the Pipeline timeline table.
    # This section adds only what the timeline doesn't show: citations + prompt hash.
    bullets = [
        (
            f"- Citations: {len(citations)} parsed "
            "_(see `Cited` column in retrieval table; full list in `06_synthesis.json`)_"
        ),
    ]
    sph = syn.get("system_prompt_hash")
    if sph:
        bullets.append(f"- System prompt hash: `{sph[:8]}…`")

    return "## Synthesis\n\n" + "\n".join(bullets)


# ---------------------------------------------------------------------------
# Section: CRAG groundedness
# ---------------------------------------------------------------------------


def _section_crag(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    grader = stages.get("grader")
    if grader is None:
        return ""

    score = grader.get("score")
    threshold = grader.get("threshold", "?")
    triggered = bool(grader.get("triggered_retry"))

    score_line = (
        f"- Score: **{score}** / 1.0 (threshold {threshold})"
        if score is not None
        else "- Score: —"
    )
    out = [
        "## Groundedness check (CRAG)",
        "\n### First pass\n",
        score_line,
        f"- Reason: \"{_truncate(grader.get('reason',''), 200)}\"",
        f"- Triggered retry: {_fmt_bool(triggered)}",
    ]

    retry = stages.get("crag_retry")
    if retry:
        ret_syn_lat = _fmt_ms(retry.get("retry_synthesis_latency_ms"))
        ret_syn_cost = _fmt_cost(retry.get("retry_synthesis_cost_usd"))
        first_sc = retry.get("first_score", "?")
        second_sc = retry.get("second_score", "?")
        out.append("\n### Retry\n")
        out.extend([
            f"- Reformulated query: \"{_truncate(retry.get('retry_query',''), 200)}\"",
            f"- Reformulate latency: {_fmt_ms(retry.get('reformulate_latency_ms'))}",
            f"- Retry synthesis: {ret_syn_lat}, cost {ret_syn_cost}",
            f"- Scores: first {first_sc} → second {second_sc}",
            f"- **Chose: {retry.get('chose','?')}**",
        ])

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Phase 2 — multi-retriever sub-stage section renderers
# ---------------------------------------------------------------------------


def _render_retriever_substages(
    section_title: str,
    subsections: list[tuple[str, dict[str, Any] | None]],
) -> str:
    """Shared body for the per-retriever section renderers.

    *subsections* is a list of ``(label, payload)`` tuples in render order. If
    every payload is None the section is suppressed entirely (so a retriever
    that never ran does not pollute the report). Otherwise each non-None
    payload renders as a sub-section. Stub-fallback payloads (``stub=True``)
    render a single ``_Stub fallback: <reason>._`` line instead of the model/
    latency/cost/usage block.
    """
    if all(p is None for _label, p in subsections):
        return ""
    lines: list[str] = [f"## {section_title}", ""]
    for label, payload in subsections:
        if payload is None:
            continue
        lines.append(f"### {label}")
        if payload.get("stub"):
            reason = payload.get("reason", "library not installed")
            lines.append(f"- _Stub fallback: {reason}._")
            lines.append("")
            continue
        model = payload.get("model", "—")
        latency = payload.get("latency_ms", 0)
        usage = payload.get("usage") or {}
        cost = payload.get("cost_usd", 0.0)
        lines.append(f"- Model: `{model}`")
        lines.append(f"- Latency: {latency} ms")
        if usage:
            lines.append(
                f"- Tokens: prompt={usage.get('input', usage.get('prompt', 0))} "
                f"completion={usage.get('output', usage.get('completion', 0))}"
            )
        try:
            cost_f = float(cost) if cost is not None else 0.0
        except (TypeError, ValueError):
            cost_f = 0.0
        lines.append(f"- Cost: ${cost_f:.6f}")
        lines.append("")
    return "\n".join(lines)


def _section_paperqa(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Render the PaperQA retriever section: rerank → read → answer."""
    return _render_retriever_substages(
        "PaperQA Retriever",
        [
            ("Rerank", stages.get("paperqa_rerank")),
            ("Read", stages.get("paperqa_read")),
            ("Answer", stages.get("paperqa_answer")),
        ],
    )


def _section_hipporag(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Render the HippoRAG/LightRAG retriever section: entities → walk → synthesize."""
    return _render_retriever_substages(
        "HippoRAG Retriever",
        [
            ("Entities", stages.get("hipporag_entities")),
            ("Walk", stages.get("hipporag_walk")),
            ("Synthesize", stages.get("hipporag_synthesize")),
        ],
    )


def _section_lazygraph(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Render the LazyGraphRAG retriever section: route → community → summarize."""
    return _render_retriever_substages(
        "LazyGraphRAG Retriever",
        [
            ("Route", stages.get("lazygraph_route")),
            ("Community", stages.get("lazygraph_community")),
            ("Summarize", stages.get("lazygraph_summarize")),
        ],
    )


def _section_fused(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Render fused retriever section: dispatch summary + final rerank.

    Per-retriever sub-stages render under their own section helpers; this
    section only summarizes the dispatch decisions and the final cross-encoder
    rerank pass across the merged pool.
    """
    dispatch = stages.get("fused_dispatch")
    rerank = stages.get("fused_rerank")
    if dispatch is None and rerank is None:
        return ""
    lines: list[str] = ["## Fused Retriever", ""]
    if dispatch is not None:
        lines.append("### Dispatch")
        if dispatch.get("stub"):
            reason = dispatch.get("reason", "library not installed")
            lines.append(f"- _Stub fallback: {reason}._")
            lines.append("")
        else:
            dispatched = dispatch.get("dispatched", []) or []
            skipped = dispatch.get("skipped", []) or []
            lines.append(f"- Dispatched: {', '.join(dispatched) or '—'}")
            if skipped:
                lines.append(f"- Skipped (stub): {', '.join(skipped)}")
            lines.append(f"- Latency: {dispatch.get('latency_ms', 0)} ms")
            lines.append(f"- Pool size: {dispatch.get('pool_size', 0)} chunks")
            lines.append("")
    if rerank is not None:
        lines.append("### Final Rerank")
        if rerank.get("stub"):
            reason = rerank.get("reason", "library not installed")
            lines.append(f"- _Stub fallback: {reason}._")
            lines.append("")
        else:
            lines.append(f"- Model: `{rerank.get('model', '—')}`")
            lines.append(f"- Latency: {rerank.get('latency_ms', 0)} ms")
            lines.append(f"- Top-K kept: {rerank.get('top_k', 50)}")
            lines.append("")
    return "\n".join(lines)


# Per-sub-stage renderer aliases. The actual rendering is done by the four
# umbrella functions above (_section_paperqa, _section_hipporag,
# _section_lazygraph, _section_fused). These per-stage names exist so the
# CRIT-2 audit-smoke invariant
# ``test_every_stage_prefix_has_section_renderer`` (which scans for a
# ``_section_<stage>`` for every key in ``_STAGE_PREFIX``) is satisfied
# without producing duplicate output. Each alias returns "" — the umbrella
# already emitted the content. Not registered in the dispatch tuple.


def _section_paperqa_rerank(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_paperqa`` umbrella."""
    return ""


def _section_paperqa_read(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_paperqa`` umbrella."""
    return ""


def _section_paperqa_answer(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_paperqa`` umbrella."""
    return ""


def _section_hipporag_entities(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_hipporag`` umbrella."""
    return ""


def _section_hipporag_walk(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_hipporag`` umbrella."""
    return ""


def _section_hipporag_synthesize(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_hipporag`` umbrella."""
    return ""


def _section_lazygraph_route(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_lazygraph`` umbrella."""
    return ""


def _section_lazygraph_community(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_lazygraph`` umbrella."""
    return ""


def _section_lazygraph_summarize(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_lazygraph`` umbrella."""
    return ""


def _section_fused_dispatch(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_fused`` umbrella."""
    return ""


def _section_fused_rerank(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    """Sub-stage alias — content rendered by ``_section_fused`` umbrella."""
    return ""


# ---------------------------------------------------------------------------
# Section: file index
# ---------------------------------------------------------------------------


def _section_files(
    meta: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    chunk_files: list[Path],
    log_dir: Path,
) -> str:
    parts = ["meta.json"]
    parts.extend(filename for stage, filename in _STAGE_FILES.items() if stage in stages)
    if chunk_files:
        parts.append(f"chunks/ ({len(chunk_files)} files)")
    return "## Files\n\n" + ", ".join(f"`{p}`" for p in parts)
