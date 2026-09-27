/**
 * Query log list (/app/queries) — port of query-list.jsx.
 *
 * Single facet chip row where a cross-link scope chip (`paper: …`, amber-edged
 * + link glyph) sits inline as the FIRST chip — NO separate banner. Applied
 * filters render primary-filled with inline ×, available render outline. Dense
 * sortable table: Started / Query / Retriever / Model / Tokens / Cost /
 * Latency / Outcome. Cost column is NEUTRAL foreground, never amber. Retriever
 * color tags + outcome status dots.
 *
 * Filter + scope state lives in URL search params.
 */
import { useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";

import { Icon } from "@/components/phase3/Icon";
import { useQueryLogListQuery } from "@/queries/queries";
import type { Outcome, Retriever } from "@/queries/phase3Fixtures";
import { Route } from "@/routes/app/_layout/queries";

const RETRIEVERS: Retriever[] = ["milvus", "paperqa", "lightrag", "graphrag", "fused"];
const OUTCOMES: Array<[Outcome, string]> = [
  ["ok", "answered"],
  ["skipped", "skipped"],
  ["retry", "crag retry"],
  ["failed", "failed"],
];

function outcomeLabel(o: Outcome): string {
  return o === "ok"
    ? "answered"
    : o === "retry"
      ? "crag retry"
      : o === "failed"
        ? "failed"
        : "skipped";
}

export function QueryList(): ReactElement {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });

  const retrievers = search.retrievers;
  const outcome = search.outcome;
  const sortCol = search.sort;
  const sortDir = search.dir;
  const scope = search.scopePaperId ?? null;
  const cursor = search.cursor;

  const list = useQueryLogListQuery({
    retrievers,
    outcome,
    costMin: search.costMin,
    latencyMin: search.latencyMin,
    sort: sortCol,
    dir: sortDir,
    scopePaperId: scope,
    cursor: cursor ?? null,
  });
  const rows = list.data?.queries ?? [];

  const patch = (next: Partial<typeof search>): void => {
    void navigate({ search: (prev) => ({ ...prev, ...next }) });
  };

  /** Reset pagination when facet/sort filters change. */
  const patchWithReset = (next: Partial<typeof search>): void => {
    void navigate({ search: (prev) => ({ ...prev, ...next, cursor: undefined }) });
  };

  const toggleRetriever = (v: Retriever): void => {
    const set = new Set<Retriever>(retrievers);
    if (set.has(v)) set.delete(v);
    else set.add(v);
    patchWithReset({ retrievers: [...set] });
  };

  const toggleOutcome = (v: Outcome): void => {
    const set = new Set<Outcome>(outcome);
    if (set.has(v)) set.delete(v);
    else set.add(v);
    patchWithReset({ outcome: [...set] });
  };

  const onSort = (col: typeof sortCol): void => {
    if (sortCol === col) patchWithReset({ dir: sortDir === "asc" ? "desc" : "asc" });
    else patchWithReset({ sort: col, dir: "desc" });
  };

  const sortGlyph = (col: typeof sortCol): ReactElement =>
    sortCol !== col ? (
      <Icon name="arrow-up-down" size={11} className="sort-glyph" />
    ) : (
      <Icon
        name={sortDir === "asc" ? "arrow-up" : "arrow-down"}
        size={11}
        className="sort-glyph"
      />
    );

  return (
    <>
      <div className="topbar">
        <div className="crumb">
          <span aria-current="page">Queries</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-meta">
          <span className="live-dot" />
          <span>read from logs/queries/*/meta.json · no duplicate storage</span>
        </div>
      </div>

      <div className="pagehead">
        <h1>Query log</h1>
        <div className="substat">
          <span className="mono tnum">{list.data?.totalLabel ?? "—"}</span>
          <span className="sep">·</span>
          <span className="mono tnum">{rows.length} shown</span>
          <span className="sep">·</span>
          <span>per-row totals sourced live from audit trail</span>
        </div>
      </div>

      <div className="content">
        <div className="content-inner" style={{ maxWidth: 1280 }}>
          <div className="toolbar">
            <div className="facets">
              {scope !== null && (
                <div className="facet-group">
                  <div className="facet-label">Cross-link</div>
                  <div className="facet-chips">
                    <span
                      className="chip scope"
                      title="Cross-link arrival — narrowed to queries that retrieved this paper"
                    >
                      <Icon name="link" size={11} />
                      <span>paper:&nbsp;</span>
                      <span className="mono">{scope}</span>
                      <button
                        className="x"
                        aria-label="Clear scope"
                        onClick={() => patchWithReset({ scopePaperId: undefined })}
                      >
                        <Icon name="x" size={11} />
                      </button>
                    </span>
                  </div>
                </div>
              )}

              <div className="facet-group">
                <div className="facet-label">Retriever</div>
                <div className="facet-chips">
                  {RETRIEVERS.map((r) => {
                    const on = retrievers.includes(r);
                    return (
                      <button
                        key={r}
                        className={"chip " + (on ? "on" : "")}
                        onClick={() => toggleRetriever(r)}
                        aria-pressed={on}
                      >
                        <span
                          className="mono"
                          style={{
                            color: on
                              ? "var(--p3-fg)"
                              : `var(--p3-tag-${r}-fg)`,
                          }}
                        >
                          {r}
                        </span>
                        {on && (
                          <span className="x" aria-hidden="true">
                            <Icon name="x" size={11} />
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="facet-sep" />

              <div className="facet-group">
                <div className="facet-label">Outcome</div>
                <div className="facet-chips">
                  {OUTCOMES.map(([v, l]) => {
                    const on = outcome.includes(v);
                    return (
                      <button
                        key={v}
                        className={"chip " + (on ? "on" : "")}
                        onClick={() => toggleOutcome(v)}
                        aria-pressed={on}
                      >
                        <span>{l}</span>
                        {on && (
                          <span className="x" aria-hidden="true">
                            <Icon name="x" size={11} />
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="facet-sep" />

              <div className="facet-group">
                <div className="facet-label">Cost / latency</div>
                <div className="facet-chips">
                  <span className="chip chip-num">
                    <span style={{ color: "var(--p3-muted)" }}>cost &gt;</span>
                    <span style={{ color: "var(--p3-muted)" }}>$</span>
                    <input
                      value={search.costMin}
                      onChange={(e) => patchWithReset({ costMin: e.target.value })}
                      placeholder="0.02"
                      style={{ width: 50 }}
                      aria-label="Minimum cost"
                    />
                  </span>
                  <span className="chip chip-num">
                    <span style={{ color: "var(--p3-muted)" }}>latency &gt;</span>
                    <input
                      value={search.latencyMin}
                      onChange={(e) => patchWithReset({ latencyMin: e.target.value })}
                      placeholder="5"
                      style={{ width: 36 }}
                      aria-label="Minimum latency seconds"
                    />
                    <span style={{ color: "var(--p3-muted)" }}>s</span>
                  </span>
                </div>
              </div>
            </div>
          </div>

          {rows.length === 0 ? (
            <div className="empty">
              <span className="glyph">
                <Icon name="activity" size={20} />
              </span>
              <h3>
                {scope
                  ? "No queries retrieved this paper"
                  : "No queries match these filters"}
              </h3>
              <p>
                {scope
                  ? "No logged query has surfaced this paper."
                  : "Try removing a filter."}
              </p>
            </div>
          ) : (
            <div className="tablewrap">
              <div className="tablescroll">
                <table className="data">
                  <colgroup>
                    <col style={{ width: 142 }} />
                    <col style={{ width: "auto" }} />
                    <col style={{ width: 100 }} />
                    <col style={{ width: 152 }} />
                    <col style={{ width: 96 }} />
                    <col style={{ width: 84 }} />
                    <col style={{ width: 94 }} />
                    <col style={{ width: 122 }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th
                        className={"sortable " + (sortCol === "ts" ? "active" : "")}
                        onClick={() => onSort("ts")}
                      >
                        Started {sortGlyph("ts")}
                      </th>
                      <th>Query</th>
                      <th>Retriever</th>
                      <th>Model</th>
                      <th className="num">Tokens</th>
                      <th
                        className={
                          "sortable num " + (sortCol === "cost" ? "active" : "")
                        }
                        onClick={() => onSort("cost")}
                      >
                        Cost {sortGlyph("cost")}
                      </th>
                      <th
                        className={
                          "sortable num " +
                          (sortCol === "latencyMs" ? "active" : "")
                        }
                        onClick={() => onSort("latencyMs")}
                      >
                        Latency {sortGlyph("latencyMs")}
                      </th>
                      <th>Outcome</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((q) => {
                      const open = (): void => {
                        void navigate({
                          to: "/app/queries/$queryId",
                          params: { queryId: q.id },
                        });
                      };
                      return (
                        <tr
                          key={q.id}
                          role="link"
                          aria-label={`Open query: ${q.query}`}
                          onClick={open}
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              open();
                            }
                          }}
                        >
                          <td className="mono muted" style={{ fontSize: 11 }}>
                            {q.ts.slice(5, 16)}
                          </td>
                          <td className="title-cell">
                            <div
                              className="t truncate"
                              style={{ maxWidth: 480 }}
                              title={q.query}
                            >
                              {q.query}
                            </div>
                            <div
                              className="mono"
                              style={{
                                fontSize: 10.5,
                                color: "var(--p3-muted-2)",
                                marginTop: 2,
                              }}
                            >
                              {q.id.slice(-12)}
                            </div>
                          </td>
                          <td>
                            <span className={"tag r-" + q.retriever}>
                              {q.retriever}
                            </span>
                          </td>
                          <td className="mono muted" style={{ fontSize: 11 }}>
                            {q.model}
                          </td>
                          <td className="num mono">{q.tokens.toLocaleString()}</td>
                          <td className="num mono">${q.cost.toFixed(3)}</td>
                          <td className="num mono">
                            {(q.latencyMs / 1000).toFixed(1)}s
                          </td>
                          <td>
                            <span className={"outcome " + q.outcome}>
                              {outcomeLabel(q.outcome)}
                            </span>
                            <Icon
                              name="chevron-right"
                              size={13}
                              className="row-arrow"
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="pager">
            <span>
              <span className="tnum" style={{ color: "var(--p3-fg-2)" }}>
                {rows.length}
              </span>{" "}
              shown · cursor-paginated · ≤200/page · sort:{" "}
              <span className="mono" style={{ color: "var(--p3-fg-2)" }}>
                {sortCol} {sortDir}
              </span>
            </span>
            <div className="pager-buttons">
              <button
                className="btn ghost"
                disabled={cursor == null}
                onClick={() => patch({ cursor: undefined })}
              >
                <Icon name="arrow-left" size={13} /> Prev
              </button>
              <button
                className="btn"
                disabled={list.data?.nextCursor == null}
                onClick={() => {
                  if (list.data?.nextCursor) patch({ cursor: list.data.nextCursor });
                }}
              >
                Next <Icon name="arrow-right" size={13} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
