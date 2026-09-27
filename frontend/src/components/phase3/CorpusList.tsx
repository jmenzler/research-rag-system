/**
 * Corpus list (/app/corpus) — port of the design bundle's corpus-list.jsx.
 *
 * Read-only paper browser: Corpus-scoped tab bar (All Papers active; Recent
 * Chunks + Quality Alerts visibly disabled / "soon"), FTS5 search, grouped
 * facet chip row, dense sortable table with sticky header + 3-state sort glyph
 * + keyboard-focusable rows + degraded "—" cells, cursor-pagination footer.
 * NO ingest/upload/mutation affordance anywhere.
 *
 * Filter + cross-link state lives in URL search params (mirrors the existing
 * FilterChipRow useSearch/zod pattern), not Zustand.
 */
import { useNavigate } from "@tanstack/react-router";
import { useState, useRef, useEffect } from "react";
import type { ReactElement } from "react";

import { Icon } from "@/components/phase3/Icon";
import { usePaperListQuery } from "@/queries/corpus";
import type { Collection } from "@/queries/phase3Fixtures";
import { Route } from "@/routes/app/_layout/corpus";

const COLLECTIONS: Collection[] = ["trading", "ecology", "notes", "system"];
const INGESTED_OPTS: Array<[string, string]> = [
  ["any", "Any time"],
  ["7d", "Last 7 days"],
  ["30d", "Last 30 days"],
  ["1y", "Last year"],
];
const SEARCH_DEBOUNCE_MS = 200;

function digits(v: string): string {
  return v.replace(/[^\d]/g, "").slice(0, 4);
}

export function CorpusList(): ReactElement {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });

  const collections = search.collections;
  const sortCol = search.sort;
  const sortDir = search.dir;
  const scopeChat = search.scopeChatId ?? null;
  const scopeChatTitle = search.scopeChatTitle ?? "";

  const list = usePaperListQuery({
    q: search.q,
    collections,
    yearFrom: search.yearFrom,
    yearTo: search.yearTo,
    ingested: search.ingested,
    hasArxiv: search.hasArxiv,
    hasDoi: search.hasDoi,
    sort: sortCol,
    dir: sortDir,
    scopeChatId: scopeChat,
    cursor: search.cursor ?? null,
  });
  const papers = list.data?.papers ?? [];

  // Debounced search input — local state for the input value,
  // URL update fires SEARCH_DEBOUNCE_MS after last keystroke.
  const [queryInput, setQueryInput] = useState(search.q);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync local state when the URL param changes externally (clear scope, etc.)
  useEffect(() => {
    setQueryInput(search.q);
  }, [search.q]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, []);

  const patch = (next: Partial<typeof search>): void => {
    void navigate({ search: (prev) => ({ ...prev, ...next }) });
  };

  const toggleCol = (c: Collection): void => {
    const set = new Set(collections);
    if (set.has(c)) set.delete(c);
    else set.add(c);
    patch({ collections: [...set] as Collection[], cursor: undefined });
  };

  const onSort = (col: typeof sortCol): void => {
    if (sortCol === col) {
      patch({ dir: sortDir === "asc" ? "desc" : "asc" });
    } else {
      patch({ sort: col, dir: "desc", cursor: undefined });
    }
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
          <span aria-current="page">Corpus</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-meta">
          <span className="mono">parents.sqlite ⊕ meta.json ⊕ milvus</span>
          <span className="sep" style={{ color: "var(--p3-muted-2)" }}>
            ·
          </span>
          <span className="live-dot" />
          <span>refreshed 12s ago</span>
        </div>
      </div>

      <div className="tabbar" role="tablist">
        <button className="tab active" role="tab" aria-selected="true">
          <Icon name="library" size={14} className="icon" />
          <span>All Papers</span>
          <span
            className="tnum"
            style={{
              marginLeft: 2,
              color: "var(--p3-muted)",
              fontFamily: "var(--p3-font-mono)",
              fontSize: 11,
            }}
          >
            {list.data?.totalLabel ?? "—"}
          </span>
        </button>
        <button
          className="tab disabled"
          role="tab"
          aria-selected={false}
          disabled
          title="Reserved for a future phase"
          aria-disabled="true"
        >
          <Icon name="layers" size={14} className="icon" />
          <span>Recent Chunks</span>
          <span className="pill">soon</span>
        </button>
        <button
          className="tab disabled"
          role="tab"
          aria-selected={false}
          disabled
          title="Reserved for a future phase"
          aria-disabled="true"
        >
          <Icon name="alert-triangle" size={14} className="icon" />
          <span>Quality Alerts</span>
          <span className="pill">soon</span>
        </button>
      </div>

      <div className="content">
        <div className="content-inner" style={{ maxWidth: 1200 }}>
          <div className="toolbar">
            <div className="search">
              <Icon name="search" size={15} />
              <input
                value={queryInput}
                onChange={(e) => {
                  const v = e.target.value;
                  setQueryInput(v);
                  if (debounceTimer.current) clearTimeout(debounceTimer.current);
                  debounceTimer.current = setTimeout(() => {
                    patch({ q: v, cursor: undefined });
                  }, SEARCH_DEBOUNCE_MS);
                }}
                placeholder="Search title, authors, year, short-cite, arXiv, DOI… (FTS5)"
                aria-label="Search corpus"
              />
              <span className="kbd">/</span>
            </div>

            <div className="facets">
              {scopeChat !== null && (
                <div className="facet-group">
                  <div className="facet-label">Cross-link</div>
                  <div className="facet-chips">
                    <span
                      className="chip scope"
                      title="Cross-link arrival — clear to widen scope"
                    >
                      <Icon name="link" size={11} />
                      <span>cited in:&nbsp;</span>
                      <span className="mono">
                        {scopeChatTitle.slice(0, 28)}
                        {scopeChatTitle.length > 28 ? "…" : ""}
                      </span>
                      <button
                        className="x"
                        aria-label="Clear scope"
                        onClick={() =>
                          patch({ scopeChatId: undefined, scopeChatTitle: undefined })
                        }
                      >
                        <Icon name="x" size={11} />
                      </button>
                    </span>
                  </div>
                </div>
              )}

              <div className="facet-group">
                <div className="facet-label">Collection</div>
                <div className="facet-chips">
                  {COLLECTIONS.map((c) => {
                    const on = collections.includes(c);
                    return (
                      <button
                        key={c}
                        className={"chip " + (on ? "on" : "")}
                        onClick={() => toggleCol(c)}
                        aria-pressed={on}
                      >
                        <span className="mono">{c}</span>
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
                <div className="facet-label">Year</div>
                <div className="facet-chips">
                  <span className="chip chip-num">
                    <span style={{ color: "var(--p3-muted)" }}>from</span>
                    <input
                      value={search.yearFrom}
                      onChange={(e) => patch({ yearFrom: digits(e.target.value) })}
                      placeholder="2008"
                      aria-label="Year from"
                    />
                  </span>
                  <span className="chip chip-num">
                    <span style={{ color: "var(--p3-muted)" }}>to</span>
                    <input
                      value={search.yearTo}
                      onChange={(e) => patch({ yearTo: digits(e.target.value) })}
                      placeholder="2026"
                      aria-label="Year to"
                    />
                  </span>
                </div>
              </div>

              <div className="facet-sep" />

              <div className="facet-group">
                <div className="facet-label">Ingested</div>
                <div className="facet-chips">
                  {INGESTED_OPTS.map(([v, l]) => (
                    <button
                      key={v}
                      className={"chip " + (search.ingested === v ? "on" : "")}
                      onClick={() =>
                        patch({ ingested: v as typeof search.ingested })
                      }
                      aria-pressed={search.ingested === v}
                    >
                      <span>{l}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="facet-sep" />

              <div className="facet-group">
                <div className="facet-label">Identifiers</div>
                <div className="facet-chips">
                  <button
                    className={"chip " + (search.hasArxiv ? "on" : "")}
                    onClick={() => patch({ hasArxiv: !search.hasArxiv })}
                    aria-pressed={search.hasArxiv}
                  >
                    <Icon
                      name="check"
                      size={11}
                      style={{ opacity: search.hasArxiv ? 1 : 0.3 }}
                    />
                    <span>has arXiv</span>
                  </button>
                  <button
                    className={"chip " + (search.hasDoi ? "on" : "")}
                    onClick={() => patch({ hasDoi: !search.hasDoi })}
                    aria-pressed={search.hasDoi}
                  >
                    <Icon
                      name="check"
                      size={11}
                      style={{ opacity: search.hasDoi ? 1 : 0.3 }}
                    />
                    <span>has DOI</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {papers.length === 0 ? (
            <div className="empty">
              <span className="glyph">
                <Icon name="library" size={20} />
              </span>
              <h3>
                {scopeChat
                  ? "No papers cited in this chat"
                  : "No papers match these filters"}
              </h3>
              <p>
                {scopeChat
                  ? "This chat's responses didn't cite any indexed papers."
                  : "Try removing a filter or searching different terms."}
              </p>
            </div>
          ) : (
            <div className="tablewrap">
              <div className="tablescroll">
                <table className="data">
                  <colgroup>
                    <col style={{ width: "auto" }} />
                    <col style={{ width: 110 }} />
                    <col style={{ width: 70 }} />
                    <col style={{ width: 90 }} />
                    <col style={{ width: 180 }} />
                    <col style={{ width: 120 }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th
                        className={"sortable " + (sortCol === "title" ? "active" : "")}
                        onClick={() => onSort("title")}
                      >
                        Title / Authors {sortGlyph("title")}
                      </th>
                      <th>Collection</th>
                      <th
                        className={
                          "sortable num " + (sortCol === "year" ? "active" : "")
                        }
                        onClick={() => onSort("year")}
                      >
                        Year {sortGlyph("year")}
                      </th>
                      <th className="num">Chunks</th>
                      <th>arXiv / DOI</th>
                      <th
                        className={
                          "sortable " + (sortCol === "ingested" ? "active" : "")
                        }
                        onClick={() => onSort("ingested")}
                      >
                        Ingested {sortGlyph("ingested")}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {papers.map((p) => {
                      const open = (): void => {
                        void navigate({
                          to: "/app/corpus/$paperId",
                          params: { paperId: p.id },
                        });
                      };
                      return (
                        <tr
                          key={p.id}
                          role="link"
                          aria-label={`Open paper: ${p.title}`}
                          onClick={open}
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              open();
                            }
                          }}
                        >
                          <td className="title-cell">
                            <div
                              className="t truncate"
                              style={{ maxWidth: 520 }}
                              title={p.title}
                            >
                              {p.title}
                            </div>
                            <div
                              className="a truncate"
                              style={{ maxWidth: 520 }}
                              title={p.authors}
                            >
                              {p.authors}
                            </div>
                          </td>
                          <td>
                            <span className={"coltag c-" + p.collection}>
                              {p.collection}
                            </span>
                          </td>
                          <td className="num mono">{p.year}</td>
                          <td className="num">
                            {p.degradedChunks ? (
                              <span
                                className="degraded"
                                title="Milvus unreachable — chunk count temporarily missing"
                              >
                                <span className="dash" />
                              </span>
                            ) : (
                              <span className="mono">{p.chunks}</span>
                            )}
                          </td>
                          <td
                            className="mono muted"
                            style={{ fontSize: 11 }}
                            title={p.arxiv ?? p.doi ?? ""}
                          >
                            {p.arxiv ? (
                              p.arxiv
                            ) : p.doi ? (
                              p.doi
                            ) : (
                              <span className="degraded" title="no arXiv id or DOI">
                                <span className="dash" />
                              </span>
                            )}
                          </td>
                          <td className="mono muted" style={{ fontSize: 11 }}>
                            {p.ingested}
                            <Icon
                              name="chevron-right"
                              size={14}
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
                {papers.length}
              </span>{" "}
              shown · cursor-paginated · ≤200/page
            </span>
            <div className="pager-buttons">
              <button className="btn ghost" disabled>
                <Icon name="arrow-left" size={13} /> Prev
              </button>
              <button
                className="btn"
                disabled={list.data?.nextCursor == null}
                onClick={() => {
                  const nc = list.data?.nextCursor;
                  if (nc) patch({ cursor: nc });
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
