/**
 * Corpus detail (/app/corpus/$paperId) — port of corpus-detail.jsx.
 *
 * Provenance trail definition list; "Queries that retrieved this paper" list;
 * collapsible "Chats that cited this paper" minimal list; right aside actions
 * card whose button navigates to the query log carrying a `scopePaperId`
 * cross-link. No mutation affordance.
 */
import { Link, useNavigate } from "@tanstack/react-router";
import type { ReactElement, ReactNode } from "react";

import { api } from "@/lib/api";
import { Icon } from "@/components/phase3/Icon";
import { usePaperDetailQuery } from "@/queries/corpus";
import type { ProvenanceEntry } from "@/queries/phase3Fixtures";
import { Route } from "@/routes/app/_layout/corpus.$paperId";

function Degraded({ reason }: { reason: string }): ReactElement {
  return (
    <span className="degraded" title={reason}>
      <span className="dash" />
    </span>
  );
}

function ExtractionQuality({
  level,
}: {
  level: "high" | "medium" | "low";
}): ReactElement {
  if (level === "high")
    return (
      <span style={{ color: "var(--p3-ok)" }} className="mono">
        ● high
      </span>
    );
  if (level === "medium")
    return (
      <span style={{ color: "var(--p3-amber)" }} className="mono">
        ● medium
      </span>
    );
  return (
    <span style={{ color: "var(--p3-muted)" }} className="mono">
      ● low
    </span>
  );
}

function renderProvValue(
  r: ProvenanceEntry,
  collection: string,
  parentChunks: number,
  extractionQuality: "high" | "medium" | "low",
): ReactNode {
  if (r.degradedReason !== undefined) return <Degraded reason={r.degradedReason} />;
  if (r.kind === "coltag")
    return <span className={"coltag c-" + collection}>{collection}</span>;
  if (r.kind === "extraction-quality")
    return <ExtractionQuality level={extractionQuality} />;
  if (r.kind === "chunks")
    return (
      <span>
        <span className="mono">{r.v}</span>{" "}
        <span className="extra">child · {parentChunks} parent</span>
      </span>
    );
  if (r.k === "ingested")
    return (
      <span>
        <span className="mono">{r.v}</span>{" "}
        <span className="extra">14:08 UTC</span>
      </span>
    );
  return r.v;
}

export function CorpusDetail(): ReactElement {
  const { paperId } = Route.useParams();
  const navigate = useNavigate();
  const detail = usePaperDetailQuery(paperId);

  if (detail.data === undefined) {
    return (
      <>
        <div className="topbar">
          <div className="crumb">
            <Link to="/app/corpus">Corpus</Link>
            <span className="sep">/</span>
            <span className="current mono">{paperId}</span>
          </div>
        </div>
        <div className="content">
          <div className="content-inner" style={{ maxWidth: 960 }}>
            <div className="skel" style={{ width: "60%", height: 24 }} />
          </div>
        </div>
      </>
    );
  }

  const paper = detail.data;
  const openQueriesScoped = (): void => {
    void navigate({
      to: "/app/queries",
      search: { scopePaperId: paper.id },
    });
  };

  return (
    <>
      <div className="topbar">
        <div className="crumb">
          <Link to="/app/corpus">Corpus</Link>
          <span className="sep">/</span>
          <span className="current mono">{paper.id}</span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <a
            className="btn ghost"
            href={api.papers.contentListUrl(paper.id)}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon name="external-link" size={13} /> content_list.json
          </a>
        </div>
      </div>

      <div className="pagehead">
        <div
          style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}
        >
          <button
            className="btn ghost"
            onClick={() => void navigate({ to: "/app/corpus" })}
            style={{ padding: 0, height: 24, color: "var(--p3-muted)" }}
            title="Back to corpus"
            aria-label="Back to corpus"
          >
            <Icon name="arrow-left" size={14} />
          </button>
          <span className={"coltag c-" + paper.collection}>{paper.collection}</span>
          <span className="mono" style={{ color: "var(--p3-muted)", fontSize: 11 }}>
            {paper.year}
          </span>
        </div>
        <h1>{paper.title}</h1>
        <div className="substat">
          <span style={{ color: "var(--p3-fg-2)" }}>{paper.authors}</span>
          <span className="sep">·</span>
          <span className="mono">{paper.shortCite}</span>
          {paper.arxiv !== null && (
            <>
              <span className="sep">·</span>
              <a
                className="mono"
                style={{ color: "var(--p3-primary)", textDecoration: "none" }}
              >
                arXiv:{paper.arxiv}
              </a>
            </>
          )}
          {paper.doi !== null && (
            <>
              <span className="sep">·</span>
              <a
                className="mono"
                style={{ color: "var(--p3-primary)", textDecoration: "none" }}
              >
                doi:{paper.doi}
              </a>
            </>
          )}
        </div>
      </div>

      <div className="content">
        <div className="content-inner" style={{ maxWidth: 960 }}>
          <div
            style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 32 }}
          >
            <div>
              <div className="section">
                <div className="section-head">
                  <h2>Provenance trail</h2>
                  <span className="rule" />
                  <span className="meta">CORPUS-05 · 3-substrate join</span>
                </div>
                <div className="card">
                  <dl className="prov">
                    {paper.provenance.map((r) => (
                      <div key={r.k} style={{ display: "contents" }}>
                        <dt>{r.k}</dt>
                        <dd
                          className={r.mono ? "mono" : ""}
                          style={
                            r.mono
                              ? { fontSize: 12, color: "var(--p3-fg)" }
                              : undefined
                          }
                        >
                          {renderProvValue(
                            r,
                            paper.collection,
                            paper.parentChunks,
                            paper.extractionQuality,
                          )}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>

              <div className="section">
                <div className="section-head">
                  <h2>Queries that retrieved this paper</h2>
                  <span className="count tnum">{paper.queriesForPaper.length}</span>
                  <span className="rule" />
                  <Link
                    className="meta"
                    to="/app/queries"
                    search={{ scopePaperId: paper.id }}
                    style={{ color: "var(--p3-primary)" }}
                  >
                    Open in query log →
                  </Link>
                </div>
                <div className="card" style={{ padding: 0 }}>
                  {paper.queriesForPaper.map((q) => (
                    <div
                      className="linkrow"
                      key={q.id}
                      onClick={() =>
                        void navigate({
                          to: "/app/queries/$queryId",
                          params: { queryId: q.id },
                        })
                      }
                    >
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div className="lk truncate" title={q.query}>
                          {q.query}
                        </div>
                        <div
                          className="mono"
                          style={{
                            fontSize: 11,
                            color: "var(--p3-muted-2)",
                            marginTop: 2,
                          }}
                        >
                          {q.id}
                        </div>
                      </div>
                      <div className="right">
                        <span className={"tag r-" + q.retriever}>{q.retriever}</span>
                        <span>${q.cost.toFixed(3)}</span>
                        <span>{q.ts.slice(0, 10)}</span>
                        <Icon name="chevron-right" size={13} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="section">
                <details>
                  <summary
                    style={{
                      listStyle: "none",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      marginBottom: 12,
                    }}
                  >
                    <Icon
                      name="chevron-right"
                      size={13}
                      className="icon"
                      style={{ color: "var(--p3-muted)" }}
                    />
                    <h2
                      style={{
                        margin: 0,
                        fontSize: 13,
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: "0.02em",
                        color: "var(--p3-fg-2)",
                      }}
                    >
                      Chats that cited this paper
                    </h2>
                    <span className="count tnum">{paper.chatsForPaper.length}</span>
                    <span
                      className="rule"
                      style={{
                        flex: 1,
                        height: 1,
                        background: "var(--p3-border-soft)",
                      }}
                    />
                  </summary>
                  <div className="card" style={{ padding: 0 }}>
                    {paper.chatsForPaper.map((c) => (
                      <div
                        className="linkrow"
                        key={c.id}
                        onClick={() =>
                          void navigate({
                            to: "/app/chat/$chatId",
                            params: { chatId: c.id },
                          })
                        }
                      >
                        <span className="lk">{c.title}</span>
                        <div className="right">
                          <span>turn {c.turn}</span>
                          <Icon name="external-link" size={13} />
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            </div>

            <aside>
              <div className="section">
                <div className="section-head">
                  <h2>Cross-links</h2>
                  <span className="rule" />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <button className="btn" onClick={openQueriesScoped}>
                    <Icon name="activity" size={13} />
                    Queries that retrieved this
                    <Icon
                      name="arrow-right"
                      size={13}
                      style={{ marginLeft: "auto", color: "var(--p3-muted)" }}
                    />
                  </button>
                  <a
                    className="btn"
                    href={api.papers.contentListUrl(paper.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon name="file-json" size={13} />
                    View raw content_list.json
                    <Icon
                      name="external-link"
                      size={12}
                      style={{ marginLeft: "auto", color: "var(--p3-muted)" }}
                    />
                  </a>
                </div>
                <p
                  style={{
                    fontSize: 11,
                    color: "var(--p3-muted)",
                    marginTop: 12,
                    lineHeight: 1.6,
                  }}
                >
                  Cross-links arrive as a removable scope chip in the
                  destination&apos;s facet row.
                </p>
              </div>

              <div className="section">
                <div className="section-head">
                  <h2>Read-view freshness</h2>
                  <span className="rule" />
                </div>
                <div className="card" style={{ padding: 14 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 6,
                    }}
                  >
                    <span className="live-dot" />
                    <span style={{ fontSize: 12, fontWeight: 500 }}>live</span>
                  </div>
                  <div
                    className="mono"
                    style={{
                      fontSize: 11,
                      color: "var(--p3-muted)",
                      lineHeight: 1.7,
                    }}
                  >
                    refreshed 12s ago
                    <br />
                    ttl 60s · watermark @ {paper.ingested}
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </div>
    </>
  );
}
