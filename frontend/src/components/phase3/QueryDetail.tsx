/**
 * Query detail (/app/queries/$queryId) — port of query-detail.jsx. The
 * forensic-audit centerpiece.
 *
 * Backlink to source chat + turn (or CLI-origin notice) → KPI totals strip
 * (tokens / cost / latency, mono figures) → rendered report.md card with
 * inline citation markers → collapsible per-stage accordion (left pip-rail)
 * with a 4-token JSON highlighter (keys blue / strings amber / numbers green /
 * bools violet) → retrieved-chunk file list.
 */
import { useState } from "react";
import type { ReactElement } from "react";
import { Link, useNavigate } from "@tanstack/react-router";

import { Icon } from "@/components/phase3/Icon";
import { highlightJson, plainJson } from "@/components/phase3/highlightJson";
import { SanitizedHtml } from "@/lib/sanitized-html";
import {
  useQueryChunkQuery,
  useQueryDetailQuery,
  useQueryStageQuery,
} from "@/queries/queries";
import type { Outcome, QueryStage, RetrievedChunk } from "@/queries/phase3Fixtures";
import { Route } from "@/routes/app/_layout/queries.$queryId";

function outcomeLabel(o: Outcome): string {
  return o === "ok"
    ? "answered"
    : o === "retry"
      ? "crag retry"
      : o === "failed"
        ? "failed"
        : "skipped";
}

const SOURCE_CHAT_TITLES: Record<string, string> = {
  c01: "GLFT vs Avellaneda-Stoikov spread",
  c02: "VPIN toxicity windowing",
  c03: "LVR hedging on Uniswap v3",
  c05: "OFI as a short-horizon signal",
  c06: "Reranker fusion pool sizing",
  c08: "HNSW params for hybrid search",
  c11: "Hawkes processes for order arrivals",
};

export function QueryDetail(): ReactElement {
  const { queryId } = Route.useParams();
  const navigate = useNavigate();
  const detail = useQueryDetailQuery(queryId);
  const [openStages, setOpenStages] = useState<Set<string>>(new Set(["01"]));
  const [jsonHighlight] = useState(true);

  const toggleStage = (n: string): void => {
    setOpenStages((prev) => {
      const s = new Set(prev);
      if (s.has(n)) s.delete(n);
      else s.add(n);
      return s;
    });
  };

  if (detail.data === undefined) {
    return (
      <div className="content">
        <div className="content-inner" style={{ maxWidth: 920 }}>
          <div className="skel" style={{ width: "50%", height: 24 }} />
        </div>
      </div>
    );
  }

  const { row: q, report, stages, chunks, kpi, reportPath } = detail.data;
  const chatTitle =
    q.chat !== null ? (SOURCE_CHAT_TITLES[q.chat] ?? q.chat) : null;

  const runStages = stages.filter((s) => s.state === "run");
  const totalCost = runStages.reduce((a, s) => a + (s.cost ?? 0), 0);
  const totalMs = runStages.reduce((a, s) => a + (s.ms ?? 0), 0);

  const stageMeta = (s: QueryStage): ReactElement | null => {
    if (s.state === "skip") return <span className="skipped">skipped</span>;
    return (
      <>
        {s.model !== undefined && (
          <span>
            <b>{s.model}</b>
          </span>
        )}
        {s.tokens !== undefined && (
          <span className="tnum">{s.tokens.toLocaleString()} tok</span>
        )}
        {s.cost !== undefined && (
          <span className="tnum">${s.cost.toFixed(4)}</span>
        )}
        {s.ms !== undefined && <span className="tnum">{s.ms}ms</span>}
      </>
    );
  };

  return (
    <>
      <div className="topbar">
        <div className="crumb">
          <Link to="/app/queries">Queries</Link>
          <span className="sep">/</span>
          <span className="current mono" style={{ fontSize: 11 }}>
            {q.id}
          </span>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          <button className="iconbtn" title="Copy query id" aria-label="Copy query id">
            <Icon name="hash" size={14} />
          </button>
          <button
            className="iconbtn"
            title="Open raw audit dir"
            aria-label="Open raw audit dir"
          >
            <Icon name="external-link" size={14} />
          </button>
        </div>
      </div>

      <div className="pagehead">
        <div
          style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}
        >
          <button
            className="btn ghost"
            onClick={() => void navigate({ to: "/app/queries" })}
            style={{ padding: 0, height: 24, color: "var(--p3-muted)" }}
            title="Back to queries"
            aria-label="Back to queries"
          >
            <Icon name="arrow-left" size={14} />
          </button>
          <span className={"tag r-" + q.retriever}>{q.retriever}</span>
          <span className={"outcome " + q.outcome}>{outcomeLabel(q.outcome)}</span>
          <span className="mono" style={{ color: "var(--p3-muted)", fontSize: 11 }}>
            {q.ts} UTC
          </span>
        </div>
        <h1>{q.query}</h1>
      </div>

      <div className="content">
        <div className="content-inner" style={{ maxWidth: 920 }}>
          {chatTitle !== null && q.chat !== null ? (
            <Link
              className="backlink"
              to="/app/chat/$chatId"
              params={{ chatId: q.chat }}
            >
              <Icon name="arrow-left" size={13} />
              <span>from chat</span>
              <span className="lk">&quot;{chatTitle}&quot;</span>
              <span style={{ color: "var(--p3-muted)" }}>·</span>
              <span style={{ color: "var(--p3-muted)" }}>
                assistant turn {q.turn}
              </span>
            </Link>
          ) : (
            <div
              className="backlink"
              style={{ opacity: 0.55, cursor: "default" }}
              title="This query did not originate from a chat (CLI-origin)."
            >
              <Icon name="alert-circle" size={13} />
              <span style={{ color: "var(--p3-muted)" }}>
                CLI-origin · no source chat
              </span>
            </div>
          )}

          <div className="kpi-strip">
            <div className="kpi">
              <div className="lbl">
                <Icon name="hash" size={11} /> Tokens
              </div>
              <div className="val tnum">{q.tokens.toLocaleString()}</div>
              <div className="sub">{kpi.tokensSub}</div>
              <Icon name="hash" size={28} className="spark" />
            </div>
            <div className="kpi">
              <div className="lbl">
                <Icon name="coins" size={11} /> Cost
              </div>
              <div className="val tnum">${q.cost.toFixed(4)}</div>
              <div className="sub">{kpi.costSub}</div>
              <Icon name="coins" size={28} className="spark" />
            </div>
            <div className="kpi">
              <div className="lbl">
                <Icon name="zap" size={11} /> Latency
              </div>
              <div className="val tnum">{(q.latencyMs / 1000).toFixed(2)}s</div>
              <div className="sub">{kpi.latencySub}</div>
              <Icon name="zap" size={28} className="spark" />
            </div>
          </div>

          <div className="section">
            <div className="section-head">
              <h2>report.md</h2>
              <span className="rule" />
              <span className="meta mono">rendered from {reportPath}</span>
            </div>
            <div className="card report">
              {report.map((block, bi) => (
                <div key={bi}>
                  <h3>{block.heading}</h3>
                  <p style={{ whiteSpace: "pre-wrap" }}>
                    {block.parts.map((part, pi) =>
                      "text" in part ? (
                        <span key={pi}>{part.text}</span>
                      ) : (
                        <span
                          key={pi}
                          className="cite"
                          style={
                            part.resolved
                              ? undefined
                              : {
                                  background: "transparent",
                                  borderColor: "var(--p3-border)",
                                  color: "var(--p3-muted)",
                                  textDecoration: "line-through",
                                }
                          }
                          title={
                            part.resolved
                              ? `Citation ${part.cite}`
                              : "Unresolved citation marker"
                          }
                        >
                          {part.resolved ? part.cite : "?"}
                        </span>
                      ),
                    )}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="section">
            <div className="section-head">
              <h2>Pipeline stages</h2>
              <span className="count tnum">
                {runStages.length} / {stages.length}
              </span>
              <span className="rule" />
              <span className="meta mono">
                {totalMs.toLocaleString()}ms · ${totalCost.toFixed(4)}
              </span>
            </div>
            <div className="stages">
              {stages.map((s) => {
                const isSkip = s.state === "skip";
                const open = openStages.has(s.n) && !isSkip;
                return (
                  <details
                    key={s.n}
                    className={"stage " + s.state}
                    open={open}
                    onClick={(e) => {
                      if (isSkip) e.preventDefault();
                    }}
                  >
                    <summary
                      onClick={(e) => {
                        e.preventDefault();
                        if (!isSkip) toggleStage(s.n);
                      }}
                    >
                      <div className="num">
                        <span className="pip" />
                        <span>{s.n}</span>
                      </div>
                      <div className="head-main">
                        <div className="name">{s.name}</div>
                        <div className="desc">{isSkip ? s.note : s.desc}</div>
                      </div>
                      <div className="head-meta">
                        {stageMeta(s)}
                        {!isSkip && (
                          <Icon
                            name="chevron-right"
                            size={14}
                            className="chev"
                          />
                        )}
                      </div>
                    </summary>
                    {!isSkip && open && (
                      <StagePayload
                        queryId={q.id}
                        n={s.n}
                        name={s.name}
                        highlight={jsonHighlight}
                      />
                    )}
                  </details>
                );
              })}
            </div>
          </div>

          <div className="section">
            <div className="section-head">
              <h2>Retrieved chunks</h2>
              <span className="count tnum">{chunks.length}</span>
              <span className="rule" />
              <span className="meta mono">guarded read · path-canonicalize gate</span>
            </div>
            <div className="card" style={{ padding: 0 }}>
              {chunks.map((c) => (
                <ChunkRow key={c.f} queryId={q.id} chunk={c} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/**
 * Lazy-fetched per-stage payload. The query only fires when the parent
 * accordion row is open (QLOG-02): eager-fetching every stage payload on
 * page load is forbidden — same discipline as AuditDetailSheet's StageRow.
 */
function StagePayload(props: {
  queryId: string;
  n: string;
  name: string;
  highlight: boolean;
}): ReactElement {
  const stage = useQueryStageQuery(props.queryId, props.name, { enabled: true });
  const payloadJson =
    stage.data !== undefined
      ? JSON.stringify((stage.data as { payload?: unknown }).payload ?? stage.data, null, 2)
      : "";
  return (
    <div className="payload">
      <div className="crumbline">
        <span style={{ color: "var(--p3-fg-2)" }}>
          {props.n}_{props.name}.json
        </span>
        <span style={{ flex: 1 }} />
      </div>
      {stage.isLoading && <div className="json-pre">Loading stage payload…</div>}
      {stage.isError && (
        <div className="json-pre">Error: {String(stage.error)}</div>
      )}
      {stage.data !== undefined && (
        <SanitizedHtml
          className="json-pre"
          html={props.highlight ? highlightJson(payloadJson) : plainJson(payloadJson)}
        />
      )}
    </div>
  );
}

/**
 * One retrieved-chunk row. Clicking expands to lazily fetch the raw chunk
 * text via GET /api/queries/{qid}/chunks/{filename} (QLOG-03). When the chunk
 * resolves to an in-corpus paper, also offers a jump to the corpus detail.
 */
function ChunkRow(props: { queryId: string; chunk: RetrievedChunk }): ReactElement {
  const { chunk: c } = props;
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const text = useQueryChunkQuery(props.queryId, c.f, { enabled: open });
  const hasPaper = c.paperId !== "";
  return (
    <>
      <div
        className="chunk"
        onClick={() => setOpen((p) => !p)}
        title={`Open ${c.f}`}
        aria-expanded={open}
      >
        <Icon name="file-text" size={14} className="icon-bg" />
        <span
          style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}
        >
          <span className="filename">{c.f}</span>
          {c.cite ? (
            <span className="cite truncate">
              {c.cite} · p.{c.page}
            </span>
          ) : (
            <span className="cite truncate" style={{ color: "var(--p3-muted)" }}>
              chunk file
            </span>
          )}
        </span>
        {c.score > 0 ? <span className="score">{c.score.toFixed(2)}</span> : null}
        <Icon
          name="chevron-right"
          size={13}
          className="chev"
          style={{ color: "var(--p3-muted-2)" }}
        />
      </div>
      {open && (
        <div className="payload">
          {text.isLoading && <div className="json-pre">Loading chunk…</div>}
          {text.isError && (
            <div className="json-pre">Error: {String(text.error)}</div>
          )}
          {text.data !== undefined && (
            <pre className="json-pre" style={{ whiteSpace: "pre-wrap" }}>
              {text.data}
            </pre>
          )}
          {hasPaper && (
            <button
              className="btn ghost"
              style={{ height: 22, padding: "0 8px", fontSize: 11 }}
              onClick={() =>
                void navigate({
                  to: "/app/corpus/$paperId",
                  params: { paperId: c.paperId },
                })
              }
            >
              <Icon name="external-link" size={11} /> open paper {c.paperId}
            </button>
          )}
        </div>
      )}
    </>
  );
}
