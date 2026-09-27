/**
 * TanStack Query modules for the Retrieval Trace Inspector / Audit Detail
 * surfaces (Phase 1 Plan 05, D-13 / D-14a).
 *
 * staleTime discipline (CRIT-3): the per-query audit trail on disk is canonical;
 * never serve a stale UI view.
 *
 * Phase 3 additions (Plan 03-05):
 *   - useQueryLogListQuery now calls the REAL GET /api/queries endpoint
 *     (no longer fixture-resolved).
 *   - useQueryDetailQuery now calls the REAL GET /api/queries/{qid} endpoint.
 *   - Mapping helpers convert the snake_case API response to the UI's camelCase
 *     QueryDetail / QueryLogRow shapes, with graceful degradation for fields
 *     not present in the on-disk audit trail (paperId, cite, chat/turn).
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  QueryAuditPayload,
  QueryListRowApi,
  QueryLogListResponse,
} from "@/lib/api";
import type {
  Outcome,
  QueryDetail,
  QueryListResult,
  QueryLogRow,
  QueryStage,
  ReportBlock,
  RetrievedChunk,
  Retriever,
  StageState,
} from "@/queries/phase3Fixtures";

/* ------------------------------------------------------------------ */
/*  Phase 1 — existing audit-trail hooks (unchanged contract)         */
/* ------------------------------------------------------------------ */

export function useQueryAuditQuery(
  queryId: string | null,
): UseQueryResult<QueryAuditPayload> {
  return useQuery({
    queryKey: ["queries", queryId],
    queryFn: () => api.queries.get(queryId as string),
    enabled: queryId !== null,
    staleTime: 0, // CRIT-3 — audit trail is canonical on disk
  });
}

export function useQueryStageQuery(
  queryId: string,
  name: string,
  options: { enabled?: boolean } = {},
): UseQueryResult<{ name: string; payload: unknown }> {
  return useQuery({
    queryKey: ["queries", queryId, "stage", name],
    queryFn: () => api.queries.stage(queryId, name),
    enabled: options.enabled !== false && queryId !== "",
    staleTime: 0,
  });
}

export function useQueryChunkQuery(
  queryId: string,
  filename: string,
  options: { enabled?: boolean } = {},
): UseQueryResult<string> {
  return useQuery({
    queryKey: ["queries", queryId, "chunk", filename],
    queryFn: () => api.queries.chunk(queryId, filename),
    enabled: options.enabled !== false && queryId !== "",
    staleTime: 0, // CRIT-3
  });
}

/* ------------------------------------------------------------------ */
/*  Phase 3 — query-log browser list + detail                         */
/* ------------------------------------------------------------------ */

export interface QueryLogListParams {
  retrievers: Retriever[];
  outcome: Outcome[];
  costMin: string;
  latencyMin: string;
  sort: "ts" | "cost" | "latencyMs";
  dir: "asc" | "desc";
  scopePaperId?: string | null;
  cursor?: string | null;
  limit?: number;
}

/* ---------- mapping helpers ---------- */

/**
 * Convert a snake_case API row to the UI's QueryLogRow shape.
 * Some fields (retriever, model, chat, turn) may be null in the API
 * response — apply sensible defaults for rendering.
 */
function toQueryRow(apiRow: QueryListRowApi): QueryLogRow {
  return {
    id: apiRow.query_id,
    ts: apiRow.ts_started ?? "",
    query: apiRow.query ?? "",
    retriever: (apiRow.retriever ?? "milvus") as Retriever,
    model: apiRow.model ?? "unknown",
    tokens: apiRow.tokens ?? 0,
    cost: apiRow.cost_usd ?? 0,
    latencyMs: apiRow.latency_ms ?? 0,
    outcome: apiRow.outcome as Outcome,
    chat: apiRow.chat_id ?? null,
    // turn is not stored in the audit trail; messages join supplies chat_id only.
    turn: null,
  };
}

/**
 * Map the raw GET /api/queries/{qid} response (meta + report_md +
 * stage/chunk filename lists) to the UI's QueryDetail shape.
 *
 * Several QueryDetail fields are NOT present in the on-disk audit trail
 * (paperId, cite, page, score for chunks; citation markers for report):
 * the mapper provides sensible empty/fallback values so the component
 * renders without crashing, and the visual shows what IS available.
 */
function toQueryDetail(
  queryId: string,
  payload: QueryAuditPayload,
): QueryDetail {
  const meta = payload.meta as Record<string, unknown>;
  const totals = (meta.totals as Record<string, unknown>) ?? {};
  const config = (meta.config as Record<string, unknown>) ?? {};
  const outcomeMeta = (meta.outcome as Record<string, unknown>) ?? {};
  const tokensTotal = ((totals.tokens as Record<string, unknown>)?.total as number) ?? 0;
  const costTotal = totals.cost_usd as number | null ?? 0;
  const latencyTotal = (totals.total_latency_ms as number) ?? 0;
  const nLlmCalls = (totals.n_llm_calls as number) ?? 0;

  // Derive outcome string (same logic as the backend _derive_outcome).
  let outcomeStr: Outcome = "ok";
  if (outcomeMeta.crag_retried || outcomeMeta.crag_chose === "retry") {
    outcomeStr = "retry";
  } else if (
    outcomeMeta.synthesis_skipped &&
    (outcomeMeta.n_chunks_to_synth as number) === 0
  ) {
    outcomeStr = "skipped";
    // Without totals, mark as failed.
  } else if (totals.cost_usd === undefined && Object.keys(totals).length === 0) {
    outcomeStr = "failed";
  }

  // Derive model from config, fallback to reranker.
  const modelStr: string = (config.gen_model as string)
    ?? ((meta.reranker as Record<string, unknown>)?.model as string)
    ?? "unknown";

  // Build QueryLogRow for the existing QueryDetail.row consumer.
  const row: QueryLogRow = {
    id: queryId,
    ts: (meta.ts_started as string) ?? "",
    query: (meta.query as string) ?? "",
    retriever: "milvus" as Retriever,
    model: modelStr,
    tokens: tokensTotal,
    cost: costTotal,
    latencyMs: latencyTotal,
    outcome: outcomeStr,
    chat: null,
    turn: null,
  };

  // Build report block from raw report_md.
  const report: ReportBlock[] = payload.report_md
    ? [
        {
          heading: "Answer",
          parts: [{ text: payload.report_md }],
        },
      ]
    : [
        {
          heading: "Answer",
          parts: [{ text: "(no report.md on disk)" }],
        },
      ];

  // Build stages from filename list.
  const stages: QueryStage[] = payload.stages.map((entry) => {
    const filename = typeof entry === "string" ? entry : "";
    const match = filename.match(/^(\d{2})_(.+)\.json$/);
    const prefix = match?.[1] ?? "99";
    const name = match?.[2] ?? filename.replace(/\.json$/, "");
    const state: StageState = "run";
    return {
      n: prefix,
      name,
      state,
      desc: `${name} stage`,
      // Payload is lazy-fetched; omit the key entirely so that TS strict
      // optional-property checking is satisfied (exactOptionalPropertyTypes).
    };
  });

  // Build chunks from filename list.  The audit trail stores chunk text with
  // header metadata (child_id, parent_id, source_file, page, score), but the
  // GET /api/queries/{qid} list endpoint returns only filenames.  Fetching
  // each chunk's text to extract metadata would be expensive; instead provide
  // a thin stub that shows the filename.
  const chunks: RetrievedChunk[] = payload.chunks.map((f) => ({
    f,
    cite: "",
    page: 0,
    score: 0,
    paperId: "",
  }));

  // KPI subtext derived from totals.
  const retrievalMs = (totals.retrieval_latency_ms as number) ?? 0;
  const synthesisMs = (totals.synthesis_latency_ms as number) ?? 0;
  const kpi = {
    tokensSub: `${nLlmCalls} LLM call${nLlmCalls !== 1 ? "s" : ""} · ${tokensTotal.toLocaleString()} total`,
    costSub: `$${costTotal.toFixed(4)} total`,
    latencySub:
      retrievalMs > 0
        ? `retrieve ${retrievalMs}ms · synth ${synthesisMs}ms`
        : `${latencyTotal}ms total`,
  };

  return { row, reportPath: "logs/queries/…/report.md", report, stages, chunks, kpi };
}

/* ---------- hooks ---------- */

/**
 * Fetch the paginated, filtered, sorted query log list from
 * ``GET /api/queries``.
 *
 * Replaces the ``fixtureQueries`` stub from Phase 1.  Builds
 * ``URLSearchParams`` from the params object and passes the query string
 * to ``api.queries.list()``.
 */
export function useQueryLogListQuery(
  params: QueryLogListParams,
): UseQueryResult<QueryListResult> {
  return useQuery({
    queryKey: ["queries", "log", params],
    queryFn: async (): Promise<QueryListResult> => {
      const sp = new URLSearchParams();
      if (params.retrievers.length > 0) {
        sp.set("retriever", params.retrievers.join(","));
      }
      if (params.outcome.length > 0) {
        sp.set("outcome", params.outcome.join(","));
      }
      if (params.costMin) sp.set("costMin", params.costMin);
      if (params.latencyMin) sp.set("latencyMin", params.latencyMin);
      sp.set("sort", params.sort);
      sp.set("dir", params.dir);
      if (params.cursor) sp.set("cursor", params.cursor);
      if (params.limit) sp.set("limit", String(params.limit));
      if (params.scopePaperId) sp.set("paper", params.scopePaperId);

      const data: QueryLogListResponse = await api.queries.list(sp.toString());
      return {
        queries: data.queries.map(toQueryRow),
        totalLabel: `${data.total.toLocaleString()} queries`,
        nextCursor: data.next_cursor,
      };
    },
    placeholderData: keepPreviousData,
    staleTime: 0,
  });
}

/**
 * Fetch query detail from ``GET /api/queries/{queryId}`` and map to the
 * UI's ``QueryDetail`` shape.
 *
 * Replaces the ``buildQueryDetail`` fixture resolution.
 */
export function useQueryDetailQuery(
  queryId: string | null,
): UseQueryResult<QueryDetail> {
  return useQuery({
    queryKey: ["queries", "log", "detail", queryId],
    queryFn: async (): Promise<QueryDetail> => {
      const payload = await api.queries.get(queryId as string);
      return toQueryDetail(queryId!, payload);
    },
    enabled: queryId !== null,
    placeholderData: keepPreviousData,
    staleTime: 0,
  });
}
