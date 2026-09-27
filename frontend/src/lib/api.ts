/**
 * API client. All paths are RELATIVE (`/api/...`) — same origin in production,
 * proxied via Vite dev server in development. NEVER use `http://localhost:8766`
 * or any hardcoded host literal (PITFALLS MOD-14).
 */

export interface VersionPayload {
  sha: string;
  built_at: string | null;
}

export interface HealthPayload {
  ok: boolean;
  sha: string;
  started_at: string;
}

// Phase 1 Plan 05 additions ------------------------------------------------

export interface ChatSummary {
  id: string;
  title: string | null;
  retriever: string;
  collections: string; // JSON string; UI parses
  version: number;
  created_at: string;
  updated_at: string;
}

/** Server-persisted citation row attached to an assistant message on reload.
 * Matches the SSE ``citations`` event payload shape (CitationPayload) so the
 * live and rehydrate paths feed the same downstream renderers.
 */
export interface PersistedCitation {
  marker: number;
  child_id: string | null;
  parent_id: string | null;
  paper_id: string | null;
  score_dense: number | null;
  score_sparse: number | null;
  score_rerank: number | null;
  resolved: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  query_id: string | null;
  message_uuid: string;
  created_at: string;
  version: number;
  /** Citations attached by the backend (assistant turns only). Empty list
   * for user turns and assistants with no extracted markers. */
  citations?: PersistedCitation[];
}

export interface ChatDetail extends ChatSummary {
  messages: ChatMessage[];
}

export interface CreateChatBody {
  title?: string | null;
  retriever?: string;
  collections?: string[];
}

export interface ChunkPayload {
  id: string;
  text: string;
  source_file: string;
  notebook: string;
  modality: string;
  page_number: number;
}

/**
 * Stage entry shape. The backend (src/server/api_queries.py) returns filenames
 * like "01_decompose.json" as strings. Some downstream callers (tests / future
 * Phase 3 expansion) emit a structured shape `{name, prefix}` so the Sheet can
 * render `01_decompose` while passing only `decompose` to the
 * /api/queries/{qid}/stages/{name} endpoint. AuditDetailSheet normalises both.
 */
export type StageEntry =
  | string
  | {
      name: string;
      prefix?: string;
      [key: string]: unknown;
    };

/**
 * Single row returned by the ``GET /api/queries`` list endpoint.
 * snake_case fields as the server returns them (query_index.py QueryRow);
 * queries.ts maps to the camelCase component shape.
 */
export interface QueryListRowApi {
  query_id: string;
  query: string | null;
  ts_started: string | null;
  retriever: string | null;
  model: string | null;
  tokens: number | null;
  cost_usd: number | null;
  latency_ms: number | null;
  outcome: string;
  chat_id: string | null;
}

/**
 * Response shape for ``GET /api/queries``.
 */
export interface QueryLogListResponse {
  queries: QueryListRowApi[];
  next_cursor: string | null;
  total: number;
}

export interface QueryAuditPayload {
  query_id: string;
  meta: { totals?: Record<string, unknown>; outcome?: Record<string, unknown> };
  report_md: string;
  // Phase 1 backend (Plan 03) emits strings: ["01_decompose.json", ...].
  // Phase 3+ may switch to structured entries; Sheet handles both shapes.
  stages: StageEntry[];
  chunks: string[]; // ["01_<child_id>.txt", ...]
}

// Phase 3 Plan 04 — PaperListApi / PaperDetailApi matching the real backend
// contract from api_papers.py + papers_view.py.  snake_case fields as the
// server returns them; the corpus.ts mappings convert to camelCase for the
// component prop shapes (defined in phase3Fixtures.ts).

/** Single paper row from the 3-substrate join read-view (PaperRow TypedDict). */
export interface PaperRowApi {
  paper_id: string;
  title: string | null;
  /** JSON-encoded array, e.g. '["Avellaneda","Stoikov"]', or null. */
  authors: string | null;
  year: number | null;
  collection: string;
  chunk_count: number | null;
  arxiv_id: string | null;
  doi: string | null;
  short_cite: string | null;
  /** Epoch timestamp from sources.updated_at. */
  ingested_at: number | null;
  extraction_quality: string | null;
  fetch_source: string | null;
  parser_version: string | null;
}

/** Raw GET /api/papers list response. */
export interface PaperListApi {
  papers: PaperRowApi[];
  next_cursor: string | null;
  total: number;
}

/** Raw GET /api/papers/{id} detail response. */
export interface PaperDetailApi extends PaperRowApi {
  content_list: string;
}

// Phase 4 — Graph Explorer API types (snake_case, matching api_graph.py Pydantic models)

export interface GraphNodeApi {
  corpus_id: number;
  title: string;
  year: number | null;
  citation_count: number | null;
  abstract?: string | null;
  short_cite?: string | null;
  is_seed: boolean;
}

export interface GraphEdgeApi {
  from_id: number;
  to_id: number;
}

export interface GraphResponseApi {
  nodes: GraphNodeApi[];
  edges: GraphEdgeApi[];
}

export interface SeedSearchHitApi {
  corpus_id: number;
  title: string;
  arxiv_id: string | null;
  doi: string | null;
  confidence: number;
  source: string;
}

export interface CollectionInfoApi {
  name: string;
  has_graph_data: boolean;
}

export interface InCorpusResultApi {
  in_corpus: boolean;
  collection: string | null;
}

export interface WalkRequest {
  corpus_ids: number[];
  depth: number;
  direction: string;
  year_from?: number | null;
  year_to?: number | null;
  min_citations: number;
  budget: number;
}

export interface ExpandRequest {
  corpus_ids: number[];
  direction: string;
  limit: number;
  min_citations: number;
}

export interface InCorpusCheckRequest {
  corpus_ids: number[];
}

export interface NodeDetailResponseApi {
  corpus_id: number;
  title: string;
  year: number | null;
  citation_count: number | null;
  authors: string[] | null;
  abstract: string | null;
  arxiv_id: string | null;
  doi: string | null;
}

// Ingest + Maps API types (snake_case, matching api_ingest.py Pydantic models)

export interface IngestRequestApi {
  corpus_ids: number[];
  collection: string;
}

export interface IngestResponseApi {
  run_ids: Array<{ run_id: string; corpus_ids: number[] }>;
  n_enqueued: number;
}

export interface MapApi {
  id: string;
  name: string;
  collection: string;
  snapshot: string;
  created_at: string;
  updated_at: string;
}

export type MapMetaApi = Omit<MapApi, "snapshot">;

export interface MapCoverageApi {
  in_corpus: number;
  total: number;
  missing_ids: number[];
}

export interface CreateMapBody {
  name: string;
  collection: string;
  snapshot: string;
}

// Errors thrown by postJson surface the parsed JSON body via `detail` so the
// optimistic-lock toast (CHAT-11) can read `detail.current_version` etc.
export interface ApiError extends Error {
  status?: number;
  detail?: unknown;
}

// 409 optimistic-lock toast wording — locked verbatim in Phase 1 D-17.
// Plan 02-12 moved this from ChatSurface.tsx into the api module so the new
// sidebar mutations (rename, archive, delete) can surface the same toast
// without forcing a circular import through ChatSurface.tsx.
const OPTIMISTIC_409_RE = /\bHTTP 409\b|\bversion mismatch\b/i;
export const OPTIMISTIC_409_TOAST =
  "This chat was updated in another tab — reloading…";

/**
 * Returns true when the given error carries a 409 status (structured ApiError)
 * OR the error message contains the "HTTP 409" / "version mismatch" sigil
 * (SSE error-string fallback path — see ChatSurface.tsx). Plan 02-12 extracted
 * this from the chat surface so the sidebar mutations can reuse it.
 */
export function isLock409(err: unknown): boolean {
  if (typeof err === "object" && err !== null && "status" in err) {
    return (err as ApiError).status === 409;
  }
  const msg = err instanceof Error ? err.message : String(err);
  return OPTIMISTIC_409_RE.test(msg);
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    // Surface 409 detail body so the optimistic-lock toast can read current_version.
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      /* ignore — non-JSON error body */
    }
    const err = new Error(`${path}: HTTP ${response.status}`) as ApiError;
    err.status = response.status;
    err.detail = detail;
    throw err;
  }
  return (await response.json()) as T;
}

async function getText(path: string): Promise<string> {
  const response = await fetch(path, {
    method: "GET",
    headers: { Accept: "text/plain" },
  });
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return await response.text();
}

// Plan 02-12 — generic PATCH wrapper. Surfaces 409 detail body so the
// optimistic-lock toast can introspect `detail.current_version` (Phase 1
// pattern carries through verbatim).
async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(`${path}: HTTP ${response.status}`) as ApiError;
    err.status = response.status;
    err.detail = detail;
    throw err;
  }
  return (await response.json()) as T;
}

// Plan 02-12 — generic DELETE wrapper. 204 No Content is the success shape;
// callers receive `void`. Errors surface as ApiError with status + detail.
async function deleteRequest(path: string): Promise<void> {
  const response = await fetch(path, { method: "DELETE" });
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = await response.json();
    } catch {
      /* ignore */
    }
    const err = new Error(`${path}: HTTP ${response.status}`) as ApiError;
    err.status = response.status;
    err.detail = detail;
    throw err;
  }
}

// Plan 02-12 — search payload shape (matches Plan 02-04 GET /api/chats/search).
// `snippet` arrives from sqlite FTS5 `snippet()` with <mark>…</mark> already
// wrapping the matched terms. Rendered through CITATION_SANITIZE_SCHEMA, which
// allowlists <mark> with no attributes (XSS-safe).
export interface ChatSearchResult {
  id: string;
  title: string | null;
  retriever: string;
  collections: string;
  version: number;
  created_at: string;
  updated_at: string;
  snippet: string;
}

export interface ChatSearchPayload {
  results: ChatSearchResult[];
}

export interface ChatSearchFilters {
  collections?: string[];
  retrievers?: string[];
  archived?: boolean;
}

export interface ChatPatchBody {
  title?: string;
  retriever?: string;
  collections?: string[];
  archived?: boolean;
  expected_version: number;
}

// Plan 02-13 — retriever availability map returned by GET /api/chats/retrievers.
// Used by <NewChatDialog> / <RetrieverPicker> to disable options whose backing
// Python library is not installed on the server (stub-fallback mode per
// 02-01-LIBRARY-SPIKE.md). The server (Pydantic Literal on PATCH/POST) is the
// canonical gate — even if the client bypasses the UI by toggling its own
// availability state, the server rejects unknown retriever strings (T-02-13-06).
export type RetrieverId =
  | "milvus"
  | "paperqa"
  | "hipporag"
  | "lazygraph"
  | "fused";

export type RetrieverAvailability = Record<RetrieverId, boolean>;

function buildSearchUrl(q: string, filters: ChatSearchFilters): string {
  const params = new URLSearchParams();
  params.set("q", q);
  if (filters.collections && filters.collections.length > 0) {
    params.set("collections", filters.collections.join(","));
  }
  if (filters.retrievers && filters.retrievers.length > 0) {
    params.set("retrievers", filters.retrievers.join(","));
  }
  if (filters.archived) {
    params.set("archived", "1");
  }
  return `/api/chats/search?${params.toString()}`;
}

export const api = {
  version: (): Promise<VersionPayload> => getJson("/api/version"),
  health: (): Promise<HealthPayload> => getJson("/api/health"),
  chats: {
    list: (): Promise<{ chats: ChatSummary[] }> => getJson("/api/chats"),
    get: (id: string): Promise<ChatDetail> => getJson(`/api/chats/${id}`),
    create: (body: CreateChatBody): Promise<ChatDetail> =>
      postJson("/api/chats", body),
    // Plan 02-12 — rename / archive / unarchive / retriever-or-collections
    // edits all funnel through PATCH /api/chats/{id} per Plan 02-04 contract.
    // Server enforces expected_version (409 on mismatch — surfaced via
    // OPTIMISTIC_409_TOAST in the caller's onError).
    patch: (id: string, body: ChatPatchBody): Promise<ChatSummary> =>
      patchJson(`/api/chats/${id}`, body),
    delete: (id: string): Promise<void> => deleteRequest(`/api/chats/${id}`),
    search: (
      q: string,
      filters: ChatSearchFilters = {},
    ): Promise<ChatSearchPayload> => getJson(buildSearchUrl(q, filters)),
    // Plan 02-13 — GET /api/chats/retrievers. Returns {milvus, paperqa,
    // hipporag, lazygraph, fused: bool}. Used by the new-chat dialog to
    // render disabled options when a retriever's Python library is not
    // installed on the server (graceful stub-fallback per
    // 02-01-LIBRARY-SPIKE.md).
    availability: (): Promise<RetrieverAvailability> =>
      getJson("/api/chats/retrievers"),
  },
  chunks: {
    get: (parentId: string): Promise<ChunkPayload> =>
      getJson(`/api/chunks/${parentId}`),
  },
  queries: {
    list: (queryString: string): Promise<QueryLogListResponse> =>
      getJson(`/api/queries?${queryString}`),
    get: (queryId: string): Promise<QueryAuditPayload> =>
      getJson(`/api/queries/${queryId}`),
    stage: (
      queryId: string,
      name: string,
    ): Promise<{ name: string; payload: unknown }> =>
      getJson(`/api/queries/${queryId}/stages/${name}`),
    chunk: (queryId: string, filename: string): Promise<string> =>
      getText(`/api/queries/${queryId}/chunks/${filename}`),
  },
  papers: {
    list: (params: URLSearchParams): Promise<PaperListApi> =>
      getJson(`/api/papers?${params.toString()}`),
    detail: (id: string): Promise<PaperDetailApi> =>
      getJson(`/api/papers/${encodeURIComponent(id)}`),
    contentListUrl: (id: string): string =>
      `/api/papers/${encodeURIComponent(id)}/content-list`,
  },
  graph: {
    collections: (): Promise<CollectionInfoApi[]> =>
      getJson("/api/graph/collections"),
    corpusView: (collection: string): Promise<GraphResponseApi> =>
      getJson(`/api/graph/corpus-view?collection=${encodeURIComponent(collection)}`),
    seedSearch: (q: string): Promise<SeedSearchHitApi> =>
      postJson("/api/graph/seed-search", { q }),
    walk: (body: WalkRequest): Promise<GraphResponseApi> =>
      postJson("/api/graph/walk", body),
    expand: (body: ExpandRequest): Promise<GraphResponseApi> =>
      postJson("/api/graph/expand", body),
    inCorpusCheck: (body: InCorpusCheckRequest): Promise<Record<number, InCorpusResultApi>> =>
      postJson("/api/graph/in-corpus-check", body),
    node: (corpusId: number): Promise<NodeDetailResponseApi> =>
      getJson(`/api/graph/node/${corpusId}`),
  },
  ingest: {
    start: (body: IngestRequestApi): Promise<IngestResponseApi> =>
      postJson("/api/ingest", body),
  },
  maps: {
    create: (body: CreateMapBody): Promise<{ id: string }> =>
      postJson("/api/maps", body),
    list: (collection: string): Promise<MapMetaApi[]> =>
      getJson(`/api/maps?collection=${encodeURIComponent(collection)}`),
    get: (id: string): Promise<MapApi> =>
      getJson(`/api/maps/${id}`),
    coverage: (id: string): Promise<MapCoverageApi> =>
      getJson(`/api/maps/${id}/coverage`),
    delete: (id: string): Promise<void> =>
      deleteRequest(`/api/maps/${id}`),
  },
};

export { getJson, postJson, getText };
