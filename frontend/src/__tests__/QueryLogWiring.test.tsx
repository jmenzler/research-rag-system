/**
 * Phase 3 Plan 05 — Query Log wiring tests.
 *
 * Tests that the real GET /api/queries endpoint responses are correctly
 * mapped to QueryLogRow / QueryDetail shapes and that URLSearchParams are
 * built correctly from QueryLogListParams.
 *
 * These tests mirror the CorpusWiring.test.tsx pattern established in Plan 04.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

import type { QueryLogListParams } from "@/queries/queries";

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

const SAMPLE_API_RESPONSE = {
  queries: [
    {
      query_id: "20260519T211408Z_a1b2c3d4",
      ts_started: "2026-05-19T21:14:08.866354+00:00",
      query: "GLFT vs Avellaneda-Stoikov spread derivation",
      retriever: "milvus",
      model: "gemini-3.1-flash",
      tokens: 18204,
      cost_usd: 0.011,
      latency_ms: 3214,
      outcome: "ok",
      chat_id: null,
    },
    {
      query_id: "20260518T114703Z_c3d4e5f6",
      ts_started: "2026-05-18T11:47:03.000000+00:00",
      query: "Optimal quoting with terminal inventory penalty",
      retriever: null,
      model: "gemini-3.1-pro",
      tokens: 41330,
      cost_usd: 0.031,
      latency_ms: 8112,
      outcome: "retry",
      chat_id: null,
    },
  ],
  next_cursor: "2026-05-19T21:14:08.866354+00:00|20260519T211408Z_a1b2c3d4",
  total: 42,
};

const SAMPLE_DETAIL_RESPONSE = {
  query_id: "20260519T211408Z_a1b2c3d4",
  meta: {
    query: "GLFT vs Avellaneda-Stoikov spread derivation",
    ts_started: "2026-05-19T21:14:08.866354+00:00",
    config: { gen_model: "gemini-3.1-flash" },
    outcome: { crag_retried: false, synthesis_skipped: false, n_chunks_to_synth: 8 },
    totals: {
      cost_usd: 0.011,
      total_latency_ms: 3214,
      retrieval_latency_ms: 880,
      synthesis_latency_ms: 1180,
      n_llm_calls: 3,
      tokens: { total: 18204 },
    },
  },
  report_md: "# Answer\nThe GLFT framework recovers the Avellaneda–Stoikov optimal spread.\n\n[1] See Guéant et al. 2013.",
  stages: ["01_decompose.json", "03_milvus.json", "04_rerank.json"],
  chunks: ["01_a3f1.txt", "02_b8c2.txt"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stubFetch(table: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      const key = Object.keys(table).find((k) => url.includes(k));
      if (key === undefined) {
        return Promise.resolve(new Response("not stubbed", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify(table[key]), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }),
  );
}

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

function renderWithClient(ui: ReactNode): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("QueryLogWiring — list endpoint mapping", () => {
  it("maps API row to QueryLogRow with default retriever when null", async () => {
    stubFetch({ "/api/queries?": SAMPLE_API_RESPONSE });

    const { useQueryLogListQuery } = await import("@/queries/queries");
    renderWithClient(
      <div data-testid="wrapper">
        <ListHarness hook={useQueryLogListQuery} />
      </div>,
    );

    await vi.waitFor(
      () => {
        const el = screen.getByTestId("list-data");
        expect(el.textContent).not.toBe("loading");
      },
      { timeout: 3000 },
    );

    const data = JSON.parse(screen.getByTestId("list-data").textContent ?? "null");
    expect(data).not.toBeNull();
    expect(data.queries).toHaveLength(2);

    // First row: retriever was "milvus" in the API response.
    expect(data.queries[0].retriever).toBe("milvus");
    expect(data.queries[0].model).toBe("gemini-3.1-flash");
    expect(data.queries[0].tokens).toBe(18204);
    expect(data.queries[0].cost).toBe(0.011);
    expect(data.queries[0].latencyMs).toBe(3214);
    expect(data.queries[0].outcome).toBe("ok");
    expect(data.queries[0].chat).toBeNull();

    // Second row: retriever was null in API => defaults to "milvus".
    expect(data.queries[1].retriever).toBe("milvus");
    expect(data.queries[1].outcome).toBe("retry");
  });

  it("passes through totalLabel and nextCursor", async () => {
    stubFetch({ "/api/queries?": SAMPLE_API_RESPONSE });

    const { useQueryLogListQuery } = await import("@/queries/queries");
    renderWithClient(
      <div data-testid="wrapper">
        <ListHarness hook={useQueryLogListQuery} />
      </div>,
    );

    await vi.waitFor(
      () => {
        const el = screen.getByTestId("list-data");
        expect(el.textContent).not.toBe("loading");
      },
      { timeout: 3000 },
    );

    const data = JSON.parse(screen.getByTestId("list-data").textContent ?? "null");
    expect(data.totalLabel).toBe("42 queries");
    expect(data.nextCursor).toBe("2026-05-19T21:14:08.866354+00:00|20260519T211408Z_a1b2c3d4");
  });
});

describe("QueryLogWiring — detail endpoint mapping", () => {
  it("maps GET /api/queries/{qid} response to QueryDetail shape", async () => {
    stubFetch({ "/api/queries/20260519T211408Z_a1b2c3d4": SAMPLE_DETAIL_RESPONSE });

    const { useQueryDetailQuery } = await import("@/queries/queries");
    renderWithClient(
      <div data-testid="wrapper">
        <DetailHarness hook={useQueryDetailQuery} queryId="20260519T211408Z_a1b2c3d4" />
      </div>,
    );

    await vi.waitFor(
      () => {
        const el = screen.getByTestId("detail-data");
        expect(el.textContent).not.toBe("loading");
      },
      { timeout: 3000 },
    );

    const data = JSON.parse(screen.getByTestId("detail-data").textContent ?? "null");

    expect(data.row.id).toBe("20260519T211408Z_a1b2c3d4");
    expect(data.row.query).toBe("GLFT vs Avellaneda-Stoikov spread derivation");
    expect(data.row.tokens).toBe(18204);
    expect(data.row.cost).toBe(0.011);
    expect(data.row.latencyMs).toBe(3214);
    expect(data.row.outcome).toBe("ok");

    // KPI subtext derived from totals.
    expect(data.kpi.tokensSub).toContain("18,204 total");
    expect(data.kpi.costSub).toContain("$0.0110");
    expect(data.kpi.latencySub).toContain("·");

    // report — raw markdown in a single block.
    expect(data.report).toHaveLength(1);
    expect(data.report[0].heading).toBe("Answer");
    expect(data.report[0].parts[0].text).toContain("GLFT framework");

    // stages — one per filename.
    expect(data.stages).toHaveLength(3);
    expect(data.stages[0].n).toBe("01");
    expect(data.stages[0].name).toBe("decompose");
    expect(data.stages[0].state).toBe("run");
    expect(data.stages[0].payload).toBeUndefined();

    // chunks — one per filename, empty metadata.
    expect(data.chunks).toHaveLength(2);
    expect(data.chunks[0].f).toBe("01_a3f1.txt");
    expect(data.chunks[0].cite).toBe("");
    expect(data.chunks[0].paperId).toBe("");
    expect(data.chunks[0].score).toBe(0);
  });

  it("derives outcome 'skipped' from synthesis_skipped meta", async () => {
    stubFetch({
      "/api/queries/skipped-q": {
        query_id: "skipped-q",
        meta: {
          query: "skipped query",
          config: {},
          outcome: { synthesis_skipped: true, n_chunks_to_synth: 0 },
          totals: { cost_usd: 0, total_latency_ms: 100, tokens: { total: 0 } },
        },
        report_md: "",
        stages: [],
        chunks: [],
      },
    });

    const { useQueryDetailQuery } = await import("@/queries/queries");
    renderWithClient(
      <div data-testid="wrapper">
        <DetailHarness hook={useQueryDetailQuery} queryId="skipped-q" />
      </div>,
    );

    await vi.waitFor(
      () => {
        const el = screen.getByTestId("detail-data");
        expect(el.textContent).not.toBe("loading");
      },
      { timeout: 3000 },
    );

    const data = JSON.parse(screen.getByTestId("detail-data").textContent ?? "null");
    expect(data.row.outcome).toBe("skipped");
  });
});

describe("QueryLogWiring — URLSearchParams construction", () => {
  it("includes facet params in the API query string", async () => {
    const fetchSpy = vi.fn((_url: string) =>
      Promise.resolve(
        new Response(
          JSON.stringify({ queries: [], next_cursor: null, total: 0 }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const { useQueryLogListQuery } = await import("@/queries/queries");

    renderWithClient(
      <div data-testid="wrapper">
        <ListHarness
          hook={useQueryLogListQuery}
          overrides={{
            retrievers: ["milvus", "paperqa"],
            outcome: ["ok", "failed"],
            costMin: "0.01",
            latencyMin: "5",
            sort: "cost",
            dir: "asc",
            cursor: "2026-05-19T21:14:08Z|qid",
            limit: 50,
          }}
        />
      </div>,
    );

    // Wait for fetch to fire.
    await vi.waitFor(() => {
      expect(fetchSpy).toHaveBeenCalled();
    });

    const callUrl = String(fetchSpy.mock.calls[0]?.[0] ?? "");
    expect(callUrl).toContain("retriever=milvus%2Cpaperqa");
    expect(callUrl).toContain("outcome=ok%2Cfailed");
    expect(callUrl).toContain("costMin=0.01");
    expect(callUrl).toContain("latencyMin=5");
    expect(callUrl).toContain("sort=cost");
    expect(callUrl).toContain("dir=asc");
    expect(callUrl).toContain("cursor=2026-05-19T21%3A14%3A08Z%7Cqid");
    expect(callUrl).toContain("limit=50");
  });

  it("omits empty filters from the query string", async () => {
    const fetchSpy = vi.fn((_url: string) =>
      Promise.resolve(
        new Response(
          JSON.stringify({ queries: [], next_cursor: null, total: 0 }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );
    vi.stubGlobal("fetch", fetchSpy);

    const { useQueryLogListQuery } = await import("@/queries/queries");

    renderWithClient(
      <div data-testid="wrapper">
        <ListHarness
          hook={useQueryLogListQuery}
          overrides={{
            retrievers: [],
            outcome: [],
            costMin: "",
            latencyMin: "",
            sort: "ts",
            dir: "desc",
          }}
        />
      </div>,
    );

    await vi.waitFor(() => {
      expect(fetchSpy).toHaveBeenCalled();
    });

    const callUrl = String(fetchSpy.mock.calls[0]?.[0] ?? "");
    expect(callUrl).not.toContain("retriever=");
    expect(callUrl).not.toContain("outcome=");
    expect(callUrl).not.toContain("costMin=");
    expect(callUrl).not.toContain("latencyMin=");
    expect(callUrl).toContain("sort=ts");
    expect(callUrl).toContain("dir=desc");
  });
});

// ---------------------------------------------------------------------------
// Test harness components
// ---------------------------------------------------------------------------

function ListHarness({
  hook: useHook,
  overrides,
}: {
  hook: (params: QueryLogListParams) => { data: unknown };
  overrides?: Partial<QueryLogListParams>;
}): ReactNode {
  const defaults: QueryLogListParams = {
    retrievers: [],
    outcome: [],
    costMin: "",
    latencyMin: "",
    sort: "ts",
    dir: "desc",
    ...overrides,
  };
  const result = useHook(defaults);
  return (
    <span data-testid="list-data">
      {result.data !== undefined ? JSON.stringify(result.data) : "loading"}
    </span>
  );
}

function DetailHarness({
  hook: useHook,
  queryId,
}: {
  hook: (id: string | null) => { data: unknown };
  queryId: string;
}): ReactNode {
  const result = useHook(queryId);
  return (
    <span data-testid="detail-data">
      {result.data !== undefined ? JSON.stringify(result.data) : "loading"}
    </span>
  );
}
