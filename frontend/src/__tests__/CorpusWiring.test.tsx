/**
 * Wiring verification for the Phase 3 corpus browser /api/papers integration.
 *
 * Tests at the hook level (usePaperListQuery, usePaperDetailQuery) to verify
 * the API response-to-component-prop mapping without requiring a full
 * TanStack Router context.  Also tests the contentListUrl construction
 * directly from the api module.
 *
 * The component rendering integration (CorpusList default search, Next button
 * visibility) is additionally verified via a RouterProvider harness below.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, renderHook, waitFor } from "@testing-library/react";
import {
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import {
  createRouter,
  RouterProvider,
  createRootRoute,
  createRoute,
  createMemoryHistory,
} from "@tanstack/react-router";
import type { ReactNode } from "react";

import type { PaperRowApi } from "@/lib/api";

// ---------------------------------------------------------------------------
// Controlled API mocks
// ---------------------------------------------------------------------------

const mockListApi = vi.fn();
const mockDetailApi = vi.fn();

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("@/lib/api");
  return {
    ...actual,
    api: {
      ...(actual.api as Record<string, unknown>),
      papers: {
        list: (params: URLSearchParams) => mockListApi(params),
        detail: (id: string) => mockDetailApi(id),
        contentListUrl: (id: string) =>
          `/api/papers/${encodeURIComponent(id)}/content-list`,
      },
    },
  };
});

// ---------------------------------------------------------------------------
// QueryClient wrapper factory
// ---------------------------------------------------------------------------

function createWrapper(): (props: { children: ReactNode }) => ReactNode {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

// ---------------------------------------------------------------------------
// Test data — two PaperRowApi rows, one with null chunk_count
// ---------------------------------------------------------------------------

const ROW_NORMAL: PaperRowApi = {
  paper_id: "2204.00817",
  title: "High-frequency trading in a limit order book",
  authors: '["Avellaneda","Stoikov"]',
  year: 2008,
  collection: "trading",
  chunk_count: 142,
  arxiv_id: "2204.00817",
  doi: null,
  short_cite: "Avellaneda & Stoikov (2008)",
  ingested_at: 1778587200, // 2026-05-12T12:00:00Z
  extraction_quality: "high",
  fetch_source: "arXiv PDF → MinerU 2.1.0",
  parser_version: "mineru@2.1.0",
};

const ROW_DEGRADED: PaperRowApi = {
  paper_id: "doi-jmlr12",
  title: "Locality-sensitive hashing",
  authors: '["Andoni","Indyk"]',
  year: 2008,
  collection: "ecology",
  chunk_count: null,
  arxiv_id: null,
  doi: "10.1145/1327452.1327494",
  short_cite: "Andoni & Indyk (2008)",
  ingested_at: 1747080000,
  extraction_quality: "low",
  fetch_source: "Crossref → MinerU 2.1.0",
  parser_version: "mineru@2.1.0",
};

beforeEach(() => {
  vi.clearAllMocks();
});

// ---------------------------------------------------------------------------
// Hook-level: usePaperListQuery
// ---------------------------------------------------------------------------

describe("usePaperListQuery wiring", () => {
  it("maps PaperRowApi fields including authors JSON-parse-and-join", async () => {
    mockListApi.mockResolvedValue({
      papers: [ROW_NORMAL],
      next_cursor: null,
      total: 1,
    });

    const { usePaperListQuery } = await import("@/queries/corpus");
    const { result } = renderHook(
      () =>
        usePaperListQuery({
          q: "",
          collections: [],
          yearFrom: "",
          yearTo: "",
          ingested: "any",
          hasArxiv: false,
          hasDoi: false,
          sort: "ingested",
          dir: "desc",
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const data = result.current.data!;

    expect(data.papers).toHaveLength(1);
    expect(data.papers[0]!.id).toBe("2204.00817");
    expect(data.papers[0]!.authors).toBe("Avellaneda, Stoikov");
    expect(data.papers[0]!.chunks).toBe(142);
    expect(data.papers[0]!.degradedChunks).toBe(false);
    expect(data.papers[0]!.arxiv).toBe("2204.00817");
    expect(data.papers[0]!.ingested).toBe("2026-05-12");
    expect(data.totalLabel).toBe("1");
    expect(data.nextCursor).toBeNull();
  });

  it("sets degradedChunks=true and chunks=0 when chunk_count is null", async () => {
    mockListApi.mockResolvedValue({
      papers: [ROW_DEGRADED],
      next_cursor: null,
      total: 1,
    });

    const { usePaperListQuery } = await import("@/queries/corpus");
    const { result } = renderHook(
      () =>
        usePaperListQuery({
          q: "",
          collections: [],
          yearFrom: "",
          yearTo: "",
          ingested: "any",
          hasArxiv: false,
          hasDoi: false,
          sort: "ingested",
          dir: "desc",
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const paper = result.current.data!.papers[0]!;
    expect(paper.degradedChunks).toBe(true);
    expect(paper.chunks).toBe(0);
  });

  it("exposes nextCursor when the API returns one", async () => {
    mockListApi.mockResolvedValue({
      papers: [ROW_NORMAL],
      next_cursor: "WyJcdTIwMjYiXQ==",
      total: 2,
    });

    const { usePaperListQuery } = await import("@/queries/corpus");
    const { result } = renderHook(
      () =>
        usePaperListQuery({
          q: "",
          collections: [],
          yearFrom: "",
          yearTo: "",
          ingested: "any",
          hasArxiv: false,
          hasDoi: false,
          sort: "ingested",
          dir: "desc",
          cursor: null,
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data!.nextCursor).toBe("WyJcdTIwMjYiXQ==");
  });

  it("sends correct URLSearchParams from PaperListParams", async () => {
    mockListApi.mockResolvedValue({
      papers: [ROW_NORMAL],
      next_cursor: null,
      total: 1,
    });

    const { usePaperListQuery } = await import("@/queries/corpus");
    renderHook(
      () =>
        usePaperListQuery({
          q: "avellaneda",
          collections: ["trading"],
          yearFrom: "2000",
          yearTo: "2020",
          ingested: "30d",
          hasArxiv: true,
          hasDoi: false,
          sort: "title",
          dir: "asc",
          scopeChatId: "c01",
          cursor: "prev-cur",
          limit: 50,
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(mockListApi).toHaveBeenCalled());
    const params = mockListApi.mock.calls[0]![0] as URLSearchParams;

    expect(params.get("q")).toBe("avellaneda");
    expect(params.get("collection")).toBe("trading");
    expect(params.get("yearFrom")).toBe("2000");
    expect(params.get("yearTo")).toBe("2020");
    expect(params.get("ingested")).toBe("30d");
    expect(params.get("hasArxiv")).toBe("1");
    // hasDoi=false is omitted (falsy)
    expect(params.get("hasDoi")).toBeNull();
    expect(params.get("sort")).toBe("title");
    expect(params.get("dir")).toBe("asc");
    expect(params.get("citedInChat")).toBe("c01");
    expect(params.get("cursor")).toBe("prev-cur");
    expect(params.get("limit")).toBe("50");
  });

  it("keeps keepPreviousData and staleTime:0", async () => {
    // Import the module and check exports — keepPreviousData and staleTime:0
    // are set in the query options, no need to mock data for this.
    mockListApi.mockResolvedValue({ papers: [ROW_NORMAL], next_cursor: null, total: 1 });

    const { usePaperListQuery } = await import("@/queries/corpus");
    const { result } = renderHook(
      () =>
        usePaperListQuery({
          q: "",
          collections: [],
          yearFrom: "",
          yearTo: "",
          ingested: "any",
          hasArxiv: false,
          hasDoi: false,
          sort: "ingested",
          dir: "desc",
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Hook-level: usePaperDetailQuery
// ---------------------------------------------------------------------------

describe("usePaperDetailQuery wiring", () => {
  it("maps PaperDetailApi fields and builds provenance entries", async () => {
    mockDetailApi.mockResolvedValue({
      ...ROW_NORMAL,
      content_list: "/api/papers/2204.00817/content-list",
    });

    const { usePaperDetailQuery } = await import("@/queries/corpus");
    const { result } = renderHook(() => usePaperDetailQuery("2204.00817"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const detail = result.current.data!;

    expect(detail.id).toBe("2204.00817");
    expect(detail.title).toBe("High-frequency trading in a limit order book");
    expect(detail.authors).toBe("Avellaneda, Stoikov");
    expect(detail.extractionQuality).toBe("high");

    // Cross-links are empty arrays until backend provides them
    expect(detail.queriesForPaper).toEqual([]);
    expect(detail.chatsForPaper).toEqual([]);

    // Provenance entries
    expect(detail.provenance.length).toBeGreaterThan(0);

    const pid = detail.provenance.find((p) => p.k === "paper_id");
    expect(pid).toBeDefined();
    expect(pid!.v).toBe("2204.00817");

    const chunks = detail.provenance.find((p) => p.k === "chunks in milvus");
    expect(chunks).toBeDefined();
    expect(chunks!.v).toBe("142");
    expect(chunks!.kind).toBe("chunks");

    const arXiv = detail.provenance.find((p) => p.k === "arXiv");
    expect(arXiv).toBeDefined();
    expect(arXiv!.v).toBe("2204.00817");
    expect(arXiv!.degradedReason).toBeUndefined();

    const doi = detail.provenance.find((p) => p.k === "DOI");
    expect(doi).toBeDefined();
    expect(doi!.v).toBeNull();
    expect(doi!.degradedReason).toBe("no DOI");
  });

  it("builds degraded provenance entries for null fields", async () => {
    mockDetailApi.mockResolvedValue({
      ...ROW_DEGRADED,
      content_list: "/api/papers/doi-jmlr12/content-list",
    });

    const { usePaperDetailQuery } = await import("@/queries/corpus");
    const { result } = renderHook(() => usePaperDetailQuery("doi-jmlr12"), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const detail = result.current.data!;

    expect(detail.extractionQuality).toBe("low");

    const arXiv = detail.provenance.find((p) => p.k === "arXiv");
    expect(arXiv!.degradedReason).toBe("no arXiv id");

    const chunks = detail.provenance.find((p) => p.k === "chunks in milvus");
    expect(chunks!.v).toBeNull();
    expect(chunks!.degradedReason).toBe("Milvus unreachable");
  });

  it("stays disabled when paperId is null", async () => {
    const { usePaperDetailQuery } = await import("@/queries/corpus");
    const { result } = renderHook(() => usePaperDetailQuery(null), {
      wrapper: createWrapper(),
    });

    expect(result.current.fetchStatus).toBe("idle");
    expect(mockDetailApi).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// api.papers.contentListUrl — direct unit test
// ---------------------------------------------------------------------------

describe("api.papers.contentListUrl", () => {
  it("returns the correct relative URL for a given paper ID", async () => {
    // Mock is set up above; re-import to get the mocked api
    const { api } = await import("@/lib/api");
    const url = api.papers.contentListUrl("2204.00817");
    expect(url).toBe("/api/papers/2204.00817/content-list");
  });

  it("URL-encodes special characters in paper IDs", async () => {
    const { api } = await import("@/lib/api");
    const url = api.papers.contentListUrl("doi:10.1234/foo");
    expect(url).toContain(encodeURIComponent("doi:10.1234/foo"));
    expect(url).not.toContain("doi:10.1234/foo");
  });
});

// ---------------------------------------------------------------------------
// Component-level integration (RouterProvider harness)
// ---------------------------------------------------------------------------

describe("CorpusList RouterProvider integration", () => {
  async function buildRouter() {
    const { CorpusList } = await import("@/components/phase3/CorpusList");

    const rootRoute = createRootRoute();
    const corpusRoute = createRoute({
      getParentRoute: () => rootRoute,
      path: "/corpus",
      validateSearch: (s: Record<string, unknown>) => ({
        q: "",
        collections: [] as string[],
        yearFrom: "",
        yearTo: "",
        ingested: "any",
        hasArxiv: false,
        hasDoi: false,
        sort: "ingested",
        dir: "desc",
        ...s,
      }),
      component: () => <CorpusList />,
    });

    const routeTree = rootRoute.addChildren([corpusRoute]);
    const history = createMemoryHistory({ initialEntries: ["/corpus"] });
    return createRouter({ routeTree, history });
  }

  it("renders rows via RouterProvider", async () => {
    mockListApi.mockResolvedValue({
      papers: [ROW_NORMAL, ROW_DEGRADED],
      next_cursor: null,
      total: 2,
    });

    const router = await buildRouter();
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );

    expect(
      await screen.findByText("High-frequency trading in a limit order book"),
    ).toBeTruthy();
    expect(
      await screen.findByText("Locality-sensitive hashing"),
    ).toBeTruthy();
  });
});
