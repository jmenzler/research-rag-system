/**
 * Unit tests for graph and maps empty states (D-11, POLISH-08).
 *
 * Behavioural contract:
 *   1. When hasNoGraph is true (no walk loaded, zero corpus nodes),
 *      GraphPage renders a .empty block with a "search a seed" message.
 *   2. When the maps list is empty, MapsPanel renders a .empty block
 *      with "No maps saved" copy.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

// ---------------------------------------------------------------------------
// Mock heavy dependencies
// ---------------------------------------------------------------------------

vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }),
}));

vi.mock("@tanstack/react-router", () => ({
  useNavigate: vi.fn(() => null),
  createFileRoute: vi.fn(() => ({
    component: (c: unknown) => c,
    validateSearch: (s: unknown) => s,
  })),
}));

vi.mock("@/routes/app/_layout/graph", () => ({
  Route: {
    useSearch: vi.fn(() => ({
      collection: "trading",
      depth: 2,
      direction: "both",
      minCites: 0,
      listOpen: false,
    })),
  },
}));

// useQuery is called twice in GraphPage: once for corpusViewQuery (returns {nodes, edges})
// and once for useInCorpusCheck (returns a Map). Alternates calls via mockReturnValueOnce.
// For the test, we just need a stable sequence so hasNoGraph stays true.
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  const emptyCorpus = { data: { nodes: [], edges: [] }, isLoading: false, isError: false };
  const emptyMap = { data: new Map(), isLoading: false, isError: false };
  const useQueryMock = vi.fn()
    .mockReturnValueOnce(emptyCorpus)
    .mockReturnValue(emptyMap);
  return {
    ...actual,
    useQuery: useQueryMock,
  };
});

vi.mock("@/queries/graph", () => ({
  corpusViewQuery: vi.fn(() => ({ queryKey: ["corpus-view", "trading"], queryFn: vi.fn() })),
  useInCorpusCheck: vi.fn(() => ({ queryKey: ["in-corpus"], queryFn: vi.fn() })),
  useWalkMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useExpandMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useIngestMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useMapsListQuery: vi.fn(() => ({
    data: [],
    isPending: false,
    isError: false,
  })),
  useMapQuery: vi.fn(() => ({ data: null, isPending: false, isError: false })),
  useMapCoverageQuery: vi.fn(() => ({ data: null, isPending: false, isError: false })),
  useCreateMapMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  useDeleteMapMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
}));

vi.mock("@/components/graph/GraphCanvas", () => ({ GraphCanvas: () => null }));
vi.mock("@/components/graph/GraphToolbar", () => ({ GraphToolbar: () => null }));
vi.mock("@/components/graph/GraphLegend", () => ({ GraphLegend: () => null }));
vi.mock("@/components/graph/GraphRail", () => ({ GraphRail: () => null }));
vi.mock("@/components/graph/GraphListPane", () => ({ GraphListPane: () => null }));
vi.mock("@/components/graph/GraphMetadataHeader", () => ({ GraphMetadataHeader: () => null }));
vi.mock("@/components/graph/GraphControlPanel", () => ({ GraphControlPanel: () => null }));

vi.mock("@/state/graphStore", () => ({
  useGraphStore: vi.fn((selector: (s: Record<string, unknown>) => unknown) =>
    selector({
      selectedNodeId: null,
      setSelectedNodeId: vi.fn(),
      focusedNodeId: null,
      listOpen: false,
      ceilingOverride: false,
      setCeilingOverride: vi.fn(),
      addInFlight: vi.fn(),
      overlayNodes: null,
      overlayEdges: null,
      overflowCount: 0,
      setOverlayNodes: vi.fn(),
      setOverlayEdges: vi.fn(),
      setOverflowCount: vi.fn(),
    }),
  ),
}));

vi.mock("@/state/graphDisplayStore", () => ({
  useGraphDisplayStore: vi.fn(() => ({})),
}));

vi.mock("@/lib/graphLabels", () => ({
  deriveShortLabel: vi.fn(() => null),
}));

vi.mock("@/lib/api", () => ({
  api: {
    ingest: { start: vi.fn().mockResolvedValue({ run_ids: [], n_enqueued: 0 }) },
    maps: {
      create: vi.fn().mockResolvedValue({ id: "new-map-id" }),
      list: vi.fn().mockResolvedValue([]),
      get: vi.fn(),
      coverage: vi.fn(),
      delete: vi.fn().mockResolvedValue(undefined),
    },
    graph: { inCorpusCheck: vi.fn() },
  },
}));

vi.mock("@/hooks/useQueueStream", () => ({
  useQueueStream: () => ({
    jobs: {},
    counts: { running: 0, queued: 0 },
    isConnected: false,
  }),
}));

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function renderWithClient(ui: ReactNode): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, enabled: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GraphPage — empty state when no graph loaded (D-11)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a .empty block when hasNoGraph is true (corpus empty, no overlay)", async () => {
    const GraphPage = (await import("@/components/graph/GraphPage")).default;

    const { container } = renderWithClient(
      <div className="p3 h-full">
        <GraphPage />
      </div>,
    );

    const emptyEl = container.querySelector(".empty");
    expect(emptyEl).not.toBeNull();
    expect(emptyEl?.textContent).toMatch(/No graph loaded|search a seed/i);
  });
});

describe("MapsPanel — empty state when maps list is empty (D-11)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a .empty block when mapsListResult.data is empty array", async () => {
    const { MapsPanel } = await import("@/components/graph/MapsPanel");

    const { container } = renderWithClient(
      <div className="p3">
        <MapsPanel
          collection="trading"
          nodes={[]}
          edges={[]}
          onLoadSnapshot={() => undefined}
        />
      </div>,
    );

    const emptyEl = container.querySelector(".empty");
    expect(emptyEl).not.toBeNull();
    expect(emptyEl?.textContent).toMatch(/No maps saved/i);
  });
});
