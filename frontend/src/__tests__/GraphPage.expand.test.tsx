/**
 * Wave-0 RED tests for GraphPage expand non-blocking canvas mount (D-05).
 *
 * Implementation lands in Plan 07-03 (frontend/src/components/graph/GraphPage.tsx).
 *
 * Behavioural contract:
 *   - Test A (RED until Plan 03): expandMutation.isPending === true
 *     → GraphCanvas container stays in the DOM (canvas not replaced by full-canvas spinner)
 *   - Test B (GREEN — existing behavior preserved): walkMutation.isPending === true
 *     → full-canvas role="status" spinner renders
 *
 * RED state is intentional: Test A asserts behavior that requires splitting the single
 * isMutating ternary into separate walk/expand branches (Plan 07-03). Until then,
 * expandMutation.isPending unmounts GraphCanvas (same as walkMutation.isPending).
 */
import { describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

// ---------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------

function renderWithClient(ui: ReactNode): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("@/queries/graph", () => ({
  useWalkMutation: vi.fn(),
  useExpandMutation: vi.fn(),
  useInCorpusCheck: vi.fn(() => ({ data: undefined, isLoading: false })),
  useIngestMutation: vi.fn(() => ({ mutate: vi.fn(), isPending: false })),
  corpusViewQuery: vi.fn(() => ({ queryKey: ["corpus-view"], queryFn: async () => ({ nodes: [], edges: [] }) })),
}));

const GRAPH_STORE_STATE = {
  focusedNodeId: null,
  setFocusedNodeId: vi.fn(),
  clearFocus: vi.fn(),
  selectedNodeId: null,
  setSelectedNodeId: vi.fn(),
  overlayNodes: [] as unknown[],
  overlayEdges: [] as unknown[],
  setOverlayNodes: vi.fn(),
  setOverlayEdges: vi.fn(),
  walkSeedId: null,
  setWalkSeedId: vi.fn(),
  hoveredPathNodeSet: new Set<string>(),
  hoveredPathEdgeSet: new Set<string>(),
  setHoveredPath: vi.fn(),
  clearHoveredPath: vi.fn(),
};

vi.mock("@/state/graphStore", () => ({
  useGraphStore: vi.fn((selector?: (s: typeof GRAPH_STORE_STATE) => unknown) =>
    selector ? selector(GRAPH_STORE_STATE) : GRAPH_STORE_STATE,
  ),
}));

const DISPLAY_SETTINGS = {
  nodeSize: 60,
  linkThick: 50,
  textFade: 40,
  arrowsOn: true,
  centerForce: 50,
  repelForce: 50,
  linkForce: 50,
  linkDistance: 50,
  animate: false,
};

vi.mock("@/state/graphDisplayStore", () => ({
  useGraphDisplayStore: vi.fn(() => DISPLAY_SETTINGS),
}));

vi.mock("@/routes/app/_layout/graph", () => ({
  Route: {
    useSearch: vi.fn(() => ({})),
  },
}));

vi.mock("@tanstack/react-router", () => ({
  useNavigate: vi.fn(() => vi.fn()),
}));

vi.mock("@/components/graph/GraphCanvas", () => ({
  GraphCanvas: ({ "data-testid": testId }: { "data-testid"?: string }) => (
    <div data-testid={testId ?? "graph-canvas-mock"}>GraphCanvas</div>
  ),
}));

vi.mock("@/components/graph/GraphToolbar", () => ({
  GraphToolbar: () => <div data-testid="toolbar-mock">GraphToolbar</div>,
}));

vi.mock("@/components/graph/GraphLegend", () => ({
  GraphLegend: () => <div>GraphLegend</div>,
}));

vi.mock("@/components/graph/GraphRail", () => ({
  GraphRail: () => <div>GraphRail</div>,
}));

vi.mock("@/components/graph/GraphListPane", () => ({
  GraphListPane: () => <div>GraphListPane</div>,
}));

vi.mock("@/components/graph/GraphMetadataHeader", () => ({
  GraphMetadataHeader: () => <div>GraphMetadataHeader</div>,
}));

vi.mock("@/components/graph/GraphControlPanel", () => ({
  GraphControlPanel: () => <div>GraphControlPanel</div>,
}));

vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));

vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return {
    ...actual,
    useQuery: vi.fn((queryDef: unknown) => {
      const key = Array.isArray((queryDef as { queryKey?: unknown[] }).queryKey)
        ? String((queryDef as { queryKey: unknown[] }).queryKey[0])
        : "";
      if (key.includes("corpus-view")) {
        return { data: { nodes: [], edges: [] }, isLoading: false, isSuccess: true };
      }
      // in-corpus check returns a Map
      return { data: new Map(), isLoading: false, isSuccess: true };
    }),
  };
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GraphPage — expand non-blocking canvas mount (D-05)", () => {
  it("Test A (RED until Plan 03): GraphCanvas stays mounted when expandMutation.isPending", async () => {
    const { useWalkMutation, useExpandMutation } = await import("@/queries/graph");
    const mod = await import("@/components/graph/GraphPage").catch((e) => {
      throw new Error(`GraphPage import failed: ${String(e)}`);
    });
    const GraphPage = (mod as { default: React.ComponentType }).default;

    vi.mocked(useWalkMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      isSuccess: false,
      isError: false,
      data: undefined,
      error: null,
      reset: vi.fn(),
    } as unknown as ReturnType<typeof useWalkMutation>);

    vi.mocked(useExpandMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: true,
      isSuccess: false,
      isError: false,
      data: undefined,
      error: null,
      reset: vi.fn(),
    } as unknown as ReturnType<typeof useExpandMutation>);

    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    try {
      renderWithClient(<GraphPage />);

      await waitFor(() => {
        const canvas = screen.queryByTestId("graph-canvas-mock");
        expect(canvas).not.toBeNull();
      });
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("Test B (existing behavior): walkMutation.isPending renders full-canvas role=status spinner", async () => {
    const { useWalkMutation, useExpandMutation } = await import("@/queries/graph");
    const mod = await import("@/components/graph/GraphPage").catch((e) => {
      throw new Error(`GraphPage import failed: ${String(e)}`);
    });
    const GraphPage = (mod as { default: React.ComponentType }).default;

    vi.mocked(useWalkMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: true,
      isSuccess: false,
      isError: false,
      data: undefined,
      error: null,
      reset: vi.fn(),
    } as unknown as ReturnType<typeof useWalkMutation>);

    vi.mocked(useExpandMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      isSuccess: false,
      isError: false,
      data: undefined,
      error: null,
      reset: vi.fn(),
    } as unknown as ReturnType<typeof useExpandMutation>);

    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    try {
      renderWithClient(<GraphPage />);

      await waitFor(() => {
        const spinner = screen.queryByRole("status", {
          name: /loading citation graph/i,
        });
        expect(spinner).not.toBeNull();
      });
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
