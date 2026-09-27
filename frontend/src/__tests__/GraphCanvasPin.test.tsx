/**
 * RED — GREEN after 04-03.
 *
 * GRAPH-09 pinning contract: existing node positions must be preserved when new
 * nodes are added via an incremental expand. Simulates a user-pinned layout where
 * node "A" is at a known {x, y} and a subsequent expand adds node "B" — "A" must
 * not move.
 *
 * Observable behavior holds under both spike verdicts:
 *   - GO (SigmaContainer): graphology graph.setNodeAttribute("A", "fixed", true)
 *   - FALLBACK (direct sigma.js): node position locked via imperative sigma layout API
 */
import { describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
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
// Test data
// ---------------------------------------------------------------------------

type GraphNode = {
  corpus_id: number;
  title: string;
  x: number;
  y: number;
  fixed?: boolean;
};

type GraphEdge = {
  source: number;
  target: number;
};

const INITIAL_NODES: GraphNode[] = [
  { corpus_id: 1, title: "Node A", x: 100, y: 200, fixed: true },
  { corpus_id: 2, title: "Node B", x: 300, y: 400 },
];

const INITIAL_EDGES: GraphEdge[] = [{ source: 1, target: 2 }];

const EXPANDED_NODES: GraphNode[] = [
  ...INITIAL_NODES,
  { corpus_id: 3, title: "Node C", x: 0, y: 0 },
];

const EXPANDED_EDGES: GraphEdge[] = [
  ...INITIAL_EDGES,
  { source: 2, target: 3 },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GraphCanvas — incremental expand preserves existing node positions (GRAPH-09)", () => {
  it("node A position is unchanged after expand adds node C", async () => {
    // GraphCanvas import will fail (RED state) until 04-03 ships the component.
    const { GraphCanvas } = await import("@/components/graph/GraphCanvas").catch(
      (e) => {
        throw new Error(
          `GraphCanvas not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    // Mock sigma/WebGL — jsdom has no WebGL
    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    // Use a ref callback to read the graphology graph's node positions
    let getPositions: (() => Record<number, { x: number; y: number }>) | null = null;

    const { rerender } = renderWithClient(
      <GraphCanvas
        nodes={INITIAL_NODES}
        edges={INITIAL_EDGES}
        onGraphReady={(graph: { getNodeAttribute: (id: string, attr: string) => number }) => {
          getPositions = () => {
            const pos: Record<number, { x: number; y: number }> = {};
            EXPANDED_NODES.forEach((n) => {
              pos[n.corpus_id] = {
                x: graph.getNodeAttribute(String(n.corpus_id), "x"),
                y: graph.getNodeAttribute(String(n.corpus_id), "y"),
              };
            });
            return pos;
          };
        }}
      />,
    );

    // Wait until onGraphReady has fired (getPositions set). Polling instead of a
    // fixed setTimeout removes a race: under full-suite parallel load the mount
    // effect that calls onGraphReady can flush after a fixed 50ms window, leaving
    // getPositions null → nodeABefore undefined. waitFor retries until ready.
    await waitFor(() => {
      expect(getPositions).not.toBeNull();
    });

    const beforeExpand = (getPositions as (() => Record<number, { x: number; y: number }>) | null)?.();
    const nodeABefore = beforeExpand?.[1];
    expect(nodeABefore).toBeDefined();

    // Trigger expand: add node C
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <GraphCanvas
          nodes={EXPANDED_NODES}
          edges={EXPANDED_EDGES}
          onGraphReady={(graph: { getNodeAttribute: (id: string, attr: string) => number }) => {
            getPositions = () => {
              const pos: Record<number, { x: number; y: number }> = {};
              EXPANDED_NODES.forEach((n) => {
                pos[n.corpus_id] = {
                  x: graph.getNodeAttribute(String(n.corpus_id), "x"),
                  y: graph.getNodeAttribute(String(n.corpus_id), "y"),
                };
              });
              return pos;
            };
          }}
        />
      </QueryClientProvider>,
    );

    // Wait until the expanded graph is ready (node C indexed). Poll instead of a
    // fixed timeout for the same race-free reason as above.
    await waitFor(() => {
      const pos = (getPositions as (() => Record<number, { x: number; y: number }>) | null)?.();
      expect(pos?.[3]).toBeDefined();
    });

    // Node A must remain at its original position (within floating-point epsilon)
    const afterExpand = (getPositions as (() => Record<number, { x: number; y: number }>) | null)?.();
    const nodeAAfter = afterExpand?.[1];

    expect(nodeAAfter).toBeDefined();
    expect(nodeAAfter!.x).toBeCloseTo(nodeABefore!.x, 1);
    expect(nodeAAfter!.y).toBeCloseTo(nodeABefore!.y, 1);

    vi.unstubAllGlobals();
  });

  it("GraphCanvas renders without crashing with empty graph", async () => {
    const { GraphCanvas } = await import("@/components/graph/GraphCanvas").catch(
      (e) => {
        throw new Error(
          `GraphCanvas not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    expect(() =>
      renderWithClient(<GraphCanvas nodes={[]} edges={[]} />),
    ).not.toThrow();

    vi.unstubAllGlobals();
  });
});
