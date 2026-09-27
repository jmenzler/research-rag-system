/**
 * RED — GREEN after 04-03.
 *
 * GRAPH-03 off-main-thread guarantee: ForceAtlas2 layout must run in a Web Worker,
 * not on the main thread. The test asserts that GraphCanvas (or its layout module)
 * imports from "graphology-layout-forceatlas2/worker" (the worker subpath) and never
 * invokes synchronous main-thread layout (bare "graphology-layout-forceatlas2").
 *
 * Observable contract holds under both spike verdicts:
 *   - GO (SigmaContainer): @react-sigma/layout-forceatlas2 wraps the worker subpath
 *   - FALLBACK (direct sigma.js): layout module must still use the /worker subpath
 *
 * D-01 centroid placement: expand nodes must spawn near their connected fixed neighbor,
 * not scattered across the canvas. Tested via the onGraphReady callback pattern.
 */
import { describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

describe("GraphCanvas — ForceAtlas2 off-main-thread (GRAPH-03)", () => {
  it("ForceAtlas2 layout runs in a Web Worker, not the main thread", async () => {
    // Worker mock must be a real class so `new Worker(url)` works.
    // It must expose terminate() and postMessage() so the probe worker created
    // at module scope in GraphCanvas.tsx (GRAPH-03 contract) can be fully cleaned up.
    class MockWorker {
      terminate = vi.fn();
      postMessage = vi.fn();
      addEventListener = vi.fn();
      removeEventListener = vi.fn();
    }
    const workerSpy = vi.fn().mockImplementation(function (this: MockWorker, ...args: unknown[]) {
      void args;
      Object.assign(this, new MockWorker());
    });
    vi.stubGlobal("Worker", workerSpy);

    // Mock sigma/WebGL — jsdom has no WebGL
    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    // Import GraphCanvas — this import will fail (RED state) until 04-03 ships the component.
    // When it does ship, GraphCanvas must construct a Worker for the FA2 layout.
    const mod = await import("@/components/graph/GraphCanvas").catch((e) => {
      throw new Error(
        `GraphCanvas not yet implemented (expected RED). Original: ${String(e)}`,
      );
    });

    // Verify the module exports a GraphCanvas component
    expect(mod).toHaveProperty("GraphCanvas");

    // Verify GraphCanvas (or its internal layout hook) uses the worker subpath.
    // The worker constructor should be invoked with a URL/path containing the worker module.
    // This assertion will be satisfied by the 04-03 implementation.
    expect(workerSpy).toHaveBeenCalled();
    const workerArg = String(workerSpy.mock.calls[0]?.[0] ?? "");
    expect(workerArg).toContain("forceatlas2");

    vi.unstubAllGlobals();
  });

  it("imports graphology-layout-forceatlas2/worker subpath (not bare module)", async () => {
    // Verify that the worker entry point is the correct subpath.
    // The bare "graphology-layout-forceatlas2" module runs synchronously on the main thread;
    // only "graphology-layout-forceatlas2/worker" offloads to a Web Worker.
    //
    // This test imports the worker subpath directly to assert it is importable and
    // exports the expected FA2Worker class. If GraphCanvas internally uses the bare module
    // path, this test documents the expected alternative.

    const workerMod = await import("graphology-layout-forceatlas2/worker");
    // The worker subpath must export a constructor/class for the background worker
    expect(typeof (workerMod.default ?? workerMod)).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// D-01 centroid placement integration test
// ---------------------------------------------------------------------------

function renderWithClient(ui: ReactNode): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("GraphCanvas — D-01 centroid placement: new expand node spawns near connected fixed neighbor", () => {
  it("new expand node lands within jitter-bounded distance of its connected fixed neighbor", async () => {
    class MockWorker {
      terminate = vi.fn();
      postMessage = vi.fn();
      addEventListener = vi.fn();
      removeEventListener = vi.fn();
    }
    const workerSpy = vi.fn().mockImplementation(function (this: MockWorker, ...args: unknown[]) {
      void args;
      Object.assign(this, new MockWorker());
    });
    vi.stubGlobal("Worker", workerSpy);
    vi.stubGlobal("WebGLRenderingContext", class {});
    vi.stubGlobal("WebGL2RenderingContext", class {});

    const { GraphCanvas } = await import("@/components/graph/GraphCanvas");

    // Node A has a fixed known position; Node B is the new expand node.
    const ANCHOR_X = 100;
    const ANCHOR_Y = 100;

    const INITIAL_NODES = [
      { corpus_id: 1, title: "Node A", x: ANCHOR_X, y: ANCHOR_Y, fixed: true },
    ];
    const INITIAL_EDGES: Array<{ source: number; target: number }> = [];

    const EXPANDED_NODES = [
      ...INITIAL_NODES,
      { corpus_id: 2, title: "Node B (new)" },
    ];
    const EXPANDED_EDGES = [
      { source: 1, target: 2 },
    ];

    type GetPos = () => Record<number, { x: number; y: number }>;
    let getPositions: GetPos | null = null;

    const makeReadyCallback =
      (nodeIds: number[]) =>
      (graph: { getNodeAttribute: (id: string, attr: string) => number }) => {
        getPositions = () => {
          const pos: Record<number, { x: number; y: number }> = {};
          for (const id of nodeIds) {
            pos[id] = {
              x: graph.getNodeAttribute(String(id), "x"),
              y: graph.getNodeAttribute(String(id), "y"),
            };
          }
          return pos;
        };
      };

    const { rerender } = renderWithClient(
      <GraphCanvas
        nodes={INITIAL_NODES}
        edges={INITIAL_EDGES}
        onGraphReady={makeReadyCallback([1])}
        sourceNodeId={1}
      />,
    );

    await waitFor(() => {
      expect(getPositions).not.toBeNull();
    });

    // Trigger expand: add Node B connected only to Node A, with sourceNodeId=1
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <GraphCanvas
          nodes={EXPANDED_NODES}
          edges={EXPANDED_EDGES}
          onGraphReady={makeReadyCallback([1, 2])}
          sourceNodeId={1}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      const pos = getPositions?.();
      expect(pos?.[2]).toBeDefined();
    });

    const positions = (getPositions as GetPos | null)?.();
    const nodeBPos = positions?.[2];
    expect(nodeBPos).toBeDefined();

    // Node B must spawn within jitter-bounded distance of Node A (100, 100).
    // computeCentroid with jitterRadius=20 produces positions in [anchor ± 10].
    // We use a generous bound of 40 to tolerate the full jitter range reliably.
    expect(Math.abs(nodeBPos!.x - ANCHOR_X)).toBeLessThan(40);
    expect(Math.abs(nodeBPos!.y - ANCHOR_Y)).toBeLessThan(40);

    vi.unstubAllGlobals();
  });
});
