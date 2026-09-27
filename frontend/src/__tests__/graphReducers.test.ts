/**
 * Wave-0 tests for buildNodeReducer + buildEdgeReducer.
 *
 * D-04 / D-05 base cases: implemented in Plan 04.1-01.
 * D-03 path-set + D-04 dim-strength: RED contracts for Plans 02/06.
 *   These cases assert behavior that ships in Plan 02 (D-04 hex suffix + size shrink)
 *   and Plan 06 (D-03 path-set arguments). They WILL fail until those plans land.
 *
 * Behavioural contract:
 *   - buildNodeReducer: non-focused/non-neighbor node returns label:"" and color ending in "1F"
 *   - buildNodeReducer: dimmed node size < data.size * sizeFactor (size-shrink)
 *   - buildNodeReducer: focused/neighbor node returns forceLabel:true and full color
 *   - buildNodeReducer: size scaled by (0.65 + nodeSize/100)
 *   - buildEdgeReducer: incident-to-focused edge → color "#3B82F6"
 *   - buildEdgeReducer: non-incident when focus active → dimmed color
 *   - buildEdgeReducer: null focus → default color "#475569"
 *
 *   D-03 path-set (buildNodeReducer 5th arg, buildEdgeReducer trailing arg):
 *   - path node (in pathNodeSet, focusedId null) → full color + forceLabel:true
 *   - off-path node (focusedId null, pathNodeSet non-empty) → color ends /1F$/, label ""
 *   - focusedId set → ego-highlight wins, pathNodeSet ignored
 *   - path edge (key in pathEdgeSet) → color "#3B82F6"
 *   - non-path edge (focusedId null, pathEdgeSet non-empty) → dimmed
 */
import { describe, expect, it } from "vitest";

interface NodeDisplayData {
  label: string;
  color: string;
  size: number;
  forceLabel?: boolean;
  [key: string]: unknown;
}

interface EdgeDisplayData {
  color: string;
  size: number;
  [key: string]: unknown;
}

type NodeReducer = (node: string, data: NodeDisplayData) => Partial<NodeDisplayData>;
type EdgeReducer = (edge: string, data: EdgeDisplayData) => Partial<EdgeDisplayData>;

type BuildNodeReducer = (
  focusedId: string | null,
  neighborIds: Set<string>,
  selectedId: string | null,
  display: { nodeSize: number },
  pathNodeSet?: Set<string>,
) => NodeReducer;

type BuildEdgeReducer = (
  focusedId: string | null,
  neighborIds: Set<string>,
  display: { linkThick: number },
  edgeEndpoints: (edge: string) => { source: string; target: string },
  isLayoutRunning?: () => boolean,
  pathEdgeSet?: Set<string>,
) => EdgeReducer;

describe("buildNodeReducer (D-04, D-05) — Wave 0 pure-function tests", () => {
  it("dims non-focused/non-neighbor node: color ends in '1F', label empty, size shrunk (D-04 RED)", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const reducer = mod.buildNodeReducer("node-A", new Set(["node-A"]), null, {
      nodeSize: 60,
    });
    const result = reducer("node-B", { label: "NodeB", color: "#3B82F6", size: 10 });
    expect(result.label).toBe("");
    expect(result.color).toMatch(/1F$/);
    // D-04 size-shrink: dimmed size must be less than full-size-factored size
    const sizeFactor = 0.65 + 60 / 100;
    expect(result.size).toBeLessThan(10 * sizeFactor);
  });

  it("focused node: forceLabel is true and color unchanged", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const reducer = mod.buildNodeReducer("node-A", new Set(["node-B"]), null, {
      nodeSize: 60,
    });
    const result = reducer("node-A", { label: "NodeA", color: "#3B82F6", size: 10 });
    expect(result.forceLabel).toBe(true);
    expect(result.label).not.toBe("");
  });

  it("neighbor node: forceLabel is true", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const reducer = mod.buildNodeReducer("node-A", new Set(["node-B"]), null, {
      nodeSize: 60,
    });
    const result = reducer("node-B", { label: "NodeB", color: "#8B5CF6", size: 12 });
    expect(result.forceLabel).toBe(true);
  });

  it("size is scaled by (0.65 + nodeSize/100)", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const reducer = mod.buildNodeReducer(null, new Set(), null, { nodeSize: 60 });
    const result = reducer("node-A", { label: "NodeA", color: "#3B82F6", size: 10 });
    const expectedFactor = 0.65 + 60 / 100;
    expect(result.size).toBeCloseTo(10 * expectedFactor, 5);
  });

  it("with no focus active: all nodes render with full color", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const reducer = mod.buildNodeReducer(null, new Set(), null, { nodeSize: 50 });
    const result = reducer("node-X", { label: "X", color: "#3B82F6", size: 8 });
    expect(result.color).toBe("#3B82F6");
    expect(result.label).not.toBe("");
  });
});

describe("buildEdgeReducer (D-04) — Wave 0 pure-function tests", () => {
  const endpoints = (edge: string) => {
    const [source, target] = edge.split("->");
    return { source: source ?? "", target: target ?? "" };
  };

  it("incident edge returns primary blue #3B82F6", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const reducer = mod.buildEdgeReducer("node-A", new Set(["node-B"]), { linkThick: 50 }, endpoints);
    const result = reducer("node-A->node-B", { color: "#475569", size: 1 });
    expect(result.color).toBe("#3B82F6");
  });

  it("non-incident edge when focus active returns dimmed color", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const reducer = mod.buildEdgeReducer("node-A", new Set(["node-B"]), { linkThick: 50 }, endpoints);
    const result = reducer("node-C->node-D", { color: "#475569", size: 1 });
    expect(result.color).not.toBe("#3B82F6");
    expect(result.color).not.toBe("#475569");
  });

  it("no focus active: returns default edge color #475569", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const reducer = mod.buildEdgeReducer(null, new Set(), { linkThick: 50 }, endpoints);
    const result = reducer("node-A->node-B", { color: "#475569", size: 1 });
    expect(result.color).toBe("#475569");
  });

  // Perf: a real walk returns ~500 nodes / ~10k edges. Rendering + re-reducing
  // every edge each frame across the FA2 reheat pegs the main thread + GPU
  // (the freeze). While layout is running, edges must be hidden WITHOUT the
  // per-edge endpoint resolution that dominates the reducer cost.
  it("layout running: hides the edge and skips endpoint resolution", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    let resolverCalls = 0;
    const spyEndpoints = (edge: string) => {
      resolverCalls += 1;
      return endpoints(edge);
    };
    const reducer = mod.buildEdgeReducer(
      "node-A",
      new Set(["node-B"]),
      { linkThick: 50 },
      spyEndpoints,
      () => true,
    );
    const result = reducer("node-A->node-B", { color: "#475569", size: 1 });
    expect(result.hidden).toBe(true);
    expect(resolverCalls).toBe(0);
  });

  it("layout stopped: edges render normally (not hidden)", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const reducer = mod.buildEdgeReducer(
      null,
      new Set(),
      { linkThick: 50 },
      endpoints,
      () => false,
    );
    const result = reducer("node-A->node-B", { color: "#475569", size: 1 });
    expect(result.hidden).toBeFalsy();
    expect(result.color).toBe("#475569");
  });
});

// ---------------------------------------------------------------------------
// D-03 RED contracts — path-set behavior (implemented in Plans 06)
// These tests WILL fail until Plan 06 extends buildNodeReducer/buildEdgeReducer
// with pathNodeSet/pathEdgeSet parameters. RED state is intentional Wave-0.
// ---------------------------------------------------------------------------

describe("buildNodeReducer — path-set (D-03)", () => {
  it("path node renders at full color + forceLabel when pathNodeSet provided and no focus", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const pathNodeSet = new Set(["node-A", "node-B"]);
    const reducer = mod.buildNodeReducer(null, new Set(), null, { nodeSize: 60 }, pathNodeSet);
    const result = reducer("node-A", { label: "NodeA", color: "#3B82F6", size: 10 });
    expect(result.forceLabel).toBe(true);
    expect(result.color).toBe("#3B82F6");
  });

  it("off-path node dims when pathNodeSet provided and no focus", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    const pathNodeSet = new Set(["node-A", "node-B"]);
    const reducer = mod.buildNodeReducer(null, new Set(), null, { nodeSize: 60 }, pathNodeSet);
    const result = reducer("node-C", { label: "NodeC", color: "#3B82F6", size: 10 });
    expect(result.label).toBe("");
    expect(result.color).toMatch(/1F$/);
  });

  it("ego-focus takes precedence over pathNodeSet when focusedId is set", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildNodeReducer: BuildNodeReducer;
    };
    // focusedId is set — ego-dim should apply, not path-highlight
    const pathNodeSet = new Set(["node-A", "node-B"]);
    const reducer = mod.buildNodeReducer(
      "node-A",
      new Set(["node-B"]),
      null,
      { nodeSize: 60 },
      pathNodeSet,
    );
    // node-C is not neighbor of node-A; it should be dimmed by ego-focus (not path)
    const result = reducer("node-C", { label: "NodeC", color: "#3B82F6", size: 10 });
    expect(result.label).toBe("");
    expect(result.color).toMatch(/1F$/);
  });
});

describe("buildEdgeReducer — pathEdgeSet (D-03)", () => {
  const endpoints = (edge: string) => {
    const [source, target] = edge.split("->");
    return { source: source ?? "", target: target ?? "" };
  };

  it("path edge renders #3B82F6 when pathEdgeSet contains edge key", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const pathEdgeSet = new Set(["node-A->node-B"]);
    const reducer = mod.buildEdgeReducer(
      null,
      new Set(),
      { linkThick: 50 },
      endpoints,
      undefined,
      pathEdgeSet,
    );
    const result = reducer("node-A->node-B", { color: "#475569", size: 1 });
    expect(result.color).toBe("#3B82F6");
  });

  it("non-path edge dims when pathEdgeSet provided and no focus", async () => {
    const mod = (await import("@/lib/graphReducers")) as unknown as {
      buildEdgeReducer: BuildEdgeReducer;
    };
    const pathEdgeSet = new Set(["node-A->node-B"]);
    const reducer = mod.buildEdgeReducer(
      null,
      new Set(),
      { linkThick: 50 },
      endpoints,
      undefined,
      pathEdgeSet,
    );
    const result = reducer("node-C->node-D", { color: "#475569", size: 1 });
    expect(result.color).not.toBe("#3B82F6");
    expect(result.color).not.toBe("#475569");
  });
});
