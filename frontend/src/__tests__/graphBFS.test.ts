/**
 * Wave-0 GREEN test for computeBFSDepths + computeShortestPath (D-02, D-03).
 *
 * Implementation lands in Plan 07-01 (frontend/src/lib/graphBFS.ts).
 *
 * Behavioural contract:
 *   computeBFSDepths(nodes, edges):
 *     - seed (isSeed:true) → depth 0
 *     - direct neighbor → depth 1
 *     - 2-hop neighbor → depth 2
 *     - disconnected node → NOT present in the returned Map
 *     - no node has isSeed:true → returns empty Map
 *     - edges treated UNDIRECTED (a→b makes b reachable from a and vice versa)
 *
 *   computeShortestPath(seedId, targetId, edges):
 *     - 3-node chain seed→m→target → [seed, m, target]
 *     - target unreachable → []
 *     - seedId === targetId → [seedId]
 */
import { describe, expect, it } from "vitest";

interface GraphNodeInput {
  corpusId: number;
  isSeed: boolean;
}

interface GraphEdgeInput {
  fromId: number;
  toId: number;
}

type ComputeBFSDepths = (
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
) => Map<number, number>;

type ComputeShortestPath = (
  seedId: number,
  targetId: number,
  edges: GraphEdgeInput[],
) => number[];

// Fixed small graph:
// 1 (seed) -- 2 -- 3 -- 4
//                       5 (disconnected)
const NODES: GraphNodeInput[] = [
  { corpusId: 1, isSeed: true },
  { corpusId: 2, isSeed: false },
  { corpusId: 3, isSeed: false },
  { corpusId: 4, isSeed: false },
  { corpusId: 5, isSeed: false },
];

const EDGES: GraphEdgeInput[] = [
  { fromId: 1, toId: 2 },
  { fromId: 2, toId: 3 },
  { fromId: 3, toId: 4 },
];

describe("computeBFSDepths (D-02) — BFS depth computation", () => {
  it("seed node has depth 0", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    const depths = mod.computeBFSDepths(NODES, EDGES);
    expect(depths.get(1)).toBe(0);
  });

  it("direct neighbor has depth 1", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    const depths = mod.computeBFSDepths(NODES, EDGES);
    expect(depths.get(2)).toBe(1);
  });

  it("2-hop neighbor has depth 2", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    const depths = mod.computeBFSDepths(NODES, EDGES);
    expect(depths.get(3)).toBe(2);
  });

  it("disconnected node is absent from the depth Map", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    const depths = mod.computeBFSDepths(NODES, EDGES);
    expect(depths.has(5)).toBe(false);
  });

  it("returns empty Map when no node has isSeed:true", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    const noSeedNodes: GraphNodeInput[] = [
      { corpusId: 1, isSeed: false },
      { corpusId: 2, isSeed: false },
    ];
    const depths = mod.computeBFSDepths(noSeedNodes, EDGES);
    expect(depths.size).toBe(0);
  });

  it("edges are undirected: reverse traversal works", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeBFSDepths: ComputeBFSDepths;
    };
    // Edge goes 2→1 but seed is 1; node 2 should still be depth 1
    const nodes: GraphNodeInput[] = [
      { corpusId: 1, isSeed: true },
      { corpusId: 2, isSeed: false },
    ];
    const edges: GraphEdgeInput[] = [{ fromId: 2, toId: 1 }];
    const depths = mod.computeBFSDepths(nodes, edges);
    expect(depths.get(2)).toBe(1);
  });
});

describe("computeShortestPath (D-03) — shortest path reconstruction", () => {
  it("3-node chain returns full ordered path", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeShortestPath: ComputeShortestPath;
    };
    // chain: 1 -- 2 -- 3
    const edges: GraphEdgeInput[] = [
      { fromId: 1, toId: 2 },
      { fromId: 2, toId: 3 },
    ];
    const path = mod.computeShortestPath(1, 3, edges);
    expect(path).toEqual([1, 2, 3]);
  });

  it("unreachable target returns empty array []", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeShortestPath: ComputeShortestPath;
    };
    const path = mod.computeShortestPath(1, 99, EDGES);
    expect(path).toEqual([]);
  });

  it("seedId === targetId returns [seedId]", async () => {
    const mod = (await import("@/lib/graphBFS")) as unknown as {
      computeShortestPath: ComputeShortestPath;
    };
    const path = mod.computeShortestPath(1, 1, EDGES);
    expect(path).toEqual([1]);
  });
});
