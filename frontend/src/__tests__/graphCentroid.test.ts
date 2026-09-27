/**
 * Wave-0 GREEN test for computeCentroid (D-01).
 *
 * Implementation lands in Plan 07-01 (frontend/src/lib/graphCentroid.ts).
 *
 * Behavioural contract:
 *   computeCentroid(newNodeId, edges, existingPositions, fallbackNodeId, jitterRadius?):
 *     - One already-placed neighbor → result within jitterRadius of that neighbor
 *     - Two already-placed neighbors → result near mean of the two positions (±jitter)
 *     - No placed neighbors, valid fallbackNodeId → near fallback pos (±jitter)
 *     - No placed neighbors, no fallback → finite numbers (Math.random path)
 *     - Pitfall 5: both endpoints new → fallback chain (treated as no placed neighbors)
 */
import { describe, expect, it } from "vitest";

interface EdgeInput {
  fromId: number;
  toId: number;
}

type ComputeCentroid = (
  newNodeId: number,
  edges: EdgeInput[],
  existingPositions: Map<number, { x: number; y: number }>,
  fallbackNodeId: number | null,
  jitterRadius?: number,
) => { x: number; y: number };

describe("computeCentroid (D-01) — edge-aware centroid placement", () => {
  it("one placed neighbor: result equals neighbor position when jitterRadius is 0", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    const existingPositions = new Map([[10, { x: 50, y: 100 }]]);
    const edges: EdgeInput[] = [{ fromId: 1, toId: 10 }];
    const result = mod.computeCentroid(1, edges, existingPositions, null, 0);
    expect(result.x).toBeCloseTo(50, 5);
    expect(result.y).toBeCloseTo(100, 5);
  });

  it("two placed neighbors: result equals their mean when jitterRadius is 0", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    const existingPositions = new Map([
      [10, { x: 0, y: 0 }],
      [20, { x: 100, y: 200 }],
    ]);
    const edges: EdgeInput[] = [
      { fromId: 1, toId: 10 },
      { fromId: 1, toId: 20 },
    ];
    const result = mod.computeCentroid(1, edges, existingPositions, null, 0);
    expect(result.x).toBeCloseTo(50, 5);
    expect(result.y).toBeCloseTo(100, 5);
  });

  it("no placed neighbors, valid fallback: result equals fallback when jitterRadius is 0", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    const existingPositions = new Map([[99, { x: 30, y: 40 }]]);
    const edges: EdgeInput[] = [];
    const result = mod.computeCentroid(1, edges, existingPositions, 99, 0);
    expect(result.x).toBeCloseTo(30, 5);
    expect(result.y).toBeCloseTo(40, 5);
  });

  it("no placed neighbors, no fallback: result is finite numbers", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    const existingPositions = new Map<number, { x: number; y: number }>();
    const result = mod.computeCentroid(1, [], existingPositions, null);
    expect(isFinite(result.x)).toBe(true);
    expect(isFinite(result.y)).toBe(true);
    expect(isNaN(result.x)).toBe(false);
    expect(isNaN(result.y)).toBe(false);
  });

  it("Pitfall 5: both endpoints new → treated as no placed neighbors → fallback chain", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    // Node 1 and node 2 are BOTH new (neither in existingPositions).
    // The edge between them should contribute nothing.
    const existingPositions = new Map([[99, { x: 77, y: 88 }]]);
    const edges: EdgeInput[] = [{ fromId: 1, toId: 2 }];
    const result = mod.computeCentroid(1, edges, existingPositions, 99, 0);
    // Falls back to fallback node 99's position
    expect(result.x).toBeCloseTo(77, 5);
    expect(result.y).toBeCloseTo(88, 5);
  });

  it("default jitter: result is within ±10 of anchor for one placed neighbor", async () => {
    const mod = (await import("@/lib/graphCentroid")) as unknown as {
      computeCentroid: ComputeCentroid;
    };
    const existingPositions = new Map([[10, { x: 50, y: 50 }]]);
    const edges: EdgeInput[] = [{ fromId: 1, toId: 10 }];
    // Default jitterRadius = 20, so max offset is ±10
    const result = mod.computeCentroid(1, edges, existingPositions, null);
    expect(result.x).toBeGreaterThanOrEqual(40);
    expect(result.x).toBeLessThanOrEqual(60);
    expect(result.y).toBeGreaterThanOrEqual(40);
    expect(result.y).toBeLessThanOrEqual(60);
  });
});
