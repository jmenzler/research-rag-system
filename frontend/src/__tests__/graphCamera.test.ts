/**
 * Wave-0 RED test for computeFit (D-06).
 *
 * Implementation lands in Plan 04.1-01 (frontend/src/lib/graphCamera.ts).
 *
 * Behavioural contract:
 *   computeFit({minX,minY,maxX,maxY}, {w,h}):
 *     cx = (minX + maxX) / 2
 *     cy = (minY + maxY) / 2
 *     graphW = maxX - minX + 1
 *     graphH = maxY - minY + 1
 *     ratio = Math.max(graphW/w, graphH/h) * 1.2  (20% padding)
 *     returns {x: cx, y: cy, ratio}
 *   Empty/non-finite box → returns null
 */
import { describe, expect, it } from "vitest";

interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface Viewport {
  w: number;
  h: number;
}

interface FitResult {
  x: number;
  y: number;
  ratio: number;
}

type ComputeFit = (box: BoundingBox, viewport: Viewport) => FitResult | null;

describe("computeFit (D-06) — Wave 0 pure-function tests", () => {
  it("returns correct centroid for a square bounding box", async () => {
    const mod = (await import("@/lib/graphCamera")) as unknown as {
      computeFit: ComputeFit;
    };
    const result = mod.computeFit(
      { minX: 0, minY: 0, maxX: 10, maxY: 10 },
      { w: 100, h: 100 },
    );
    expect(result).not.toBeNull();
    expect(result!.x).toBeCloseTo(5, 5);
    expect(result!.y).toBeCloseTo(5, 5);
  });

  it("returns correct ratio with 20% padding", async () => {
    const mod = (await import("@/lib/graphCamera")) as unknown as {
      computeFit: ComputeFit;
    };
    const result = mod.computeFit(
      { minX: 0, minY: 0, maxX: 10, maxY: 10 },
      { w: 100, h: 100 },
    );
    expect(result).not.toBeNull();
    const graphW = 10 - 0 + 1;
    const graphH = 10 - 0 + 1;
    const expectedRatio = Math.max(graphW / 100, graphH / 100) * 1.2;
    expect(result!.ratio).toBeCloseTo(expectedRatio, 5);
  });

  it("returns null for infinite/non-finite bounding box (empty graph)", async () => {
    const mod = (await import("@/lib/graphCamera")) as unknown as {
      computeFit: ComputeFit;
    };
    const result = mod.computeFit(
      { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity },
      { w: 100, h: 100 },
    );
    expect(result).toBeNull();
  });

  it("handles asymmetric viewport (uses max of w-ratio, h-ratio)", async () => {
    const mod = (await import("@/lib/graphCamera")) as unknown as {
      computeFit: ComputeFit;
    };
    const result = mod.computeFit(
      { minX: 0, minY: 0, maxX: 100, maxY: 20 },
      { w: 200, h: 100 },
    );
    expect(result).not.toBeNull();
    const graphW = 101;
    const graphH = 21;
    const expectedRatio = Math.max(graphW / 200, graphH / 100) * 1.2;
    expect(result!.ratio).toBeCloseTo(expectedRatio, 5);
  });

  it("centroid is correctly computed for non-zero-origin box", async () => {
    const mod = (await import("@/lib/graphCamera")) as unknown as {
      computeFit: ComputeFit;
    };
    const result = mod.computeFit(
      { minX: 20, minY: 30, maxX: 80, maxY: 70 },
      { w: 500, h: 500 },
    );
    expect(result).not.toBeNull();
    expect(result!.x).toBeCloseTo(50, 5);
    expect(result!.y).toBeCloseTo(50, 5);
  });
});
