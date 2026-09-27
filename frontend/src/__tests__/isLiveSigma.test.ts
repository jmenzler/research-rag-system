/**
 * Regression guard for the StrictMode-killed-instance fix (04.1-03).
 *
 * React StrictMode double-invokes effects; @react-sigma kills the throwaway
 * first-mount Sigma, emptying `sigma.nodePrograms` while leaving settings intact.
 * Raw sigma effects (refresh/setSetting/createCanvas) that run against that dead
 * instance throw "could not find a suitable program for node type circle".
 * `isLiveSigma` is the liveness guard every raw-sigma effect checks first.
 *
 * Also documents that computeFit (graphCamera.ts) operates in RAW graphology
 * coordinates and MUST NOT be fed straight into sigma's camera.animate (which
 * uses the framed/normalized [0,1] space) — see GraphCanvasSigma auto-fit removal.
 */
import { describe, it, expect } from "vitest";

import { isLiveSigma } from "@/lib/sigmaLiveness";
import { computeFit } from "@/lib/graphCamera";

describe("isLiveSigma", () => {
  it("returns false for null/undefined", () => {
    expect(isLiveSigma(null)).toBe(false);
    expect(isLiveSigma(undefined)).toBe(false);
  });

  it("returns false for a killed instance (empty nodePrograms)", () => {
    // StrictMode-killed: WebGL programs torn down, nodePrograms emptied.
    expect(isLiveSigma({ nodePrograms: {} })).toBe(false);
    expect(isLiveSigma({})).toBe(false);
  });

  it("returns true for a live instance with a registered node program", () => {
    expect(isLiveSigma({ nodePrograms: { circle: {} } })).toBe(true);
  });
});

describe("computeFit coordinate-space contract", () => {
  it("returns raw graph-coordinate center (NOT framed/normalized) — do not feed to camera.animate", () => {
    // Nodes spread over raw coords ~[0, 10]; fit center is the raw midpoint, not 0.5.
    const fit = computeFit(
      { minX: 0, minY: 0, maxX: 10, maxY: 10 },
      { w: 600, h: 600 },
    );
    expect(fit).not.toBeNull();
    // Center is raw (5, 5) — far from sigma's framed default camera (0.5, 0.5).
    // This is precisely why computeFit must NOT drive camera.animate; sigma's
    // built-in autoRescale + default camera fits the normalized graph instead.
    expect(fit?.x).toBe(5);
    expect(fit?.y).toBe(5);
  });
});
