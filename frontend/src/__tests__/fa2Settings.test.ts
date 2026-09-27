/**
 * Wave-0 RED test for toFA2Settings (D-07).
 *
 * Implementation lands in Plan 04.1-01 (frontend/src/lib/fa2Settings.ts).
 *
 * Behavioural contract (slider 0–100 → FA2 setting ranges):
 *   gravity        = (centerForce  / 100) * 2          → 0–2
 *   scalingRatio   = 1 + (repelForce / 100) * 19       → 1–20
 *   edgeWeightInfluence = (linkForce / 100) * 2        → 0–2
 *   slowDown       = 1 + (linkDistance / 100) * 19     → 1–20
 */
import { describe, expect, it } from "vitest";

interface GraphDisplayInput {
  centerForce: number;
  repelForce: number;
  linkForce: number;
  linkDistance: number;
}

interface FA2Settings {
  gravity: number;
  scalingRatio: number;
  edgeWeightInfluence: number;
  slowDown: number;
}

type ToFA2Settings = (display: GraphDisplayInput) => FA2Settings;

describe("toFA2Settings (D-07) — Wave 0 pure-function tests", () => {
  it("maps slider=50 to midpoint values", async () => {
    const mod = (await import("@/lib/fa2Settings")) as unknown as {
      toFA2Settings: ToFA2Settings;
    };
    const result = mod.toFA2Settings({
      centerForce: 50,
      repelForce: 50,
      linkForce: 50,
      linkDistance: 50,
    });
    expect(result.gravity).toBeCloseTo(1, 5);
    expect(result.scalingRatio).toBeCloseTo(10.5, 5);
    expect(result.edgeWeightInfluence).toBeCloseTo(1, 5);
    expect(result.slowDown).toBeCloseTo(10.5, 5);
  });

  it("maps slider=0 to minimum values", async () => {
    const mod = (await import("@/lib/fa2Settings")) as unknown as {
      toFA2Settings: ToFA2Settings;
    };
    const result = mod.toFA2Settings({
      centerForce: 0,
      repelForce: 0,
      linkForce: 0,
      linkDistance: 0,
    });
    expect(result.gravity).toBeCloseTo(0, 5);
    expect(result.scalingRatio).toBeCloseTo(1, 5);
    expect(result.edgeWeightInfluence).toBeCloseTo(0, 5);
    expect(result.slowDown).toBeCloseTo(1, 5);
  });

  it("maps slider=100 to maximum values", async () => {
    const mod = (await import("@/lib/fa2Settings")) as unknown as {
      toFA2Settings: ToFA2Settings;
    };
    const result = mod.toFA2Settings({
      centerForce: 100,
      repelForce: 100,
      linkForce: 100,
      linkDistance: 100,
    });
    expect(result.gravity).toBeCloseTo(2, 5);
    expect(result.scalingRatio).toBeCloseTo(20, 5);
    expect(result.edgeWeightInfluence).toBeCloseTo(2, 5);
    expect(result.slowDown).toBeCloseTo(20, 5);
  });

  it("centerForce=40 (default) → gravity=0.8", async () => {
    const mod = (await import("@/lib/fa2Settings")) as unknown as {
      toFA2Settings: ToFA2Settings;
    };
    const result = mod.toFA2Settings({
      centerForce: 40,
      repelForce: 55,
      linkForce: 50,
      linkDistance: 50,
    });
    expect(result.gravity).toBeCloseTo(0.8, 5);
  });
});
