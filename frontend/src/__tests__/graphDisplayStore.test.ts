/**
 * Wave-0 RED test for graphDisplayStore (D-09).
 *
 * Implementation lands in Plan 04.1-01 (frontend/src/state/graphDisplayStore.ts).
 *
 * Behavioural contract:
 *   - Default state matches key_values defaults
 *     (nodeSize 60, linkThick 50, textFade 40, arrowsOn true,
 *      centerForce 40, repelForce 55, linkForce 50, linkDistance 50, animate false)
 *   - setNodeSize(80) updates state AND writes to localStorage under "rag-graph-display"
 *   - Setters clamp values to [0, 100] range (threat model T-04.1-01)
 *   - On re-import with pre-seeded localStorage, store rehydrates that value
 */
import { describe, expect, it, vi, beforeEach } from "vitest";

beforeEach(() => {
  vi.resetModules();
  const storage: Record<string, string> = {};
  const mock = {
    getItem: vi.fn((k: string) => storage[k] ?? null),
    setItem: vi.fn((k: string, v: string) => {
      storage[k] = v;
    }),
    removeItem: vi.fn((k: string) => {
      delete storage[k];
    }),
    clear: vi.fn(() => {
      for (const k of Object.keys(storage)) delete storage[k];
    }),
    key: vi.fn((_i: number) => null),
    length: 0,
  };
  vi.stubGlobal("localStorage", mock);
});

describe("useGraphDisplayStore (D-09) — Wave 0 store tests", () => {
  it("default state matches key_values: nodeSize=60, linkThick=50, textFade=40", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    const state = useGraphDisplayStore.getState() as unknown as {
      nodeSize: number;
      linkThick: number;
      textFade: number;
    };
    expect(state.nodeSize).toBe(60);
    expect(state.linkThick).toBe(50);
    expect(state.textFade).toBe(40);
  });

  it("default state: arrowsOn=true, centerForce=40, repelForce=55", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    const state = useGraphDisplayStore.getState() as unknown as {
      arrowsOn: boolean;
      centerForce: number;
      repelForce: number;
    };
    expect(state.arrowsOn).toBe(true);
    expect(state.centerForce).toBe(40);
    expect(state.repelForce).toBe(55);
  });

  it("default state: linkForce=50, linkDistance=50, animate=false", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    const state = useGraphDisplayStore.getState() as unknown as {
      linkForce: number;
      linkDistance: number;
      animate: boolean;
    };
    expect(state.linkForce).toBe(50);
    expect(state.linkDistance).toBe(50);
    expect(state.animate).toBe(false);
  });

  it("setNodeSize(80) updates state and writes to localStorage key 'rag-graph-display'", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    (
      useGraphDisplayStore.getState() as unknown as { setNodeSize: (v: number) => void }
    ).setNodeSize(80);

    const state = useGraphDisplayStore.getState() as unknown as { nodeSize: number };
    expect(state.nodeSize).toBe(80);

    const setItem = (localStorage as unknown as { setItem: ReturnType<typeof vi.fn> }).setItem;
    const wroteToKey = setItem.mock.calls.some((c) => String(c[0]) === "rag-graph-display");
    expect(wroteToKey).toBe(true);
  });

  it("setters clamp values above 100 to 100 (threat model T-04.1-01)", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    (
      useGraphDisplayStore.getState() as unknown as { setNodeSize: (v: number) => void }
    ).setNodeSize(200);
    const state = useGraphDisplayStore.getState() as unknown as { nodeSize: number };
    expect(state.nodeSize).toBe(100);
  });

  it("setters clamp values below 0 to 0 (threat model T-04.1-01)", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    (
      useGraphDisplayStore.getState() as unknown as { setNodeSize: (v: number) => void }
    ).setNodeSize(-50);
    const state = useGraphDisplayStore.getState() as unknown as { nodeSize: number };
    expect(state.nodeSize).toBe(0);
  });

  it("storage key is exactly 'rag-graph-display'", async () => {
    const { useGraphDisplayStore } = await import("@/state/graphDisplayStore");
    (
      useGraphDisplayStore.getState() as unknown as { setLinkThick: (v: number) => void }
    ).setLinkThick(70);

    const setItem = (localStorage as unknown as { setItem: ReturnType<typeof vi.fn> }).setItem;
    const call = setItem.mock.calls.find((c) => String(c[0]) === "rag-graph-display");
    expect(call).toBeDefined();
    const payload = String(call?.[1] ?? "");
    expect(payload).toContain("linkThick");
  });
});
