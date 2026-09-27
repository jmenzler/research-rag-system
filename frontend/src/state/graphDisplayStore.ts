/**
 * Persisted Zustand store for graph Display + Forces slider values (D-07, D-09).
 *
 * All 8 sliders + animate flag persist to localStorage key "rag-graph-display"
 * so they survive page reloads. Setters clamp numeric values to [0, 100] to
 * prevent tampered localStorage values flowing unbounded into Sigma/FA2 settings
 * (threat model T-04.1-01).
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { StateStorage } from "zustand/middleware";

function safeStorage(): StateStorage {
  const noop: StateStorage = {
    getItem: () => null,
    setItem: () => undefined,
    removeItem: () => undefined,
  };
  try {
    if (typeof localStorage === "undefined") return noop;
    if (typeof localStorage.setItem !== "function") return noop;
    return localStorage as StateStorage;
  } catch {
    return noop;
  }
}

function clamp(v: number): number {
  return Math.max(0, Math.min(100, v));
}

export interface GraphDisplaySettings {
  nodeSize: number;
  linkThick: number;
  textFade: number;
  arrowsOn: boolean;
  centerForce: number;
  repelForce: number;
  linkForce: number;
  linkDistance: number;
  animate: boolean;
}

const DEFAULTS: GraphDisplaySettings = {
  nodeSize: 60,
  linkThick: 50,
  textFade: 40,
  arrowsOn: true,
  centerForce: 40,
  repelForce: 55,
  linkForce: 50,
  linkDistance: 50,
  animate: false,
};

export interface GraphDisplayState extends GraphDisplaySettings {
  setNodeSize: (v: number) => void;
  setLinkThick: (v: number) => void;
  setTextFade: (v: number) => void;
  setArrowsOn: (v: boolean) => void;
  setCenterForce: (v: number) => void;
  setRepelForce: (v: number) => void;
  setLinkForce: (v: number) => void;
  setLinkDistance: (v: number) => void;
  setAnimate: (v: boolean) => void;
  reset: () => void;
}

export const useGraphDisplayStore = create<GraphDisplayState>()(
  persist(
    (set) => ({
      ...DEFAULTS,

      setNodeSize: (v) => set({ nodeSize: clamp(v) }),
      setLinkThick: (v) => set({ linkThick: clamp(v) }),
      setTextFade: (v) => set({ textFade: clamp(v) }),
      setArrowsOn: (v) => set({ arrowsOn: v }),
      setCenterForce: (v) => set({ centerForce: clamp(v) }),
      setRepelForce: (v) => set({ repelForce: clamp(v) }),
      setLinkForce: (v) => set({ linkForce: clamp(v) }),
      setLinkDistance: (v) => set({ linkDistance: clamp(v) }),
      setAnimate: (v) => set({ animate: v }),
      reset: () => set({ ...DEFAULTS }),
    }),
    {
      name: "rag-graph-display",
      storage: createJSONStorage(safeStorage),
    },
  ),
);
