/**
 * Zustand transient UI state for the graph explorer surface.
 *
 * Graph node/edge data is server state (TanStack Query, graph.ts).
 * This store holds only ephemeral UI state that doesn't need persistence:
 *   - selectedNodeId: which node's detail panel is open
 *   - focusedNodeId: ego-highlight focus (string — matches graphology node-key format)
 *   - listOpen: whether the accessible list pane is visible
 *   - ceilingOverride: whether the user dismissed the soft-ceiling warning
 *   - overlay{Nodes,Edges}/overflowCount: the active walk/expand result set,
 *     held here (not GraphPage-local) so it survives in-app tab unmount/remount
 *     (a tab switch must not reset the graph). Resets on collection change only.
 *
 * No persist middleware — graph state survives navigation within the SPA but
 * resets on a full page reload.
 */
import { create } from "zustand";

import type { GraphEdge, GraphNode } from "@/queries/graph";

type Updater<T> = T | ((prev: T) => T);

function applyUpdater<T>(value: Updater<T>, prev: T): T {
  return typeof value === "function" ? (value as (p: T) => T)(prev) : value;
}

interface GraphState {
  selectedNodeId: number | null;
  setSelectedNodeId: (id: number | null) => void;
  focusedNodeId: string | null;
  setFocusedNodeId: (id: string | null) => void;
  clearFocus: () => void;
  listOpen: boolean;
  setListOpen: (open: boolean) => void;
  ceilingOverride: boolean;
  setCeilingOverride: (v: boolean) => void;
  inFlightCorpusIds: Set<number>;
  addInFlight: (ids: number[]) => void;
  removeInFlight: (ids: number[]) => void;
  overlayNodes: GraphNode[] | null;
  overlayEdges: GraphEdge[] | null;
  overflowCount: number;
  setOverlayNodes: (v: Updater<GraphNode[] | null>) => void;
  setOverlayEdges: (v: Updater<GraphEdge[] | null>) => void;
  setOverflowCount: (v: Updater<number>) => void;
  walkSeedId: string | null;
  setWalkSeedId: (id: string | null) => void;
  hoveredPathNodeSet: Set<string>;
  hoveredPathEdgeSet: Set<string>;
  setHoveredPath: (nodeSet: Set<string>, edgeSet: Set<string>) => void;
  clearHoveredPath: () => void;
}

export const useGraphStore = create<GraphState>()((set) => ({
  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
  focusedNodeId: null,
  setFocusedNodeId: (id) => set({ focusedNodeId: id }),
  clearFocus: () => set({ focusedNodeId: null }),
  listOpen: false,
  setListOpen: (open) => set({ listOpen: open }),
  ceilingOverride: false,
  setCeilingOverride: (v) => set({ ceilingOverride: v }),
  overlayNodes: null,
  overlayEdges: null,
  overflowCount: 0,
  setOverlayNodes: (v) => set((s) => ({ overlayNodes: applyUpdater(v, s.overlayNodes) })),
  setOverlayEdges: (v) => set((s) => ({ overlayEdges: applyUpdater(v, s.overlayEdges) })),
  setOverflowCount: (v) => set((s) => ({ overflowCount: applyUpdater(v, s.overflowCount) })),
  walkSeedId: null,
  setWalkSeedId: (id) => set({ walkSeedId: id }),
  hoveredPathNodeSet: new Set<string>(),
  hoveredPathEdgeSet: new Set<string>(),
  setHoveredPath: (nodeSet, edgeSet) => set({ hoveredPathNodeSet: nodeSet, hoveredPathEdgeSet: edgeSet }),
  clearHoveredPath: () => set({ hoveredPathNodeSet: new Set<string>(), hoveredPathEdgeSet: new Set<string>() }),
  inFlightCorpusIds: new Set<number>(),
  // Both mutators bail out (no state change → no re-render) when the set would
  // be unchanged. SSE 'running' events repeat every poll with the same ids;
  // without this guard each repeat allocated a new Set, re-running GraphCanvas's
  // buildGraph and repainting the WebGL canvas (~1Hz flash).
  addInFlight: (ids) =>
    set((state) => {
      if (ids.every((id) => state.inFlightCorpusIds.has(id))) return {};
      const next = new Set(state.inFlightCorpusIds);
      ids.forEach((id) => next.add(id));
      return { inFlightCorpusIds: next };
    }),
  removeInFlight: (ids) =>
    set((state) => {
      if (!ids.some((id) => state.inFlightCorpusIds.has(id))) return {};
      const next = new Set(state.inFlightCorpusIds);
      ids.forEach((id) => next.delete(id));
      return { inFlightCorpusIds: next };
    }),
}));
