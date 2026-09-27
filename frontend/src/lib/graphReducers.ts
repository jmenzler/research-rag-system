/**
 * Pure factory functions for Sigma v3 nodeReducer + edgeReducer (D-04, D-05).
 *
 * No React, no sigma runtime imports — fully unit-testable in jsdom.
 * Types are imported as type-only to keep the module free of sigma runtime.
 */

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

export interface NodeDisplayData {
  label: string;
  color: string;
  size: number;
  forceLabel?: boolean;
  [key: string]: unknown;
}

export interface EdgeDisplayData {
  color: string;
  size?: number;
  hidden?: boolean;
  [key: string]: unknown;
}

const INCIDENT_EDGE_COLOR = "#3B82F6";
const DEFAULT_EDGE_COLOR = "#475569";
const DIMMED_EDGE_COLOR = "#47556940";

// D-04 starting value — tune live (HUMAN-UAT)
const DIM_SIZE_FACTOR = 0.60;

/**
 * Build a Sigma v3 nodeReducer that applies ego-highlight dimming and optional
 * path-highlight (D-03). Precedence: ego-focus (focusedId set) wins over path.
 *
 * - Focused + neighbor nodes: forceLabel:true, full color, size scaled.
 * - All other nodes when focus active: color dimmed to 12% alpha (hex "1F" suffix)
 *   + 0.60x size shrink (DIM_SIZE_FACTOR), label hidden ("").
 * - No focus + pathNodeSet non-empty: path nodes full color + forceLabel; off-path dims.
 * - No focus + no path: all nodes at full color, size scaled.
 */
export function buildNodeReducer(
  focusedId: string | null,
  neighborIds: Set<string>,
  selectedId: string | null,
  display: Pick<GraphDisplaySettings, "nodeSize">,
  pathNodeSet?: Set<string>,
): (node: string, data: NodeDisplayData) => Partial<NodeDisplayData> {
  const sizeFactor = 0.65 + display.nodeSize / 100;

  return (node: string, data: NodeDisplayData): Partial<NodeDisplayData> => {
    const isFocused = node === focusedId;
    const isNeighbor = neighborIds.has(node);
    const dimmed = focusedId !== null && !isFocused && !isNeighbor;

    if (dimmed) {
      return {
        ...data,
        color: `${data.color}1F`,
        label: "",
        size: data.size * sizeFactor * DIM_SIZE_FACTOR,
      };
    }

    // D-03: path-highlight mode — only when no ego-focus is active (OQ-3 precedence).
    if (focusedId === null && pathNodeSet !== undefined && pathNodeSet.size > 0) {
      const isOnPath = pathNodeSet.has(node);
      if (!isOnPath) {
        return {
          ...data,
          color: `${data.color}1F`,
          label: "",
          size: data.size * sizeFactor * DIM_SIZE_FACTOR,
        };
      }
      return {
        ...data,
        size: data.size * sizeFactor,
        forceLabel: true,
      };
    }

    const result: Partial<NodeDisplayData> = {
      ...data,
      size: data.size * sizeFactor,
    };

    if (isFocused || isNeighbor || node === selectedId) {
      result.forceLabel = true;
    }

    return result;
  };
}

/**
 * Build a Sigma v3 edgeReducer that applies ego-highlight and optional path-set
 * edge coloring (D-03). Precedence: ego-focus wins over path-highlight (OQ-3).
 *
 * Requires an endpoint-resolver fn so the reducer is pure and testable without
 * a live sigma graph instance (Pitfall 5: edge data lacks node ids).
 *
 * - Layout running: all edges hidden (perf fast-path).
 * - Incident edges (touching focused node): primary blue #3B82F6.
 * - Non-incident edges when focus active: dimmed (#47556940).
 * - No focus + pathEdgeSet non-empty: path edges blue+thicker; off-path dims.
 * - No focus + no path: default edge color (#475569).
 */
export function buildEdgeReducer(
  focusedId: string | null,
  neighborIds: Set<string>,
  display: Pick<GraphDisplaySettings, "linkThick">,
  edgeEndpoints: (edge: string) => { source: string; target: string },
  isLayoutRunning?: () => boolean,
  pathEdgeSet?: Set<string>,
): (edge: string, data: EdgeDisplayData) => Partial<EdgeDisplayData> {
  const sizeFactor = 0.4 + display.linkThick / 100;

  return (edge: string, data: EdgeDisplayData): Partial<EdgeDisplayData> => {
    // A real walk is ~500 nodes / ~10k edges. Rendering + re-reducing every edge
    // each frame across the FA2 reheat saturates the main thread and GPU. While
    // the layout moves, hide edges and skip the endpoint resolution entirely —
    // the cheapest possible per-edge path. Edges reappear once motion settles.
    if (isLayoutRunning?.()) return { hidden: true };

    // D-03: path-highlight mode — only when no ego-focus is active (OQ-3 precedence).
    if (focusedId === null && pathEdgeSet !== undefined && pathEdgeSet.size > 0) {
      const isOnPath = pathEdgeSet.has(edge);
      if (isOnPath) {
        return {
          ...data,
          color: INCIDENT_EDGE_COLOR,
          size: (data.size ?? 1) * sizeFactor * 1.8,
        };
      }
      return {
        ...data,
        color: DIMMED_EDGE_COLOR,
        size: (data.size ?? 1) * sizeFactor,
      };
    }

    if (focusedId === null) {
      return {
        ...data,
        color: DEFAULT_EDGE_COLOR,
        size: (data.size ?? 1) * sizeFactor,
      };
    }

    const { source, target } = edgeEndpoints(edge);
    const isIncident = source === focusedId || target === focusedId;

    return {
      ...data,
      color: isIncident ? INCIDENT_EDGE_COLOR : DIMMED_EDGE_COLOR,
      size: (data.size ?? 1) * sizeFactor,
    };
  };
}
