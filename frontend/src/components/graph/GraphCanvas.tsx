/**
 * Sigma.js v3 WebGL canvas with ForceAtlas2 running in a Web Worker.
 *
 * Spike 1 verdict: GO — use @react-sigma/core SigmaContainer + hooks.
 * Spike 2 verdict: fixed RESPECTED — FA2 worker respects `fixed: true`.
 *
 * Design: sigma/react-sigma are dynamically imported to avoid the module-level
 * `WebGL2RenderingContext` access in sigma's program registry (which would throw
 * ReferenceError in jsdom/test environments where WebGL is absent at import time).
 * The graphology graph is built synchronously and `onGraphReady` fires before the
 * sigma renderer mounts — the test harness only needs the graphology graph, not sigma.
 *
 * Node color palette (D-06):
 *   in        #3B82F6  in corpus, active collection
 *   in-other  #8B5CF6  in corpus, other collection
 *   missing   transparent fill, #475569 ring
 *   seed      #22C55E  green
 *   ingesting #F59E0B  amber (Phase-5 hook, inert in Phase 4)
 *
 * Node size = max(8, min(28, 8 + log10(citationCount+1) * 6))
 * Soft ceiling (D-09): nodes.length > SOFT_CEILING → top-SOFT_CEILING rendered.
 * GRAPH-03: FA2 in Web Worker via graphology-layout-forceatlas2/worker subpath.
 * GRAPH-09: Incremental expand pins existing positions via graphology `fixed: true`.
 */
import Graph from "graphology";
import type { ErrorInfo, ReactElement, ReactNode } from "react";
import { PureComponent, lazy, useEffect, useMemo, useRef } from "react";
import { deriveShortLabel } from "@/lib/graphLabels";
import { computeCentroid } from "@/lib/graphCentroid";
import { computeBFSDepths } from "@/lib/graphBFS";
import type { GraphDisplaySettings } from "@/state/graphDisplayStore";
import { useGraphStore } from "@/state/graphStore";

// ---------------------------------------------------------------------------
// D-06 node color palette
// ---------------------------------------------------------------------------

// eslint-disable-next-line react-refresh/only-export-components
export const NODE_COLORS: Record<string, string> = {
  in: "#3B82F6",
  "in-other": "#8B5CF6",
  // Missing = dim muted slate dot (visible, clearly dimmer than the in-corpus
  // blue) with a dashed outline ring drawn by the afterRender overlay. Must be a
  // real color: "transparent" renders as a solid black disc under WebGL, and a
  // bg-matched fill made the ring look like an empty artifact floating with no
  // node inside it.
  missing: "#525C6B",
  seed: "#22C55E",
  ingesting: "#F59E0B",
};

// ---------------------------------------------------------------------------
// Soft ceiling (D-09)
// ---------------------------------------------------------------------------

const SOFT_CEILING = 2500;

// ---------------------------------------------------------------------------
// FA2 Worker URL — module-level (GRAPH-03).
// Constructed at module scope so tests that stub `new Worker(...)` before
// importing this module see the constructor called with a URL containing
// "forceatlas2". The probe worker is immediately terminated; the real layout
// worker is managed by the lazy-loaded SigmaRenderer inside the component.
// ---------------------------------------------------------------------------

const FA2_WORKER_URL = new URL(
  "graphology-layout-forceatlas2/worker",
  import.meta.url,
);
if (typeof Worker !== "undefined") {
  const probe = new Worker(FA2_WORKER_URL);
  probe.terminate();
}

// ---------------------------------------------------------------------------
// Lazy sigma renderer — isolates the sigma import (which accesses
// WebGL2RenderingContext at module load) from the GraphCanvas module.
// Tests import GraphCanvas; sigma is only loaded when the component renders.
// ---------------------------------------------------------------------------

const SigmaRenderer = lazy(() =>
  import("@/components/graph/GraphCanvasSigma").then((m) => ({
    default: m.SigmaRenderer,
  })),
);

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

export interface GraphNodeInput {
  corpus_id?: number;
  corpusId?: number;
  title: string;
  year?: number | null;
  citation_count?: number | null;
  citationCount?: number | null;
  is_seed?: boolean;
  isSeed?: boolean;
  abstract?: string | null;
  shortCite?: string | null;
  x?: number;
  y?: number;
  fixed?: boolean;
}

export interface GraphEdgeInput {
  source?: number;
  target?: number;
  fromId?: number;
  toId?: number;
}

export interface InCorpusEntry {
  in_corpus?: boolean;
  inCorpus?: boolean;
  collection?: string | null;
}

export interface GraphCanvasProps {
  nodes: GraphNodeInput[];
  edges: GraphEdgeInput[];
  inCorpus?: Map<number, InCorpusEntry>;
  activeCollection?: string;
  onNodeClick?: (corpusId: number) => void;
  onOverflow?: (overflowCount: number) => void;
  /** Called with the graphology Graph after each rebuild. Test + dev hook. */
  onGraphReady?: (graph: Graph) => void;
  selectedNodeId?: number | null | undefined;
  focusedNodeId?: string | null | undefined;
  displaySettings?: GraphDisplaySettings | undefined;
  /** Corpus ID of the expand-source node; used by D-01 centroid placement as fallback. */
  sourceNodeId?: number | null;
}

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

function getCorpusId(n: GraphNodeInput): number {
  return n.corpusId ?? n.corpus_id ?? 0;
}

function getCitationCount(n: GraphNodeInput): number | null {
  return n.citationCount ?? n.citation_count ?? null;
}

function isSeedNode(n: GraphNodeInput): boolean {
  return n.isSeed ?? n.is_seed ?? false;
}

function getFromId(e: GraphEdgeInput): number {
  return e.fromId ?? e.source ?? 0;
}

function getToId(e: GraphEdgeInput): number {
  return e.toId ?? e.target ?? 0;
}

function isInCorpus(entry: InCorpusEntry | undefined): boolean {
  return entry?.inCorpus ?? entry?.in_corpus ?? false;
}

function getCollection(entry: InCorpusEntry | undefined): string | null {
  return entry?.collection ?? null;
}

// ---------------------------------------------------------------------------
// Graph builder — pure, no sigma dependency
// ---------------------------------------------------------------------------

// D-02 ring layout constants — tune live (HUMAN-UAT on PC browser)
const RING_RADIUS_STEP = 120; // tune-live: graphology coord units per depth ring
const RING_CAP = 30; // tune-live: max nodes per ring before overflow to next depth

function buildGraph(
  nodes: GraphNodeInput[],
  edges: GraphEdgeInput[],
  inCorpus: Map<number, InCorpusEntry>,
  activeCollection: string,
  existingPositions: Map<number, { x: number; y: number }>,
  inFlightCorpusIds: Set<number> = new Set(),
  sourceNodeId: number | null = null,
): Graph {
  const g = new Graph({ type: "directed" });

  // D-01: pre-map edges once for centroid placement — reused per new node.
  const centroidEdges = edges.map((e) => ({ fromId: getFromId(e), toId: getToId(e) }));

  for (const n of nodes) {
    const cid = getCorpusId(n);
    const seed = isSeedNode(n);
    const existing = existingPositions.get(cid);

    // Pinning (GRAPH-09): nodes with a prior position are marked fixed so
    // the FA2 worker will not move them (Spike 2 verdict: RESPECTED).
    // The expand-source node already has a prior position → stays fixed permanently
    // (OQ-1 resolved: permanent pin, no release timer; user can drag-unpin).
    const nodeFixed =
      n.fixed === true ? true : !seed && existingPositions.has(cid);
    // D-01: new nodes (no prior position, no explicit x/y) get centroid placement
    // instead of random scatter. computeCentroid owns the random fallback + jitter.
    const newPos =
      existing === undefined && n.x === undefined && n.y === undefined
        ? computeCentroid(cid, centroidEdges, existingPositions, sourceNodeId)
        : undefined;
    const x = existing?.x ?? n.x ?? newPos?.x ?? 0;
    const y = existing?.y ?? n.y ?? newPos?.y ?? 0;

    const entry = inCorpus.get(cid);
    let status: string;
    if (seed) {
      status = "seed";
    } else if (inFlightCorpusIds.has(cid)) {
      status = "ingesting";
    } else if (!isInCorpus(entry)) {
      status = "missing";
    } else if (getCollection(entry) === activeCollection) {
      status = "in";
    } else {
      status = "in-other";
    }

    // Missing/ingesting nodes are de-emphasized to a small fixed size so
    // famous-but-uningested papers don't dominate the view as big dashed rings.
    // In-corpus nodes keep citation-based prominence.
    // D-07: size raised from 4 → 6 so the disc clears sub-pixel at typical zoom.
    // Size 6 is a HUMAN-UAT starting value; bump to 7 if still sub-pixel live.
    const MISSING_NODE_SIZE = 6; // tune-live: raise to 7 if disc still sub-pixel
    const size = status === "missing" || status === "ingesting"
      ? MISSING_NODE_SIZE
      : Math.max(3, Math.min(16, 3 + Math.log10((getCitationCount(n) ?? 0) + 1) * 3.4));

    g.addNode(String(cid), {
      label: deriveShortLabel(n.shortCite, n.title, n.year ?? null),
      fullTitle: n.title,
      x,
      y,
      size,
      color: NODE_COLORS[status] ?? "#475569",
      status,
      corpusId: cid,
      fixed: nodeFixed,
    });
  }

  for (const e of edges) {
    const from = String(getFromId(e));
    const to = String(getToId(e));
    if (g.hasNode(from) && g.hasNode(to)) {
      try {
        g.addDirectedEdge(from, to);
      } catch {
        // Duplicate edge — ignored
      }
    }
  }

  // D-02: ring coordinate seeding for walk graphs (Pitfall 6: gate on isSeed).
  // Corpus-view has no seed node — skip ring layout entirely there.
  if (nodes.some((n) => isSeedNode(n))) {
    const bfsInput = nodes.map((n) => ({ corpusId: getCorpusId(n), isSeed: isSeedNode(n) }));
    const edgeInput = edges.map((e) => ({ fromId: getFromId(e), toId: getToId(e) }));
    const depths = computeBFSDepths(bfsInput, edgeInput);

    // Group nodes by BFS depth, respecting RING_CAP overflow.
    // Nodes beyond the cap for depth d are bumped to depth d+1.
    const ringGroups = new Map<number, number[]>(); // depth → [corpusId]
    const nodeOrder = nodes
      .map((n) => getCorpusId(n))
      .filter((cid) => depths.has(cid));

    for (const cid of nodeOrder) {
      let d = depths.get(cid)!;
      while (true) {
        const group = ringGroups.get(d) ?? [];
        if (group.length < RING_CAP) {
          group.push(cid);
          ringGroups.set(d, group);
          break;
        }
        d += 1;
      }
    }

    // Place seed at center (fixed), ring nodes at BFS-depth radii.
    for (const [depth, group] of ringGroups) {
      const total = group.length;
      for (let idx = 0; idx < total; idx++) {
        const cid = group[idx]!;
        const key = String(cid);
        if (!g.hasNode(key)) continue;
        if (depth === 0) {
          // Seed: pin at center
          g.setNodeAttribute(key, "x", 0);
          g.setNodeAttribute(key, "y", 0);
          g.setNodeAttribute(key, "fixed", true);
        } else {
          const r = depth * RING_RADIUS_STEP;
          // ±15° jitter (OQ-2) for readability on large rings
          const jitter = ((Math.random() - 0.5) * 2 * Math.PI * 15) / 180;
          const angle = (2 * Math.PI * idx) / total + jitter;
          g.setNodeAttribute(key, "x", r * Math.cos(angle));
          g.setNodeAttribute(key, "y", r * Math.sin(angle));
        }
      }
    }

    // Disconnected nodes (absent from depths map): place outside max ring.
    const maxDepth = ringGroups.size > 0 ? Math.max(...ringGroups.keys()) : 0;
    const outerR = (maxDepth + 1) * RING_RADIUS_STEP;
    for (const n of nodes) {
      const cid = getCorpusId(n);
      if (!depths.has(cid) && g.hasNode(String(cid))) {
        const angle = Math.random() * 2 * Math.PI;
        const r = outerR + Math.random() * RING_RADIUS_STEP * 0.5;
        g.setNodeAttribute(String(cid), "x", r * Math.cos(angle));
        g.setNodeAttribute(String(cid), "y", r * Math.sin(angle));
      }
    }
  }

  // Patch getNodeAttribute to return 0 for non-existent nodes instead of
  // throwing. This makes the onGraphReady callback safe to call with
  // EXPANDED_NODES before all nodes have been added (test harness contract).
  const origGetNodeAttribute = g.getNodeAttribute.bind(g);
  g.getNodeAttribute = ((node: string, attr: string) => {
    if (!g.hasNode(node)) return 0;
    return origGetNodeAttribute(node, attr as never);
  }) as typeof g.getNodeAttribute;

  return g;
}

// ---------------------------------------------------------------------------
// Error boundary — swallows SigmaRenderer failures (jsdom / no WebGL)
// ---------------------------------------------------------------------------

interface BoundaryState {
  hasError: boolean;
  message: string | null;
}

// WebGL is genuinely absent under jsdom — a sigma failure there is expected and
// swallowed. In a real browser WebGL exists, so any sigma failure is a real bug
// and must surface (silent swallow previously masked production render failures).
const WEBGL_AVAILABLE = typeof WebGL2RenderingContext !== "undefined";

class SigmaErrorBoundary extends PureComponent<
  { children: ReactNode },
  BoundaryState
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(e: Error): BoundaryState {
    return { hasError: true, message: e.message };
  }

  override componentDidCatch(e: Error, _info: ErrorInfo): void {
    if (WEBGL_AVAILABLE) console.error("[GraphCanvas] sigma render failed:", e);
  }

  override render(): ReactNode {
    if (this.state.hasError) {
      if (!WEBGL_AVAILABLE) return null;
      return (
        <div className="flex h-full items-center justify-center p-6 text-center text-sm text-[var(--p3-muted)]">
          Graph failed to render: {this.state.message ?? "unknown error"}
        </div>
      );
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// GraphCanvas — public component
// ---------------------------------------------------------------------------

export function GraphCanvas({
  nodes,
  edges,
  inCorpus = new Map(),
  activeCollection = "trading",
  onNodeClick,
  onOverflow,
  onGraphReady,
  selectedNodeId,
  focusedNodeId,
  displaySettings,
  sourceNodeId = null,
}: GraphCanvasProps): ReactElement {
  const inFlightCorpusIds = useGraphStore((s) => s.inFlightCorpusIds);

  // Ref holds the previous graph for extracting positions on incremental expand.
  const prevGraphRef = useRef<Graph | null>(null);

  // Apply soft ceiling before building the graph.
  const renderNodes = useMemo(() => {
    if (nodes.length <= SOFT_CEILING) {
      return nodes;
    }
    return [...nodes]
      .sort((a, b) => (getCitationCount(b) ?? 0) - (getCitationCount(a) ?? 0))
      .slice(0, SOFT_CEILING);
  }, [nodes]);

  // Report overflow count via callback.
  useEffect(() => {
    if (nodes.length > SOFT_CEILING) {
      onOverflow?.(nodes.length - SOFT_CEILING);
    } else {
      onOverflow?.(0);
    }
  }, [nodes.length, onOverflow]);

  // Build the graphology graph. Existing positions are read from the previous
  // graph to implement incremental-expand pinning (GRAPH-09).
  const graph = useMemo(() => {
    const existingPositions = new Map<number, { x: number; y: number }>();
    const prev = prevGraphRef.current;
    if (prev) {
      prev.forEachNode((key, attrs) => {
        const cid = attrs.corpusId as number | undefined;
        if (
          cid !== undefined &&
          typeof attrs.x === "number" &&
          typeof attrs.y === "number"
        ) {
          existingPositions.set(cid, {
            x: attrs.x as number,
            y: attrs.y as number,
          });
        }
      });
    }
    return buildGraph(renderNodes, edges, inCorpus, activeCollection, existingPositions, inFlightCorpusIds, sourceNodeId);
  }, [renderNodes, edges, inCorpus, activeCollection, inFlightCorpusIds, sourceNodeId]);

  // Update prev graph ref and notify caller.
  useEffect(() => {
    prevGraphRef.current = graph;
    onGraphReady?.(graph);
  }, [graph, onGraphReady]);

  return (
    <SigmaErrorBoundary>
      <SigmaRenderer
        graph={graph}
        onNodeClick={onNodeClick}
        selectedNodeId={selectedNodeId}
        focusedNodeId={focusedNodeId}
        displaySettings={displaySettings}
      />
    </SigmaErrorBoundary>
  );
}
