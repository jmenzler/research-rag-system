/**
 * Sigma-specific renderer for GraphCanvas. Dynamically imported from
 * GraphCanvas.tsx to isolate sigma's module-level WebGL2RenderingContext access
 * from test environments that don't have WebGL.
 *
 * This file statically imports sigma and @react-sigma/core. It is only loaded
 * when a GraphCanvas component mounts in a browser environment.
 */
import { SigmaContainer, useRegisterEvents, useSigma } from "@react-sigma/core";
import "@react-sigma/core/lib/style.css";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import type Graph from "graphology";
import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
} from "react";
import { buildEdgeReducer, buildNodeReducer } from "@/lib/graphReducers";
import { computeShortestPath, type GraphEdgeInput } from "@/lib/graphBFS";
import type { GraphDisplaySettings } from "@/state/graphDisplayStore";
import { useGraphStore } from "@/state/graphStore";
import { toFA2Settings } from "@/lib/fa2Settings";
import { isLiveSigma } from "@/lib/sigmaLiveness";
import { ZoomControls } from "@/components/graph/ZoomControls";

// ---------------------------------------------------------------------------
// Overlay canvas constants (SPIKE-RING.md confirmed approach)
// ---------------------------------------------------------------------------

const OVERLAY_LAYER_ID = "node-state-overlay";

// Frames of FA2 to run after a graph load or a Forces-slider change before the
// layout idles (~5s at 60fps). A slider drag re-bumps this so motion is live.
const REHEAT_FRAMES = 300;

// Only hide edges during motion once the graph is big enough that re-reducing +
// re-rendering every edge per frame would freeze the main thread (a real walk is
// ~10k edges). Below this, edges render every frame cheaply — hiding them just
// makes them vanish whenever the layout moves (e.g. continuous Animate mode),
// which is the bug, not the freeze fix. Edges always show on a settled layout.
const EDGE_HIDE_MIN_EDGES = 2000;

function hasOverlayLayer(sigma: ReturnType<typeof useSigma>): boolean {
  return OVERLAY_LAYER_ID in sigma.getCanvases();
}

// ---------------------------------------------------------------------------
// FA2 inner component — runs inside SigmaContainer context
// ---------------------------------------------------------------------------

interface FA2Props {
  graph: Graph;
  onNodeClick?: ((corpusId: number) => void) | undefined;
  selectedNodeId?: number | null | undefined;
  focusedNodeId?: string | null | undefined;
  displaySettings?: GraphDisplaySettings | undefined;
}

function FA2Manager({
  graph,
  onNodeClick,
  selectedNodeId,
  focusedNodeId,
  displaySettings,
}: FA2Props): ReactElement | null {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();
  // Per-setter selectors (stable refs) — this component must NOT re-render when
  // unrelated store slices (overlay/in-flight) change, or a walk re-render storm
  // thrashes the WebGL canvas.
  const setFocusedNodeId = useGraphStore((s) => s.setFocusedNodeId);
  const clearFocus = useGraphStore((s) => s.clearFocus);
  const walkSeedId = useGraphStore((s) => s.walkSeedId);
  const hoveredPathNodeSet = useGraphStore((s) => s.hoveredPathNodeSet);
  const hoveredPathEdgeSet = useGraphStore((s) => s.hoveredPathEdgeSet);
  const setHoveredPath = useGraphStore((s) => s.setHoveredPath);
  const clearHoveredPath = useGraphStore((s) => s.clearHoveredPath);

  // FA2 settings derived from the LIVE display sliders every render, so dragging
  // a Forces slider re-tunes the running simulation in real time — no pointer-up
  // gate, no worker restart. settingsRef lets the animation loop read the latest
  // values without re-subscribing.
  const liveSettings = useMemo(() => {
    const d = displaySettings ?? {
      centerForce: 40,
      repelForce: 55,
      linkForce: 50,
      linkDistance: 50,
    };
    return toFA2Settings(d);
  }, [displaySettings]);
  const settingsRef = useRef(liveSettings);
  settingsRef.current = liveSettings;

  const animate = displaySettings?.animate ?? false;
  const animateRef = useRef(animate);
  animateRef.current = animate;

  // Reheat budget: frames of simulation left to run. Re-bumped whenever the
  // graph or force settings change, so a slider drag reflows the layout live;
  // topped up continuously while Animate is on.
  const energyRef = useRef(REHEAT_FRAMES);
  // True while the layout is actively moving (reheat draining or Animate on).
  // The edgeReducer reads this to hide edges during motion — a real walk is
  // ~10k edges, and re-reducing + re-rendering them every frame is the freeze.
  const layoutRunningRef = useRef(true);
  useEffect(() => {
    energyRef.current = REHEAT_FRAMES;
    layoutRunningRef.current = true;
  }, [liveSettings, graph]);

  // Single main-thread FA2 loop. Replaces the @react-sigma web-worker supervisor,
  // whose settings were frozen at construction and whose kill() was terminal
  // (every re-tune threw "worker.start: layout was killed"). One iteration per
  // frame is cheap for typical graphs; barnesHutOptimize keeps large ones viable.
  // Drains the reheat budget then idles; the final frame does a full refresh so
  // the spatial index (hover hit-testing) is rebuilt after motion stops.
  useEffect(() => {
    if (!isLiveSigma(sigma)) return undefined;
    let raf = 0;
    const tick = (): void => {
      if (graph.order > 0 && (animateRef.current || energyRef.current > 0)) {
        forceAtlas2.assign(graph, { iterations: 1, settings: settingsRef.current });
        if (!animateRef.current && energyRef.current > 0) energyRef.current -= 1;
        const justStopped = !animateRef.current && energyRef.current <= 0;
        // Reveal edges on the settle frame: flip BEFORE the refresh below so its
        // edgeReducer pass renders them visible in one shot (hidden during motion).
        if (justStopped) layoutRunningRef.current = false;
        // On settle, nudge overlapping nodes apart (anticollision) without the
        // uniform-spacing that FA2 adjustSizes forces. The bbox is dominated by
        // the spread disconnected field, so this de-overlap is local and survives
        // autoRescale instead of being renormalized away.
        if (justStopped) {
          try {
            // noverlap has no fixed-attribute handling (its iterate.js moves every
            // node), so it drifts the pinned seed off (0,0) and nudges every
            // fixed:true node — silently breaking the GRAPH-09 pin invariant.
            // Snapshot pinned positions, run de-overlap, then restore them.
            const pinned: Array<[string, number, number]> = [];
            graph.forEachNode((key, attrs) => {
              if (attrs.fixed === true) {
                pinned.push([key, attrs.x as number, attrs.y as number]);
              }
            });
            noverlap.assign(graph, {
              maxIterations: 80,
              settings: { margin: 3, ratio: 1.3, expansion: 1.1, gridSize: 20 },
            });
            for (const [key, px, py] of pinned) {
              graph.setNodeAttribute(key, "x", px);
              graph.setNodeAttribute(key, "y", py);
            }
          } catch {
            // best-effort; never break the render loop
          }
        }
        if (isLiveSigma(sigma)) sigma.refresh({ skipIndexation: !justStopped });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [sigma, graph]);

  // Auto-fit on new graph (D-06): 500ms after graph reference changes, reset camera.
  // Uses animatedReset (camera-space aware) — no raw graphology coord math needed.
  const graphRef = useRef<Graph>(graph);
  useEffect(() => {
    if (graphRef.current === graph) return;
    graphRef.current = graph;
    const id = setTimeout(() => {
      if (isLiveSigma(sigma)) {
        // ratio > 1 zooms OUT — autoRescale already frames the node cloud, this
        // just leaves margin so communities read as separated, not edge-to-edge.
        sigma.getCamera().animate(
          { x: 0.5, y: 0.5, ratio: 1.4, angle: 0 },
          { duration: 350 },
        );
      }
    }, 500);
    return () => clearTimeout(id);
  }, [sigma, graph]);

  // Full refresh on graph-data change: re-index every node/edge into the program
  // buffers and reprocess. Without this, nodes that exist in graphology are never
  // written to the WebGL node program (scheduleRender only redraws already-indexed
  // state), so a freshly-loaded corpus renders an empty canvas. refresh() forces
  // process()→addNodeToProgram — GUARDED against a StrictMode-killed instance whose
  // nodePrograms is empty (would throw "no program for circle").
  useEffect(() => {
    if (!isLiveSigma(sigma)) return;
    sigma.refresh();
  }, [sigma, graph]);

  // Keep the canvas sized to its container on sidebar toggle / window resize.
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      if (isLiveSigma(sigma)) sigma.resize();
    });
    ro.observe(sigma.getContainer());
    return () => ro.disconnect();
  }, [sigma]);

  // NOTE: no manual fit-to-view. sigma's autoRescale (default) normalizes the
  // graph into framed [0,1] space and the default camera state {x:0.5,y:0.5,
  // ratio:1} already fits the whole node cloud — exactly how Wave-0 rendered all
  // 238 nodes on-screen. The previous manual computeFit fed RAW graphology
  // attrs.x/y (e.g. ~3.77) into camera.animate, but the camera operates in the
  // FRAMED/normalized space, which threw the viewport thousands of px off the
  // nodes (framedGraphToViewport ~(2088,-2714) on a 604×664 canvas → empty).
  // The zoom controls (Plan: ZoomControls) use sigma's animatedZoom/Reset, which
  // are already in the correct coordinate space.

  // Apply all visual settings (label/camera/edge-type) via setSetting() — a
  // single-key MERGE that never replaces nodeProgramClasses, so it cannot trigger
  // handleSettingsUpdate's program-unregister loop (which dropped "circle" when we
  // fed SigmaContainer a fuller settings snapshot). This is the D-01/D-02/D-03
  // styling that the minimal BASE_SETTINGS constructor intentionally omits.
  useEffect(() => {
    if (!isLiveSigma(sigma)) return;
    const d = displaySettings ?? { arrowsOn: true, textFade: 40 };
    sigma.setSetting("defaultEdgeType", (d.arrowsOn ?? true) ? "arrow" : "line");
    sigma.setSetting("labelFont", "Fira Code, ui-monospace, Menlo, monospace");
    sigma.setSetting("labelSize", 10.5);
    sigma.setSetting("labelWeight", "400");
    sigma.setSetting("labelColor", { color: "#8B98AD" });
    // D-07: raised base from 6 → 10 so dense walk graphs cull more labels.
    // 10 is a HUMAN-UAT starting value; tune to 12–14 if overlap persists.
    const LABEL_THRESHOLD_BASE = 10; // tune-live: raise to 12–14 if spaghetti persists
    sigma.setSetting("labelRenderedSizeThreshold", LABEL_THRESHOLD_BASE + ((d.textFade ?? 40) / 100) * 20);
    sigma.setSetting("minCameraRatio", 0.05);
    sigma.setSetting("maxCameraRatio", 5);
  }, [sigma, displaySettings]);

  // Ego-highlight: compute neighbor set for the focused node.
  const neighborIds = useMemo<Set<string>>(() => {
    if (!focusedNodeId || !graph.hasNode(focusedNodeId)) return new Set();
    return new Set(graph.neighbors(focusedNodeId));
  }, [focusedNodeId, graph]);

  // Edge list for hover shortest-path, memoized on the graph reference. Without
  // this, every pointer-enter in walk mode re-mapped all g.edges() (a ~10k-edge
  // walk = 10k allocs + 20k Number() parses on the main thread per hover).
  const hoverEdgeList = useMemo<GraphEdgeInput[]>(
    () =>
      graph.edges().map((ek) => ({
        fromId: Number(graph.source(ek)),
        toId: Number(graph.target(ek)),
      })),
    [graph],
  );
  const hoverEdgeListRef = useRef(hoverEdgeList);
  hoverEdgeListRef.current = hoverEdgeList;

  // nodeReducer / edgeReducer — built from pure Wave-0 factory functions.
  const selectedIdStr = selectedNodeId !== null && selectedNodeId !== undefined
    ? String(selectedNodeId)
    : null;

  // Apply nodeReducer + edgeReducer via sigma.setSetting() so they update
  // whenever focusedNodeId / neighborIds / display values / path sets change.
  // Per Pitfall 3: path sets flow through refs read inside the stable reducer
  // function rather than as factory inputs — no setSetting storm on every hover.
  // Cast via unknown: our typed reducers are structurally compatible with sigma's
  // Attributes-typed signature (sigma only cares about the shape at runtime).
  useEffect(() => {
    if (!isLiveSigma(sigma)) return;
    const reducer = buildNodeReducer(
      focusedNodeId ?? null,
      neighborIds,
      selectedIdStr,
      displaySettings ?? { nodeSize: 60 },
      pathNodeSetRef.current,
    );
    sigma.setSetting("nodeReducer", reducer as unknown as Parameters<typeof sigma.setSetting<"nodeReducer">>[1]);
  }, [sigma, focusedNodeId, neighborIds, selectedIdStr, displaySettings, hoveredPathNodeSet]);

  useEffect(() => {
    if (!isLiveSigma(sigma)) return;
    const reducer = buildEdgeReducer(
      focusedNodeId ?? null,
      neighborIds,
      displaySettings ?? { linkThick: 50 },
      (edge: string) => ({
        source: sigma.getGraph().source(edge),
        target: sigma.getGraph().target(edge),
      }),
      () => layoutRunningRef.current && graph.size > EDGE_HIDE_MIN_EDGES,
      pathEdgeSetRef.current,
    );
    sigma.setSetting("edgeReducer", reducer as unknown as Parameters<typeof sigma.setSetting<"edgeReducer">>[1]);
  }, [sigma, focusedNodeId, neighborIds, displaySettings, graph, hoveredPathEdgeSet]);

  // Hover tooltip state
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);

  // Register all Sigma events: click, hover, keyboard
  useEffect(() => {
    const events: Parameters<typeof registerEvents>[0] = {
      clickNode: ({ node, event }) => {
        const attrs = sigma.getGraph().getNodeAttributes(node);
        const cid = attrs.corpusId as number | undefined;
        if (cid !== undefined) onNodeClick?.(cid);
        setFocusedNodeId(node);
        const e = event as { x?: number; y?: number };
        setTooltip(null);
        void e;
      },
      clickStage: () => {
        clearFocus();
      },
      enterNode: ({ node, event }) => {
        const attrs = sigma.getGraph().getNodeAttributes(node);
        const title = (attrs.fullTitle as string | undefined) ?? (attrs.label as string);
        const cit = attrs.citationCount as number | null | undefined;
        const citText = cit !== null && cit !== undefined
          ? cit.toLocaleString()
          : "0";
        const e = event as { x?: number; y?: number };
        const containerRect = sigma.getContainer().getBoundingClientRect();
        setTooltip({
          x: (e.x ?? 0) - containerRect.left,
          y: (e.y ?? 0) - containerRect.top - 12,
          text: `"${title}" · ${citText} citations`,
        });
        // D-03: compute hover path if walk mode is active
        if (walkSeedId !== null) {
          const g = sigma.getGraph();
          const path = computeShortestPath(
            Number(walkSeedId),
            Number(node),
            hoverEdgeListRef.current,
          );
          if (path.length > 0) {
            const nodeSet = new Set(path.map(String));
            const edgeSet = new Set<string>();
            for (let i = 0; i < path.length - 1; i++) {
              const a = String(path[i]);
              const b = String(path[i + 1]);
              const ek = g.edge(a, b) ?? g.edge(b, a);
              if (ek !== undefined) edgeSet.add(ek);
            }
            setHoveredPath(nodeSet, edgeSet);
          }
        }
      },
      leaveNode: () => {
        setTooltip(null);
        clearHoveredPath();
      },
    };
    registerEvents(events);
  }, [sigma, registerEvents, onNodeClick, setFocusedNodeId, clearFocus, walkSeedId, setHoveredPath, clearHoveredPath]);

  // Esc key clears focus — listener on the sigma container.
  useEffect(() => {
    const container = sigma.getContainer();
    const handleKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") clearFocus();
    };
    container.addEventListener("keydown", handleKey);
    return () => container.removeEventListener("keydown", handleKey);
  }, [sigma, clearFocus]);

  // ---------------------------------------------------------------------------
  // afterRender overlay — node state rings, arcs, halos (SPIKE-RING.md contract)
  // ---------------------------------------------------------------------------

  // Dynamic focus state fed to the draw closure via refs so the overlay layer is
  // created exactly ONCE per sigma instance (effect keyed on [sigma] only). The
  // draw function reads the latest focus/neighbor state without re-registering
  // the afterRender listener or re-running createCanvas/killLayer on focus change
  // — re-running createCanvas mid-mount races sigma's own layer setup and threw
  // "reading 'after'" when this.elements["hovers"] wasn't present yet.
  const focusedRef = useRef<string | null>(focusedNodeId ?? null);
  const neighborRef = useRef<Set<string>>(neighborIds);
  focusedRef.current = focusedNodeId ?? null;
  neighborRef.current = neighborIds;
  // D-03: path-set refs — updated in render body (not as effect deps) so the
  // overlay effect is keyed on [sigma] only and never re-creates the canvas layer.
  const pathNodeSetRef = useRef<Set<string>>(hoveredPathNodeSet);
  const pathEdgeSetRef = useRef<Set<string>>(hoveredPathEdgeSet);
  pathNodeSetRef.current = hoveredPathNodeSet;
  pathEdgeSetRef.current = hoveredPathEdgeSet;
  // Holds the latest overlay draw fn so focus changes can clear+repaint the
  // overlay directly, rather than hoping sigma re-emits afterRender (it doesn't
  // reliably for a custom layer → stale halos/rings piled up across selections).
  const drawRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Guard against a StrictMode-killed instance — creating a canvas / scheduling
    // a render on a dead sigma is wasted work at best and can race teardown.
    if (!isLiveSigma(sigma)) return;
    // Idempotent creation. createCanvas with afterLayer:"hovers" reads
    // this.elements["hovers"].after(...) at runtime — if "hovers" is somehow
    // absent (instance still constructing), append without afterLayer so we
    // never throw. The contract layer order (above hovers, below mouse) is the
    // happy path; the fallback degrades stack position, not correctness.
    if (!hasOverlayLayer(sigma)) {
      const hoversReady = "hovers" in sigma.getCanvases();
      sigma.createCanvas(
        OVERLAY_LAYER_ID,
        hoversReady
          ? { style: { pointerEvents: "none" }, afterLayer: "hovers" }
          : { style: { pointerEvents: "none" } },
      );
    }

    function draw(): void {
      // Bail if layer is gone (late frame after cleanup).
      const overlayCanvas = sigma.getCanvases()[OVERLAY_LAYER_ID];
      if (!overlayCanvas) return;

      const dpr = window.devicePixelRatio || 1;
      const { width, height } = sigma.getDimensions();

      // Resize physical pixels to match sigma viewport × DPR every frame
      // (handles resize + HiDPI automatically).
      if (
        overlayCanvas.width !== width * dpr ||
        overlayCanvas.height !== height * dpr
      ) {
        overlayCanvas.width = width * dpr;
        overlayCanvas.height = height * dpr;
      }

      const ctx = overlayCanvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      const focused = focusedRef.current;
      const neighbors = neighborRef.current;
      const pathNodes = pathNodeSetRef.current;
      const g = sigma.getGraph();

      g.forEachNode((nodeKey, attrs) => {
        const d = sigma.getNodeDisplayData(nodeKey);
        if (!d) return;

        const { x: vx, y: vy } = sigma.framedGraphToViewport({ x: d.x, y: d.y });
        // All draw coords × dpr for HiDPI clarity.
        const cx = vx * dpr;
        const cy = vy * dpr;
        const r = sigma.scaleSize(d.size);

        const status = attrs.status as string | undefined;
        const isFocused = nodeKey === focused;
        const isNeighbor = neighbors.has(nodeKey);
        // When a node is focused, dim every decoration that isn't on the focused
        // node or one of its neighbors — matches the disc dimming in the
        // nodeReducer so the selected ego-network stands alone.
        const dim = focused !== null && !isFocused && !isNeighbor;
        const dimFactor = dim ? 0.12 : 1;

        // --- missing: dashed border ON the disc edge (texture, not a floating
        //     ring) so unresolved nodes read as dashed-outlined dots ---
        if (status === "missing") {
          ctx.save();
          ctx.globalAlpha = dimFactor;
          ctx.beginPath();
          ctx.arc(cx, cy, r * dpr, 0, Math.PI * 2);
          ctx.setLineDash([2.5 * dpr, 2.5 * dpr]);
          ctx.strokeStyle = "#AEB9CC";
          ctx.lineWidth = 1.25 * dpr;
          ctx.stroke();
          ctx.restore();
        }

        // --- seed: outer ring at r+3, #22C55E opacity 0.5 ---
        if (status === "seed") {
          ctx.save();
          ctx.globalAlpha = 0.5 * dimFactor;
          ctx.beginPath();
          ctx.arc(cx, cy, (r + 3) * dpr, 0, Math.PI * 2);
          ctx.setLineDash([]);
          ctx.strokeStyle = "#22C55E";
          ctx.lineWidth = 1.5 * dpr;
          ctx.stroke();
          ctx.restore();
        }

        // --- ingesting: progress arc + center dot ---
        if (status === "ingesting") {
          const progress = (attrs.progress as number | undefined) ?? 0;
          if (progress > 0) {
            const arcR = (r + 2) * dpr;
            const startAngle = -Math.PI / 2;
            const endAngle = startAngle + Math.PI * 2 * progress;
            ctx.save();
            ctx.globalAlpha = dimFactor;
            ctx.setLineDash([]);
            ctx.strokeStyle = "#F59E0B";
            ctx.lineWidth = 2 * dpr;
            ctx.beginPath();
            ctx.arc(cx, cy, arcR, startAngle, endAngle);
            ctx.stroke();
            // Amber center dot
            ctx.beginPath();
            ctx.arc(cx, cy, r * 0.32 * dpr, 0, Math.PI * 2);
            ctx.fillStyle = "#F59E0B";
            ctx.fill();
            ctx.restore();
          }
        }

        // --- neighbor ring: r+4, #3B82F6 opacity 0.5 width 1 ---
        if (isNeighbor && !isFocused && focused !== null) {
          ctx.save();
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.arc(cx, cy, (r + 4) * dpr, 0, Math.PI * 2);
          ctx.setLineDash([]);
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 1 * dpr;
          ctx.stroke();
          ctx.restore();
        }

        // --- D-03 path ring: solid blue r+4, only when no ego-focus (OQ-3) ---
        if (focused === null && pathNodes.size > 0 && pathNodes.has(nodeKey)) {
          ctx.save();
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.arc(cx, cy, (r + 4) * dpr, 0, Math.PI * 2);
          ctx.setLineDash([]);
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 2 * dpr;
          ctx.stroke();
          ctx.restore();
        }

        // --- selection halo (focused node): outer glow + inner ring ---
        if (isFocused) {
          ctx.save();
          // Outer glow fill disc at r+14
          ctx.beginPath();
          ctx.arc(cx, cy, (r + 14) * dpr, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(59,130,246,0.10)";
          ctx.fill();
          // Inner ring at r+7
          ctx.beginPath();
          ctx.arc(cx, cy, (r + 7) * dpr, 0, Math.PI * 2);
          ctx.setLineDash([]);
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 1.6 * dpr;
          ctx.stroke();
          ctx.restore();
        }
      });
    }

    drawRef.current = draw;
    sigma.on("afterRender", draw);
    // Schedule one paint so decorations show on the next frame. scheduleRender()
    // emits afterRender WITHOUT forcing a node-program reprocess (process() only
    // runs when needToProcess is already set). A full refresh() would reprocess
    // every node and hit any program-registration edge case — we only need the
    // afterRender hook to fire to repaint the ref-driven overlay.
    sigma.scheduleRender();

    return () => {
      drawRef.current = null;
      sigma.off("afterRender", draw);
      // Guarded killLayer — throws if absent (SPIKE-RING.md GOTCHA).
      if (hasOverlayLayer(sigma)) {
        sigma.killLayer(OVERLAY_LAYER_ID);
      }
    };
    // Keyed on [sigma] only: the overlay layer is created/killed once per sigma
    // instance. Dynamic focus state flows through focusedRef/neighborRef.
  }, [sigma]);

  // Repaint the overlay when focus or path set changes. Call the draw fn DIRECTLY
  // (clear + redraw) instead of only scheduling a sigma render — sigma does not
  // reliably re-emit afterRender for a custom layer on focus-only changes, which
  // left each selection's halo/neighbor rings frozen on the overlay and piling up
  // across clicks. scheduleRender still runs so the WebGL disc dimming updates.
  useEffect(() => {
    if (!isLiveSigma(sigma)) return;
    drawRef.current?.();
    sigma.scheduleRender();
  }, [sigma, focusedNodeId, neighborIds, hoveredPathNodeSet]);

  return (
    <>
      {/* Hover tooltip — positioned absolute over canvas */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x,
            top: tooltip.y,
            transform: "translate(-50%, -100%)",
            pointerEvents: "none",
            zIndex: 20,
            maxWidth: 320,
            padding: "4px 8px",
            borderRadius: 4,
            background: "rgba(24,29,41,0.92)",
            border: "1px solid var(--p3-border)",
            color: "var(--p3-fg)",
            fontSize: 12,
            fontFamily: "var(--p3-font-mono)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {tooltip.text}
        </div>
      )}
      {/* Zoom controls overlay — inside SigmaContainer so useSigma() resolves */}
      <ZoomControls graph={graph} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Settings resolver — computes Sigma settings from display store values
// ---------------------------------------------------------------------------

// Minimal constructor settings — IDENTICAL to the Wave-0 base that rendered all
// 238 nodes in the browser. Critically, NO nodeProgramClasses: sigma's built-in
// circle/arrow programs are registered by resolveSettings() at construction and
// survive because we never pass a (foreign-instance / partial) nodeProgramClasses
// snapshot that would trigger handleSettingsUpdate's unregister loop. All the
// label/camera/edge-type styling is applied post-mount via sigma.setSetting()
// (single-key merge — never touches nodeProgramClasses), see applyVisualSettings.
const BASE_SETTINGS: Record<string, unknown> = {
  defaultNodeType: "circle",
  defaultEdgeType: "arrow",
  allowInvalidContainer: true,
};

// ---------------------------------------------------------------------------
// SigmaRenderer — accepts a pre-built graphology Graph
// ---------------------------------------------------------------------------

export interface SigmaRendererProps {
  graph: Graph;
  onNodeClick?: ((corpusId: number) => void) | undefined;
  selectedNodeId?: number | null | undefined;
  focusedNodeId?: string | null | undefined;
  displaySettings?: GraphDisplaySettings | undefined;
}

export function SigmaRenderer({
  graph,
  onNodeClick,
  selectedNodeId,
  focusedNodeId,
  displaySettings,
}: SigmaRendererProps): ReactElement {
  // Absolute fill (not height:100%) — GraphPage's wrapper is `relative`, and a
  // percentage height does not resolve reliably at sigma mount time inside the
  // flex chain, which makes sigma throw "Container has no height".
  //
  // settings is the static minimal trio (BASE_SETTINGS). It NEVER changes ref, so
  // SigmaContainer's isEqual check never re-creates the instance and the built-in
  // circle/arrow programs registered at construction survive. Visual settings are
  // applied post-mount via setSetting() inside FA2Manager.
  return (
    <Suspense fallback={null}>
      <SigmaContainer
        graph={graph}
        style={{ position: "absolute", inset: 0, background: "var(--p3-bg)" }}
        settings={BASE_SETTINGS}
      >
        <FA2Manager
          graph={graph}
          onNodeClick={onNodeClick}
          selectedNodeId={selectedNodeId}
          focusedNodeId={focusedNodeId}
          displaySettings={displaySettings}
        />
      </SigmaContainer>
    </Suspense>
  );
}
