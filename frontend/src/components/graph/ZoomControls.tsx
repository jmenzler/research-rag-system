/**
 * ZoomControls — +/−/fit overlay rendered inside SigmaContainer.
 *
 * Must be a child of <SigmaContainer> so useSigma() resolves.
 * Position: absolute bottom-3 left-3 over the canvas.
 * UI-SPEC §5: 3 buttons (32×32), flex-col, rounded-md, var(--p3-surface) bg.
 */
import { useSigma } from "@react-sigma/core";
import type Graph from "graphology";
import { Maximize2, Minus, Plus } from "lucide-react";
import { type ReactElement, useCallback, useEffect } from "react";

interface ZoomControlsProps {
  graph?: Graph;
}

// Leave 20% headroom around the node cloud so communities read as separated.
const FIT_PADDING = 1.2;

export function ZoomControls({ graph }: ZoomControlsProps): ReactElement {
  const sigma = useSigma();

  const handleZoomIn = useCallback((): void => {
    sigma.getCamera().animatedZoom({ duration: 250 });
  }, [sigma]);

  const handleZoomOut = useCallback((): void => {
    sigma.getCamera().animatedUnzoom({ duration: 250 });
  }, [sigma]);

  const handleFit = useCallback((): void => {
    const camera = sigma.getCamera();
    if (!graph || graph.order === 0) {
      camera.animatedReset({ duration: 350 });
      return;
    }

    // Build the bounding box in viewport-pixel space: getNodeDisplayData returns
    // framed/normalized coords, so we project each node through
    // framedGraphToViewport before measuring. Mixing framed extents with pixel
    // container dimensions (the old computeFit call) yielded ratios ~0.002 — an
    // extreme zoom-in onto a single node.
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    graph.forEachNode((node) => {
      const d = sigma.getNodeDisplayData(node);
      if (!d) return;
      const v = sigma.framedGraphToViewport({ x: d.x, y: d.y });
      minX = Math.min(minX, v.x);
      maxX = Math.max(maxX, v.x);
      minY = Math.min(minY, v.y);
      maxY = Math.max(maxY, v.y);
    });

    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
      camera.animatedReset({ duration: 350 });
      return;
    }

    const container = sigma.getContainer();
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    const boxW = maxX - minX;
    const boxH = maxY - minY;

    // Camera ratio scales linearly with how much framed space fills the viewport,
    // so the target ratio is the current ratio times the fraction of the viewport
    // the box currently occupies (plus padding). The new center is the box center
    // converted back into framed-graph space.
    const state = camera.getState();
    const ratio =
      state.ratio * Math.max(boxW / w, boxH / h, 1e-6) * FIT_PADDING;
    const center = sigma.viewportToFramedGraph({
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
    });

    camera.animate(
      { x: center.x, y: center.y, ratio, angle: state.angle },
      { duration: 350 },
    );
  }, [sigma, graph]);

  // Contextual hotkey listeners — attach only while ZoomControls is mounted
  // (i.e. the graph route is active). Reuses the same handlers as the buttons
  // so no camera logic is duplicated (D-08). Keyed on the handlers so a graph
  // rebuild (walk/expand/corpus-switch) re-binds the fit shortcut to the fresh
  // graph closure instead of fitting the original (often empty) reference.
  useEffect(() => {
    window.addEventListener("hotkey:graph-zoom-in", handleZoomIn);
    window.addEventListener("hotkey:graph-zoom-out", handleZoomOut);
    window.addEventListener("hotkey:graph-fit", handleFit);
    return () => {
      window.removeEventListener("hotkey:graph-zoom-in", handleZoomIn);
      window.removeEventListener("hotkey:graph-zoom-out", handleZoomOut);
      window.removeEventListener("hotkey:graph-fit", handleFit);
    };
  }, [handleZoomIn, handleZoomOut, handleFit]);

  const btnClass =
    "flex w-8 h-8 items-center justify-center text-[var(--p3-muted)] " +
    "hover:bg-white/5 transition-colors";

  const hairline = (
    <div className="h-px w-full bg-[var(--p3-border)]" aria-hidden />
  );

  return (
    <div
      className="absolute bottom-3 left-3 flex flex-col rounded-md overflow-hidden shadow-xl"
      style={{
        backgroundColor: "var(--p3-surface)",
        border: "1px solid var(--p3-border)",
        zIndex: 10,
      }}
    >
      <button
        type="button"
        aria-label="Zoom in"
        onClick={handleZoomIn}
        className={btnClass}
      >
        <Plus size={13} />
      </button>
      {hairline}
      <button
        type="button"
        aria-label="Zoom out"
        onClick={handleZoomOut}
        className={btnClass}
      >
        <Minus size={13} />
      </button>
      {hairline}
      <button
        type="button"
        aria-label="Fit graph to view"
        onClick={handleFit}
        className={btnClass}
      >
        <Maximize2 size={13} />
      </button>
    </div>
  );
}
