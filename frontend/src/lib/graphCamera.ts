/**
 * Pure bounding-box camera math for Sigma graph fit-to-view (D-06).
 *
 * No React, no sigma runtime imports — fully unit-testable in jsdom.
 * The side-effecting fitCamera wrapper (which calls sigma.getCamera().animate())
 * lives in Plan 03 where sigma is available.
 */

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Viewport {
  w: number;
  h: number;
}

export interface FitResult {
  x: number;
  y: number;
  ratio: number;
}

/**
 * Compute camera position and ratio to fit a bounding box into a viewport.
 *
 * Formula (20% padding):
 *   cx = (minX + maxX) / 2
 *   cy = (minY + maxY) / 2
 *   graphW = maxX - minX + 1
 *   graphH = maxY - minY + 1
 *   ratio = Math.max(graphW/w, graphH/h) * 1.2
 *
 * Returns null if the bounding box contains no finite coordinates (empty graph).
 */
export function computeFit(box: BoundingBox, viewport: Viewport): FitResult | null {
  if (!isFinite(box.minX) || !isFinite(box.minY) || !isFinite(box.maxX) || !isFinite(box.maxY)) {
    return null;
  }

  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  const graphW = box.maxX - box.minX + 1;
  const graphH = box.maxY - box.minY + 1;
  const ratio = Math.max(graphW / viewport.w, graphH / viewport.h) * 1.2;

  return { x: cx, y: cy, ratio };
}
