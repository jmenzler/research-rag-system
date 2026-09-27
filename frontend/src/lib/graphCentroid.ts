/**
 * Edge-aware centroid placement for new expand nodes (D-01).
 *
 * No React, no sigma runtime imports — fully unit-testable in jsdom.
 */

/**
 * Compute an initial position for a new node by finding the centroid of its
 * already-placed neighbors, with a fallback chain.
 *
 * Fallback chain:
 *   1. Mean x/y of already-placed neighbors (edges where the other endpoint
 *      is in existingPositions — edges between two new nodes are excluded).
 *   2. If no placed neighbors: use existingPositions[fallbackNodeId] when defined.
 *   3. Otherwise: random position in [0, 10].
 *
 * Jitter is applied to the chosen anchor:
 *   x = anchor.x + (Math.random() - 0.5) * jitterRadius  (default 20)
 */
export function computeCentroid(
  newNodeId: number,
  edges: Array<{ fromId: number; toId: number }>,
  existingPositions: Map<number, { x: number; y: number }>,
  fallbackNodeId: number | null,
  jitterRadius?: number,
): { x: number; y: number } {
  const jitter = jitterRadius ?? 20;

  const placedNeighbors: Array<{ x: number; y: number }> = [];
  for (const e of edges) {
    let otherId: number | null = null;
    if (e.fromId === newNodeId) otherId = e.toId;
    else if (e.toId === newNodeId) otherId = e.fromId;

    if (otherId !== null && existingPositions.has(otherId)) {
      placedNeighbors.push(existingPositions.get(otherId)!);
    }
  }

  let anchor: { x: number; y: number };

  if (placedNeighbors.length > 0) {
    const sumX = placedNeighbors.reduce((acc, p) => acc + p.x, 0);
    const sumY = placedNeighbors.reduce((acc, p) => acc + p.y, 0);
    anchor = { x: sumX / placedNeighbors.length, y: sumY / placedNeighbors.length };
  } else if (fallbackNodeId !== null && existingPositions.has(fallbackNodeId)) {
    anchor = existingPositions.get(fallbackNodeId)!;
  } else {
    anchor = { x: Math.random() * 10, y: Math.random() * 10 };
  }

  return {
    x: anchor.x + (Math.random() - 0.5) * jitter,
    y: anchor.y + (Math.random() - 0.5) * jitter,
  };
}
