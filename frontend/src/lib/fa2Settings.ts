/**
 * Slider → ForceAtlas2 settings mapping (D-07).
 *
 * No React, no sigma runtime imports — fully unit-testable in jsdom.
 */

export interface FA2SliderInput {
  centerForce: number;
  repelForce: number;
  linkForce: number;
  linkDistance: number;
}

export interface FA2Settings {
  gravity: number;
  scalingRatio: number;
  edgeWeightInfluence: number;
  slowDown: number;
  adjustSizes: boolean;
  barnesHutOptimize: boolean;
  linLogMode: boolean;
  outboundAttractionDistribution: boolean;
  strongGravityMode: boolean;
}

/**
 * Map graphDisplayStore slider values (0–100) onto ForceAtlas2 algorithm settings.
 *
 * Mapping table (from UI-SPEC §7):
 *   gravity              = (centerForce  / 100) * 2          → 0–2
 *   scalingRatio         = 1 + (repelForce / 100) * 19       → 1–20
 *   edgeWeightInfluence  = (linkForce / 100) * 2             → 0–2
 *   slowDown             = 1 + (linkDistance / 100) * 19     → 1–20
 *
 * Structural flags are constants (not slider-driven): linLogMode +
 * outboundAttractionDistribution cluster communities with whitespace between
 * them and push hubs apart (the gephi/Obsidian look). adjustSizes is OFF on
 * purpose — anticollision forces uniform spacing that erases the community gaps.
 */
export function toFA2Settings(display: FA2SliderInput): FA2Settings {
  return {
    gravity: (display.centerForce / 100) * 2,
    scalingRatio: 1 + (display.repelForce / 100) * 19,
    edgeWeightInfluence: (display.linkForce / 100) * 2,
    slowDown: 1 + (display.linkDistance / 100) * 19,
    adjustSizes: false,
    barnesHutOptimize: true,
    // linLogMode OFF: LinLog uses logarithmic (weak-at-distance) attraction, so a
    // high-degree node like the walk seed loses the tug-of-war against its
    // neighbors' collective repulsion and gets flung outside its own cluster.
    // Standard linear attraction grows with distance and pulls the seed firmly
    // into its cluster centroid.
    linLogMode: false,
    // outboundAttractionDistribution OFF: when on, FA2 "dissuades hubs" — weakens
    // attraction on high-degree nodes and pushes them to the borders (same
    // seed-fling failure mode).
    outboundAttractionDistribution: false,
    strongGravityMode: false,
  };
}
