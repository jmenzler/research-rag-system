/**
 * Liveness guard for raw Sigma method calls (04.1-03).
 *
 * No sigma/react imports — pure, unit-testable in jsdom without WebGL.
 *
 * React StrictMode (dev) double-invokes effects mount→cleanup→mount. On cleanup
 * @react-sigma kills the throwaway first-mount Sigma, which empties
 * `sigma.nodePrograms` (WebGL programs torn down) while leaving settings intact.
 * Any raw sigma method that reprocesses nodes (refresh→process→addNodeToProgram)
 * then looks up nodePrograms["circle"] on the dead instance, finds nothing, and
 * throws "could not find a suitable program for node type circle". @react-sigma's
 * own hooks are StrictMode-safe; our raw refresh/setSetting/createCanvas effects
 * must guard with this.
 */
export function isLiveSigma(sigma: unknown): boolean {
  const s = sigma as { nodePrograms?: Record<string, unknown> } | null;
  return !!s && !!s.nodePrograms && Object.keys(s.nodePrograms).length > 0;
}
