/**
 * Formatting helpers for the cost/latency/tokens badge (D-13 / CHAT-13).
 *
 * Format spec (01-RESEARCH.md §"Cost + Latency Badge", lines 881-885):
 *   - latency:
 *       null / < 1ms      -> "—"
 *       < 1000ms          -> "XXXms"
 *       < 100000ms        -> "X.Xs"
 *       >= 100s           -> "XXXs"
 *   - cost:
 *       0                 -> "$0"
 *       < 0.001           -> "$1.23e-4"   (scientific, 2-sig-fig mantissa)
 *       >= 0.001          -> "$0.0042"    (4 decimals)
 *   - tokens:
 *       null              -> "—"
 *       < 1000            -> "XXX tok"
 *       >= 1000           -> "X.Xk tok"
 *
 * Helpers live in lib/ (not co-located with CostLatencyBadge.tsx) so the
 * AuditDetailSheet and tests can import them without tripping the
 * `react-refresh/only-export-components` lint rule.
 */

export function _fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 100_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 1000)}s`;
}

export function _fmtCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined) return "—";
  if (usd === 0) return "$0";
  if (usd < 0.001) return `$${usd.toExponential(2)}`;
  return `$${usd.toFixed(4)}`;
}

export function _fmtTokens(total: number | null | undefined): string {
  if (total === null || total === undefined) return "—";
  if (total < 1000) return `${total} tok`;
  return `${(total / 1000).toFixed(1)}k tok`;
}
