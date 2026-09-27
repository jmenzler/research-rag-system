/**
 * D-13 / CHAT-13: per-turn cost + latency pill.
 *
 * Reads `DonePayload` for the in-flight turn (passed via prop) OR falls back
 * to `useQueryAuditQuery(queryId)` for persisted turns (after page reload).
 *
 * Format spec (01-RESEARCH.md §"Cost + Latency Badge", lines 881-885):
 *   - latency:
 *       < 1ms      -> "—"
 *       < 1000ms   -> "XXXms"
 *       < 100000ms -> "X.Xs"
 *       >= 100s    -> "XXXs"
 *   - cost:
 *       0          -> "$0"
 *       < 0.001    -> "$1.23e-4"   (scientific notation; 2-sig-fig mantissa)
 *       >= 0.001   -> "$0.0042"    (4 decimals)
 *   - tokens:
 *       null       -> "—"
 *       < 1000     -> "XXX tok"
 *       >= 1000    -> "X.Xk tok"
 *
 * Display: `$0.0042 · 4.2s · 1.8k tok` with middle-dot separators.
 *
 * meta.totals shape (mirrors the on-disk audit-trail `meta.json` totals):
 *   { cost_usd, total_latency_ms | latency_ms, tokens: { total } | usage: { total } | total_tokens }
 *
 * Live DonePayload shape (SSE `done` event, src/lib/sse-client.ts):
 *   { cost_usd, latency_ms, usage: { total } }
 *
 * T-01-W4-07 (accepted): for persisted turns, each badge fetches
 * /api/queries/{qid} individually (N-call latency on chat load with N turns).
 * Phase 3+ may add a bulk endpoint; until then this is accepted.
 */
import { type ReactElement } from "react";

import { Badge } from "@/components/ui/badge";
import { useQueryAuditQuery } from "@/queries/queries";
import type { DonePayload } from "@/lib/sse-client";
import { _fmtCost, _fmtMs, _fmtTokens } from "@/lib/fmt-cost";

// Re-export so consumers and tests can import via "@/components/chat/CostLatencyBadge"
// (matches the 01-07-PLAN.md <interfaces> contract). The formatters themselves
// live in @/lib/fmt-cost so that the AuditDetailSheet can import them without
// pulling the React component along; the re-export below is purely a discoverability
// shim and intentionally trips the react-refresh/only-export-components rule, which
// we silence on this one line — HMR fast-refresh of the badge is unaffected because
// these symbols are re-exports of pure functions from another module.
// eslint-disable-next-line react-refresh/only-export-components
export { _fmtCost, _fmtMs, _fmtTokens };

export interface CostLatencyBadgeProps {
  queryId: string | null;
  liveDone?: DonePayload | null;
}

/**
 * Read a number from a totals bag at any of the given keys; first hit wins.
 * Keeps the lookup tolerant of the slight shape drift between
 * `meta.totals.latency_ms` vs `meta.totals.total_latency_ms` (and similar
 * for tokens).
 */
function pickNumber(
  totals: Record<string, unknown>,
  keys: ReadonlyArray<string>,
): number | null {
  for (const k of keys) {
    const v = totals[k];
    if (typeof v === "number") return v;
  }
  return null;
}

/**
 * Pull `tokens.total` / `usage.total` / `total_tokens` out of a totals bag.
 */
function pickTokens(totals: Record<string, unknown>): number | null {
  const total = pickNumber(totals, ["total_tokens"]);
  if (total !== null) return total;
  const tokens = totals.tokens;
  if (tokens && typeof tokens === "object") {
    const t = (tokens as { total?: unknown }).total;
    if (typeof t === "number") return t;
  }
  const usage = totals.usage;
  if (usage && typeof usage === "object") {
    const t = (usage as { total?: unknown }).total;
    if (typeof t === "number") return t;
  }
  return null;
}

export function CostLatencyBadge(props: CostLatencyBadgeProps): ReactElement {
  // Prefer the live `done` payload when available; no fetch needed in that case.
  // When liveDone is present, pass null to disable the audit fetch (Plan 05's
  // useQueryAuditQuery short-circuits on a null queryId).
  const audit = useQueryAuditQuery(props.liveDone ? null : props.queryId);

  let costUsd: number | null = null;
  let latencyMs: number | null = null;
  let totalTokens: number | null = null;

  if (props.liveDone) {
    costUsd = props.liveDone.cost_usd ?? null;
    latencyMs = props.liveDone.latency_ms ?? null;
    totalTokens = props.liveDone.usage?.total ?? null;
  } else if (audit.data) {
    const totals = (audit.data.meta?.totals ?? {}) as Record<string, unknown>;
    costUsd = pickNumber(totals, ["cost_usd"]);
    latencyMs = pickNumber(totals, ["total_latency_ms", "latency_ms"]);
    totalTokens = pickTokens(totals);
  }

  return (
    <Badge
      variant="secondary"
      data-testid="cost-latency-badge"
      className="gap-1.5 font-mono text-xs"
    >
      <span data-testid="cost-latency-cost">{_fmtCost(costUsd)}</span>
      <span aria-hidden>·</span>
      <span data-testid="cost-latency-latency">{_fmtMs(latencyMs)}</span>
      <span aria-hidden>·</span>
      <span data-testid="cost-latency-tokens">{_fmtTokens(totalTokens)}</span>
    </Badge>
  );
}
