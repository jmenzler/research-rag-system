/**
 * Wave-0 RED test for <CostLatencyBadge> (CHAT-13, D-13).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/CostLatencyBadge.tsx).
 *
 * Formatter conventions (01-RESEARCH.md §Cost + Latency Badge lines 881-885):
 *   - cost 0           -> "$0"
 *   - cost < 0.001     -> "$1.23e-4" (scientific)
 *   - cost >= 0.001    -> "$0.0042"  (4 sig figs)
 *   - latency < 1ms    -> "—"
 *   - latency < 1s     -> "XXXms"
 *   - latency < 100s   -> "X.Xs"
 *   - latency >= 100s  -> "XXXs"
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

function renderWithClient(ui: ReactNode): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function stubMeta(totals: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({ meta: { totals } }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    ),
  );
}

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

describe("CostLatencyBadge", () => {
  it("renders cost + latency from meta.totals shape", async () => {
    const { CostLatencyBadge } = await import(
      "@/components/chat/CostLatencyBadge"
    );
    stubMeta({
      cost_usd: 0.0042,
      total_latency_ms: 4200,
      tokens: { total: 1800 },
    });
    renderWithClient(<CostLatencyBadge queryId="abc" />);

    await screen.findByText(/\$0\.0042/);
    expect(screen.getByText(/4\.2s/)).toBeTruthy();
    expect(screen.getByText(/1\.8k tok/i)).toBeTruthy();
  });

  it("uses fmt_cost scientific notation for tiny costs", async () => {
    const { CostLatencyBadge } = await import(
      "@/components/chat/CostLatencyBadge"
    );
    stubMeta({
      cost_usd: 0.0001,
      total_latency_ms: 1000,
      tokens: { total: 100 },
    });
    renderWithClient(<CostLatencyBadge queryId="tiny" />);

    // Scientific notation: $1e-4, $1.0e-4, $1.00e-4 — accept any.
    await screen.findByText(/\$1\.?0*e-?0*4/i);
  });

  it("uses fmt_ms convention for sub-second latency", async () => {
    const { CostLatencyBadge } = await import(
      "@/components/chat/CostLatencyBadge"
    );
    stubMeta({
      cost_usd: 0.01,
      total_latency_ms: 800,
      tokens: { total: 100 },
    });
    renderWithClient(<CostLatencyBadge queryId="sub-sec" />);

    await screen.findByText(/800ms/);
  });
});
