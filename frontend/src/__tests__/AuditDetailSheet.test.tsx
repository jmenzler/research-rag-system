/**
 * Wave-0 RED test for <AuditDetailSheet> (D-13).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/AuditDetailSheet.tsx).
 *
 * Behavioural contract:
 *   - Opens on useUiStore.openAuditDetail(queryId).
 *   - Fetches /api/queries/{qid} and renders meta.totals + report.md.
 *   - Clicking a stage entry fetches /api/queries/{qid}/stages/{name}
 *     and pretty-prints it inside the same Sheet (no nested Sheet).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act } from "react";
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

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

function stubFetchByUrl(table: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      const key = Object.keys(table).find((k) => url.includes(k));
      if (key === undefined) {
        return Promise.resolve(new Response("not stubbed", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify(table[key]), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }),
  );
}

describe("AuditDetailSheet", () => {
  it("opens on openAuditDetail(queryId)", async () => {
    const { AuditDetailSheet } = await import(
      "@/components/chat/AuditDetailSheet"
    );
    const { useUiStore } = await import("@/state/uiStore");

    stubFetchByUrl({
      "/api/queries/qid-1": {
        meta: { totals: { cost_usd: 0.01, total_latency_ms: 4000 } },
        report_md: "# report heading\nbody",
      },
    });

    renderWithClient(<AuditDetailSheet />);
    act(() => {
      (
        useUiStore.getState() as unknown as {
          openAuditDetail: (qid: string) => void;
        }
      ).openAuditDetail("qid-1");
    });

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeTruthy();
  });

  it("fetches /api/queries/{qid} and renders report.md", async () => {
    const { AuditDetailSheet } = await import(
      "@/components/chat/AuditDetailSheet"
    );
    const { useUiStore } = await import("@/state/uiStore");

    stubFetchByUrl({
      "/api/queries/qid-2": {
        meta: { totals: { cost_usd: 0.01, total_latency_ms: 4000 } },
        report_md: "# report heading",
      },
    });

    renderWithClient(<AuditDetailSheet />);
    act(() => {
      (
        useUiStore.getState() as unknown as {
          openAuditDetail: (qid: string) => void;
        }
      ).openAuditDetail("qid-2");
    });
    await screen.findByText(/report heading/i);
  });

  it("drills into stage payload on click", async () => {
    const { AuditDetailSheet } = await import(
      "@/components/chat/AuditDetailSheet"
    );
    const { useUiStore } = await import("@/state/uiStore");

    stubFetchByUrl({
      "/api/queries/qid-3/stages/decompose": { foo: "bar" },
      "/api/queries/qid-3": {
        meta: { totals: { cost_usd: 0.01, total_latency_ms: 4000 } },
        report_md: "# report",
        stages: [{ name: "decompose", prefix: "01" }],
      },
    });

    renderWithClient(<AuditDetailSheet />);
    act(() => {
      (
        useUiStore.getState() as unknown as {
          openAuditDetail: (qid: string) => void;
        }
      ).openAuditDetail("qid-3");
    });

    const user = userEvent.setup();
    const stageEntry = await screen.findByText(/01_decompose/);
    await user.click(stageEntry);
    await screen.findByText(/"foo"/);
    expect(screen.getByText(/"bar"/)).toBeTruthy();
  });

  it("renders meta.totals numbers (cost, latency, tokens)", async () => {
    const { AuditDetailSheet } = await import(
      "@/components/chat/AuditDetailSheet"
    );
    const { useUiStore } = await import("@/state/uiStore");

    stubFetchByUrl({
      "/api/queries/qid-4": {
        meta: {
          totals: {
            cost_usd: 0.0042,
            total_latency_ms: 4200,
            tokens: { total: 1800 },
          },
        },
        report_md: "# report",
      },
    });

    renderWithClient(<AuditDetailSheet />);
    act(() => {
      (
        useUiStore.getState() as unknown as {
          openAuditDetail: (qid: string) => void;
        }
      ).openAuditDetail("qid-4");
    });

    // Same formatter as CostLatencyBadge: $0.0042, 4.2s, 1.8k tok.
    await screen.findByText(/\$0\.0042/);
    expect(screen.getByText(/4\.2s/)).toBeTruthy();
    expect(screen.getByText(/1\.8k tok/i)).toBeTruthy();
  });
});
