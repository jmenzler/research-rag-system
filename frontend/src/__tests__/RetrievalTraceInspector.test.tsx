/**
 * Wave-0 RED test for <RetrievalTraceInspector> (D-10, D-11, D-12, CHAT-12).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/RetrievalTraceInspector.tsx).
 *
 * Behavioural contract:
 *   - Collapsed by default with `Retrieval (N chunks) ▸` label.
 *   - Click expands to N cards with dense/sparse/rerank score badges.
 *   - Click on a card opens the CitationSheet (openCitation).
 *   - "Open full audit" button dispatches openAuditDetail(queryId) (D-12).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

function makeChunks(n: number): Array<{
  marker: number;
  child_id: string;
  parent_id: string;
  paper_id: string;
  short_cite: string;
  preview: string;
  score_dense: number;
  score_sparse: number;
  score_rerank: number;
}> {
  return Array.from({ length: n }, (_, i) => ({
    marker: i + 1,
    child_id: `c-${i + 1}`,
    parent_id: `p-${i + 1}`,
    paper_id: `paper-${i + 1}.pdf`,
    short_cite: `Paper ${i + 1} (2024)`,
    preview: `Preview of chunk ${i + 1} text content...`,
    score_dense: 0.9 - 0.05 * i,
    score_sparse: 0.8 - 0.05 * i,
    score_rerank: 0.95 - 0.05 * i,
  }));
}

describe("RetrievalTraceInspector", () => {
  it("renders collapsed by default with chunk count label (D-10)", async () => {
    const { RetrievalTraceInspector } = await import(
      "@/components/chat/RetrievalTraceInspector"
    );
    renderWithClient(
      <RetrievalTraceInspector
        chunks={makeChunks(3)}
        queryId="q1"
        messageId="m1"
      />,
    );

    expect(screen.getByText(/^Sources$/)).toBeTruthy();
    // Card content is NOT rendered yet in collapsed state.
    expect(screen.queryByTestId("trace-card-0")).toBeNull();
  });

  it("expands to show N cards with sparse/dense/rerank score badges (D-11)", async () => {
    const { RetrievalTraceInspector } = await import(
      "@/components/chat/RetrievalTraceInspector"
    );
    renderWithClient(
      <RetrievalTraceInspector
        chunks={makeChunks(3)}
        queryId="q1"
        messageId="m1"
      />,
    );

    const user = userEvent.setup();
    const trigger = screen.getByRole("button", { name: /sources/i });
    await user.click(trigger);

    expect(screen.getByTestId("trace-card-0")).toBeTruthy();
    expect(screen.getByTestId("trace-card-1")).toBeTruthy();
    expect(screen.getByTestId("trace-card-2")).toBeTruthy();

    // First card numeric badges (dense=0.9, sparse=0.8, rerank=0.95).
    const card0 = screen.getByTestId("trace-card-0");
    expect(card0.textContent).toMatch(/0\.9/);
    expect(card0.textContent).toMatch(/0\.8/);
    expect(card0.textContent).toMatch(/0\.95/);
  });

  it("click on card opens CitationSheet", async () => {
    const { RetrievalTraceInspector } = await import(
      "@/components/chat/RetrievalTraceInspector"
    );
    const { useUiStore } = await import("@/state/uiStore");

    renderWithClient(
      <RetrievalTraceInspector
        chunks={makeChunks(3)}
        queryId="q1"
        messageId="m1"
      />,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /sources/i }));
    const card = screen.getByTestId("trace-card-0");
    await user.click(card);

    expect(
      (useUiStore.getState() as unknown as { openedCitation: unknown })
        .openedCitation,
    ).toEqual({ messageId: "m1", marker: 1 });
  });

  it("Open full audit button calls openAuditDetail with queryId (D-12)", async () => {
    const { RetrievalTraceInspector } = await import(
      "@/components/chat/RetrievalTraceInspector"
    );
    const { useUiStore } = await import("@/state/uiStore");

    renderWithClient(
      <RetrievalTraceInspector
        chunks={makeChunks(3)}
        queryId="q1"
        messageId="m1"
      />,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /sources/i }));
    const auditButton = screen.getByRole("button", {
      name: /open full audit/i,
    });
    await user.click(auditButton);

    expect(
      (useUiStore.getState() as unknown as { openedAuditQueryId: string })
        .openedAuditQueryId,
    ).toBe("q1");
  });
});
