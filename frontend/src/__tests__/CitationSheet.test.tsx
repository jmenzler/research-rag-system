/**
 * Wave-0 RED test for <CitationSheet> (D-08).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/CitationSheet.tsx).
 * The Sheet slides in from the right, renders the parent chunk + paper link,
 * and closes on outside-click or Esc.
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

describe("CitationSheet", () => {
  it("opens when useUiStore.openCitation is called", async () => {
    const { CitationSheet } = await import("@/components/chat/CitationSheet");
    const { useUiStore } = await import("@/state/uiStore");

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              id: "p-1",
              text: "parent text",
              source_file: "paper.pdf",
              notebook: "trading",
              modality: "text",
              page_number: 3,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
      ),
    );

    renderWithClient(<CitationSheet />);
    act(() => {
      useUiStore.getState().openCitation("msg-1", 1);
    });
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeTruthy();
  });

  it("renders parent chunk text + paper link button", async () => {
    const { CitationSheet } = await import("@/components/chat/CitationSheet");
    const { useUiStore } = await import("@/state/uiStore");

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              id: "p-1",
              text: "parent text",
              source_file: "paper.pdf",
              notebook: "trading",
              modality: "text",
              page_number: 3,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
      ),
    );

    renderWithClient(<CitationSheet />);
    act(() => {
      useUiStore.getState().openCitation("msg-1", 1);
    });

    await screen.findByText(/parent text/i);
    // Paper-link affordance: either a button or an anchor with the paper name.
    const links = screen.queryAllByText(/paper\.pdf/i);
    expect(links.length).toBeGreaterThan(0);
  });

  it("closes on Esc", async () => {
    const { CitationSheet } = await import("@/components/chat/CitationSheet");
    const { useUiStore } = await import("@/state/uiStore");

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              id: "p-1",
              text: "parent text",
              source_file: "paper.pdf",
              notebook: "trading",
              modality: "text",
              page_number: 3,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
      ),
    );

    renderWithClient(<CitationSheet />);
    act(() => {
      useUiStore.getState().openCitation("msg-1", 1);
    });
    await screen.findByRole("dialog");

    const user = userEvent.setup();
    await user.keyboard("{Escape}");

    // Sheet either disappears from the DOM or has aria-hidden / closed state.
    // Polling-friendly assertion:
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
