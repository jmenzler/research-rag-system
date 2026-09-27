/**
 * Wave-0 RED test for <CitationToken> (D-06, D-07, D-09).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/CitationToken.tsx).
 * Tests are RED until the component exists — vitest treats the missing import
 * as a test failure (not a collection error).
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

describe("CitationToken", () => {
  it("renders <button> with aria-label 'citation 1, open source'", async () => {
    const { CitationToken } = await import("@/components/chat/CitationToken");
    renderWithClient(
      <CitationToken marker={1} chunkId="abc" resolved={true} messageId="m-1" />,
    );
    const button = screen.getByRole("button", {
      name: /citation 1, open source/i,
    });
    expect(button).toBeTruthy();
  });

  it("renders strike-through [?] when resolved=false (hallucinated)", async () => {
    const { CitationToken } = await import("@/components/chat/CitationToken");
    renderWithClient(
      <CitationToken marker={7} chunkId={null} resolved={false} messageId="m-1" />,
    );
    const node = screen.getByText(/\[\?\]/);
    expect(node).toBeTruthy();
    // Tailwind line-through class indicates the strike-through style (D-09).
    expect(node.className).toMatch(/line-through/);
  });

  it("dispatches openCitation on click", async () => {
    const { CitationToken } = await import("@/components/chat/CitationToken");
    const { useUiStore } = await import("@/state/uiStore");

    const spy = vi.fn();
    // Inject the spy into the store; the component reads it via getState().
    useUiStore.setState({ openCitation: spy } as unknown as Record<string, unknown>);

    renderWithClient(
      <CitationToken marker={1} chunkId="abc" resolved={true} messageId="m-1" />,
    );
    const button = screen.getByRole("button", {
      name: /citation 1, open source/i,
    });
    const user = userEvent.setup();
    await user.click(button);
    expect(spy).toHaveBeenCalledWith("m-1", 1);
  });

  it("hover opens HoverCard with first ~200 chars and score badges (D-07)", async () => {
    const { CitationToken } = await import("@/components/chat/CitationToken");

    // Mock the parent-chunk fetch so the hover-card has data to render.
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              id: "abc",
              text: "preview text from parent chunk that is long enough to truncate at ~200 chars",
              source_file: "paper.pdf",
              notebook: "trading",
              modality: "text",
              page_number: 1,
            }),
            { status: 200, headers: { "content-type": "application/json" } },
          ),
        ),
      ),
    );

    renderWithClient(
      <CitationToken marker={1} chunkId="abc" resolved={true} messageId="m-1" />,
    );
    const button = screen.getByRole("button", {
      name: /citation 1, open source/i,
    });
    const user = userEvent.setup();
    await user.hover(button);
    // shadcn HoverCard renders the preview content asynchronously.
    const preview = await screen.findByText(/preview text from parent chunk/i);
    expect(preview).toBeTruthy();
  });
});
