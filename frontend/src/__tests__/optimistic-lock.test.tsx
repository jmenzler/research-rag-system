/**
 * Wave-0 RED test for optimistic locking (CHAT-11, D-17).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/ChatSurface.tsx +
 * useChatStream's 409 handling).
 *
 * Mirrors frontend/src/__tests__/VersionPill.test.tsx:58-77 (sonner toast mock).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

let lastClient: QueryClient | null = null;
let invalidateSpy: ReturnType<typeof vi.fn>;

function renderWithClient(ui: ReactNode): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  invalidateSpy = vi.fn(client.invalidateQueries.bind(client));
  client.invalidateQueries = invalidateSpy as unknown as typeof client.invalidateQueries;
  lastClient = client;
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
  lastClient = null;
});

describe("Optimistic locking (CHAT-11)", () => {
  it("shows Sonner toast on 409 response", async () => {
    const { ChatSurface } = await import("@/components/chat/ChatSurface");

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              detail: "version mismatch",
              current_version: 5,
            }),
            {
              status: 409,
              headers: { "content-type": "application/json" },
            },
          ),
        ),
      ),
    );

    renderWithClient(<ChatSurface chatId="c1" />);

    const user = userEvent.setup();
    const textbox = await screen.findByRole("textbox");
    await user.click(textbox);
    await user.keyboard("hi{Enter}");

    await waitFor(() => expect(toastMock).toHaveBeenCalled());
    const calls = toastMock.mock.calls;
    const flattened = calls.map((c) => String(c[0])).join(" ");
    expect(flattened).toMatch(/updated in another tab/i);
  });

  it("invalidates useChatQuery on 409", async () => {
    const { ChatSurface } = await import("@/components/chat/ChatSurface");

    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(
            JSON.stringify({
              detail: "version mismatch",
              current_version: 5,
            }),
            {
              status: 409,
              headers: { "content-type": "application/json" },
            },
          ),
        ),
      ),
    );

    renderWithClient(<ChatSurface chatId="c1" />);
    const user = userEvent.setup();
    const textbox = await screen.findByRole("textbox");
    await user.click(textbox);
    await user.keyboard("hi{Enter}");

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["chats", "c1"] }),
      );
    });
    // Touch lastClient so the lint isn't noisy on the unused var.
    expect(lastClient).not.toBeNull();
  });
});
