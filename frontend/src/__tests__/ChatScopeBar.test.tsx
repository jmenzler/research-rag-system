/**
 * Wave-1 RED test for <ChatScopeBar> (D-01).
 *
 * Implementation lands in Plan 02-13 (frontend/src/components/chat/ChatScopeBar.tsx).
 *
 * Behavioural contract:
 *   - Shows the current chat's retriever badge (`milvus` / `paperqa` / `hipporag` / `lazygraph` / `fused`)
 *   - Shows the current chat's collection count: `"4 collections"` (default) or comma-list
 *   - Click anywhere on the bar opens <NewChatDialog> in edit-mode
 *   - Visually disabled while streaming (opacity-60 pointer-events-none)
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

describe("<ChatScopeBar> — RED tests until Plan 02-13", () => {
  it("shows the current retriever badge", async () => {
    const { ChatScopeBar } = await import(
      "@/components/chat/ChatScopeBar"
    );

    renderWithClient(
      <ChatScopeBar
        chatId="c1"
        retriever="paperqa"
        collections={["trading", "ecology"]}
        streaming={false}
      />,
    );

    expect(await screen.findByText("paperqa")).toBeTruthy();
  });

  it("shows collection count ('4 collections' for default; comma-list otherwise)", async () => {
    const { ChatScopeBar } = await import(
      "@/components/chat/ChatScopeBar"
    );

    // Case 1: all four collections selected → "4 collections"
    const { unmount } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <ChatScopeBar
          chatId="c1"
          retriever="milvus"
          collections={["trading", "ecology", "notes", "system"]}
          streaming={false}
        />
      </QueryClientProvider>,
    );
    expect(await screen.findByText(/4 collections/i)).toBeTruthy();
    unmount();

    // Case 2: subset → comma-separated list
    renderWithClient(
      <ChatScopeBar
        chatId="c2"
        retriever="milvus"
        collections={["trading", "ecology"]}
        streaming={false}
      />,
    );
    expect(await screen.findByText(/trading.*ecology/i)).toBeTruthy();
  });

  it("opens <NewChatDialog> in edit-mode on bar click", async () => {
    const { ChatScopeBar } = await import(
      "@/components/chat/ChatScopeBar"
    );
    const { useUiStore } = await import("@/state/uiStore");

    renderWithClient(
      <ChatScopeBar
        chatId="c1"
        retriever="milvus"
        collections={["trading"]}
        streaming={false}
      />,
    );

    const user = userEvent.setup();
    const bar = await screen.findByRole("button");
    await user.click(bar);

    const state = useUiStore.getState() as unknown as {
      scopeDialogOpen?: "create" | "edit" | null;
    };
    expect(state.scopeDialogOpen).toBe("edit");
  });

  it("is visually disabled during stream (opacity-60 pointer-events-none)", async () => {
    const { ChatScopeBar } = await import(
      "@/components/chat/ChatScopeBar"
    );

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <ChatScopeBar
          chatId="c1"
          retriever="milvus"
          collections={["trading"]}
          streaming={true}
        />
      </QueryClientProvider>,
    );

    const html = container.innerHTML;
    expect(html).toMatch(/opacity-60/);
    expect(html).toMatch(/pointer-events-none/);
  });
});
