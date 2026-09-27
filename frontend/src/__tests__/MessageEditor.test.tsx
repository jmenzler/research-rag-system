/**
 * Wave-1 RED test for <MessageEditor> + <HoverPencil> (D-17, D-18, Pitfall 4).
 *
 * Implementation lands in Plan 02-14:
 *   - frontend/src/components/chat/HoverPencil.tsx
 *   - frontend/src/components/chat/MessageEditor.tsx
 *
 * Behavioural contract:
 *   - Pencil renders ONLY on the LAST user turn in the merged message list.
 *   - Older user turns get no pencil.
 *   - Pencil is hidden during streaming.
 *   - Clicking pencil sets useUiStore.editingMessageId.
 *   - Save & regenerate POSTs to /api/chats/{id}/edit-last with a FRESH
 *     `message_uuid` (NOT the original) — Pitfall 4.
 *   - Cancel discards textarea content silently — no confirm dialog (D-18).
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

describe("<HoverPencil> + <MessageEditor> — RED tests until Plan 02-14", () => {
  it("pencil renders only on the LAST user turn", async () => {
    const { HoverPencil } = await import("@/components/chat/HoverPencil");

    renderWithClient(
      <HoverPencil messageId="m-last" isLastUserTurn={true} streaming={false} />,
    );
    expect(await screen.findByRole("button", { name: /edit message/i })).toBeTruthy();
  });

  it("older user turns get no pencil", async () => {
    const { HoverPencil } = await import("@/components/chat/HoverPencil");

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <HoverPencil messageId="m-old" isLastUserTurn={false} streaming={false} />
      </QueryClientProvider>,
    );

    expect(container.querySelector("button")).toBeNull();
  });

  it("pencil is hidden during streaming", async () => {
    const { HoverPencil } = await import("@/components/chat/HoverPencil");

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <HoverPencil messageId="m-last" isLastUserTurn={true} streaming={true} />
      </QueryClientProvider>,
    );

    const button = container.querySelector("button");
    if (button === null) {
      // Truly unmounted while streaming — acceptable.
      expect(button).toBeNull();
      return;
    }
    // Or rendered but pointer-events-none + opacity-0.
    const html = button.outerHTML;
    expect(html).toMatch(/pointer-events-none/);
    expect(html).toMatch(/opacity-0/);
  });

  it("click pencil sets useUiStore.editingMessageId", async () => {
    const { HoverPencil } = await import("@/components/chat/HoverPencil");
    const { useUiStore } = await import("@/state/uiStore");

    renderWithClient(
      <HoverPencil
        messageId="m-target"
        isLastUserTurn={true}
        streaming={false}
      />,
    );

    const user = userEvent.setup();
    const pencil = await screen.findByRole("button", { name: /edit message/i });
    await user.click(pencil);

    const state = useUiStore.getState() as unknown as {
      editingMessageId: string | null;
    };
    expect(state.editingMessageId).toBe("m-target");
  });

  it("Save & regenerate POSTs to /api/chats/{id}/edit-last with a FRESH message_uuid (Pitfall 4)", async () => {
    const { MessageEditor } = await import(
      "@/components/chat/MessageEditor"
    );

    const fetchSpy = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchSpy);

    renderWithClient(
      <MessageEditor
        chatId="c1"
        messageId="m1"
        originalMessageUuid="ORIGINAL-UUID-AAA"
        initialContent="hello"
        expectedVersion={5}
      />,
    );

    const user = userEvent.setup();
    const textarea = await screen.findByRole("textbox");
    await user.clear(textarea);
    await user.type(textarea, "edited content");
    const save = await screen.findByRole("button", { name: /save & regenerate/i });
    await user.click(save);

    const fetchCalls = fetchSpy.mock.calls as unknown as Array<
      [string, RequestInit | undefined]
    >;
    const call = fetchCalls.find(
      (c) => String(c[0]).includes("/edit-last") && c[1]?.method === "POST",
    );
    expect(call).toBeDefined();
    const body = JSON.parse(String(call?.[1]?.body ?? "{}"));
    expect(body.content).toBe("edited content");
    // Pitfall 4 — server creates a NEW message row; the client MUST send a fresh
    // message_uuid so the audit trail / parent_id tracking gets a unique key.
    expect(body.message_uuid).toBeDefined();
    expect(body.message_uuid).not.toBe("ORIGINAL-UUID-AAA");
  });

  it("Cancel discards textarea content silently — no confirm dialog (D-18)", async () => {
    const { MessageEditor } = await import(
      "@/components/chat/MessageEditor"
    );
    const { useUiStore } = await import("@/state/uiStore");

    // Mark this row as being edited so MessageEditor mounts in edit-mode.
    act(() => {
      (
        useUiStore.getState() as unknown as {
          setEditingMessageId?: (id: string | null) => void;
        }
      ).setEditingMessageId?.("m1");
    });

    renderWithClient(
      <MessageEditor
        chatId="c1"
        messageId="m1"
        originalMessageUuid="ORIGINAL-UUID-AAA"
        initialContent="hello"
        expectedVersion={5}
      />,
    );

    const user = userEvent.setup();
    const textarea = await screen.findByRole("textbox");
    await user.clear(textarea);
    await user.type(textarea, "dirty draft");
    const cancel = await screen.findByRole("button", { name: /^cancel$/i });
    await user.click(cancel);

    // No confirm dialog (alertdialog role) should appear.
    expect(screen.queryByRole("alertdialog")).toBeNull();

    // editingMessageId is cleared.
    const state = useUiStore.getState() as unknown as {
      editingMessageId: string | null;
    };
    expect(state.editingMessageId).toBeNull();
  });
});
