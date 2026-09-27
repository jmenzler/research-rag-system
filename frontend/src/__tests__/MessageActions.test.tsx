/**
 * Wave-1 RED test for <MessageActions> per-turn ⋯ menu (D-19).
 *
 * Implementation lands in Plan 02-14 (frontend/src/components/chat/MessageActions.tsx).
 *
 * Behavioural contract:
 *   - DropdownMenu items appear in EXACT order:
 *     1. Regenerate, 2. Copy as text, 3. Copy as markdown, (separator), 4. Open audit detail.
 *   - Regenerate is disabled while a stream is active for that chat.
 *   - Trigger button is hidden on the streaming turn.
 *   - Open audit detail invokes useUiStore.openAuditDetail(queryId).
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

describe("<MessageActions> — RED tests until Plan 02-14", () => {
  it("DropdownMenu items appear in EXACT order: Regenerate, Copy as text, Copy as markdown, Separator, Open audit detail", async () => {
    const { MessageActions } = await import(
      "@/components/chat/MessageActions"
    );

    renderWithClient(
      <MessageActions
        chatId="c1"
        messageId="m1"
        queryId="q1"
        streaming={false}
      />,
    );

    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", {
      name: /more|actions|menu/i,
    });
    await user.click(trigger);

    // Collect menu items in DOM order. shadcn DropdownMenuItem uses role="menuitem".
    const items = await screen.findAllByRole("menuitem");
    const labels = items.map((n) => (n.textContent ?? "").trim());
    expect(labels).toEqual([
      "Regenerate",
      "Copy as text",
      "Copy as markdown",
      "Open audit detail",
    ]);

    // Confirm a separator sits between "Copy as markdown" and "Open audit detail".
    const lastCopy = items[2];
    const opener = items[3];
    expect(lastCopy).toBeDefined();
    expect(opener).toBeDefined();
    let sib: Element | null = lastCopy?.nextElementSibling ?? null;
    let sawSeparator = false;
    while (sib && sib !== opener) {
      if (
        sib.getAttribute("role") === "separator" ||
        sib.tagName.toLowerCase() === "hr"
      ) {
        sawSeparator = true;
        break;
      }
      sib = sib.nextElementSibling;
    }
    expect(sawSeparator).toBe(true);
  });

  it("Regenerate is disabled while a stream is active for that chat", async () => {
    const { MessageActions } = await import(
      "@/components/chat/MessageActions"
    );

    renderWithClient(
      <MessageActions
        chatId="c1"
        messageId="m1"
        queryId="q1"
        streaming={true}
      />,
    );

    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", {
      name: /more|actions|menu/i,
    });
    await user.click(trigger);

    const regen = await screen.findByRole("menuitem", { name: /regenerate/i });
    expect(
      regen.getAttribute("aria-disabled") === "true" ||
        regen.hasAttribute("disabled") ||
        regen.getAttribute("data-disabled") !== null,
    ).toBe(true);
  });

  it("trigger button is hidden on the streaming turn", async () => {
    const { MessageActions } = await import(
      "@/components/chat/MessageActions"
    );

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <MessageActions
          chatId="c1"
          messageId="m-streaming"
          queryId={null}
          streamingThisTurn={true}
        />
      </QueryClientProvider>,
    );

    expect(container.querySelector("button")).toBeNull();
  });

  it("Open audit detail invokes useUiStore.openAuditDetail(queryId)", async () => {
    const { MessageActions } = await import(
      "@/components/chat/MessageActions"
    );
    const { useUiStore } = await import("@/state/uiStore");

    renderWithClient(
      <MessageActions
        chatId="c1"
        messageId="m1"
        queryId="q-target"
        streaming={false}
      />,
    );

    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", {
      name: /more|actions|menu/i,
    });
    await user.click(trigger);
    const openAudit = await screen.findByRole("menuitem", {
      name: /open audit detail/i,
    });
    await user.click(openAudit);

    const state = useUiStore.getState() as unknown as {
      openedAuditQueryId: string | null;
    };
    expect(state.openedAuditQueryId).toBe("q-target");
  });
});
