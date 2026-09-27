/**
 * Wave-1 RED test for <NewChatDialog> (D-03, D-04, D-05).
 *
 * Implementation lands in Plan 02-13 (frontend/src/components/chat/NewChatDialog.tsx).
 *
 * Behavioural contract:
 *   - Renders five retriever options with EXACT perf-cost label strings (D-04).
 *   - Fused option's Info popover body matches D-05 exactly.
 *   - Collections default to all four checked.
 *   - Retriever defaults to `milvus`.
 *   - A retriever flagged as missing-on-server renders disabled with the EXACT
 *     HoverCard body "This retriever isn't installed on the server. Pick another."
 *   - Submit POSTs to /api/chats with `{title?, retriever, collections}` shape.
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

describe("<NewChatDialog> — RED tests until Plan 02-13", () => {
  it("renders five retriever options with EXACT perf-cost label strings", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    renderWithClient(<NewChatDialog open mode="create" onOpenChange={() => {}} />);

    // Exact strings from UI-SPEC § Copywriting Contract.
    expect(await screen.findByText("milvus · hybrid retrieval")).toBeTruthy();
    expect(screen.getByText("paperqa · optional adapter")).toBeTruthy();
    expect(screen.getByText("hipporag · optional adapter")).toBeTruthy();
    expect(screen.getByText("lazygraph · optional adapter")).toBeTruthy();
    expect(screen.getByText("fused · merged retrieval")).toBeTruthy();
  });

  it("fused option shows Info popover with exact D-05 body", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    renderWithClient(<NewChatDialog open mode="create" onOpenChange={() => {}} />);

    const user = userEvent.setup();
    // Trigger is anchored next to the fused label; aria-label or accessible name
    // mentions "fused".
    const trigger = await screen.findByRole("button", { name: /fused.*info|info.*fused/i });
    await user.click(trigger);

    expect(
      await screen.findByText(
        "Combines available retriever results and reranks the merged candidates.",
      ),
    ).toBeTruthy();
  });

  it("collections default to all four checked", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    renderWithClient(<NewChatDialog open mode="create" onOpenChange={() => {}} />);

    for (const col of ["trading", "ecology", "notes", "system"]) {
      const cb = await screen.findByRole("checkbox", { name: new RegExp(col, "i") });
      expect((cb as HTMLInputElement).checked || cb.getAttribute("aria-checked") === "true").toBe(
        true,
      );
    }
  });

  it("retriever defaults to milvus", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    renderWithClient(<NewChatDialog open mode="create" onOpenChange={() => {}} />);

    const milvus = await screen.findByRole("radio", { name: /milvus/i });
    expect(milvus.getAttribute("aria-checked") === "true" || (milvus as HTMLInputElement).checked).toBe(
      true,
    );
  });

  it("renders missing-library retriever as disabled with EXACT D-05 HoverCard body", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    renderWithClient(
      <NewChatDialog
        open
        mode="create"
        onOpenChange={() => {}}
        availability={{
          milvus: true,
          paperqa: false,
          hipporag: true,
          lazygraph: true,
          fused: true,
        }}
      />,
    );

    const user = userEvent.setup();
    const paperqa = await screen.findByText("paperqa · optional adapter");
    const row = paperqa.closest("[role=radio], button, label, div");
    expect(row?.outerHTML).toMatch(
      /disabled|aria-disabled=["']?true|opacity-60|cursor-not-allowed/i,
    );

    // Hover over the disabled option to surface the HoverCard.
    await user.hover(paperqa);
    expect(
      await screen.findByText(
        "This retriever isn't installed on the server. Pick another.",
      ),
    ).toBeTruthy();
  });

  it("submit POSTs to /api/chats with {title?, retriever, collections}", async () => {
    const { NewChatDialog } = await import(
      "@/components/chat/NewChatDialog"
    );

    const fetchSpy = vi.fn(() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            id: "new",
            title: null,
            retriever: "milvus",
            collections: '["trading","ecology","notes","system"]',
            version: 1,
            created_at: "2026-05-19T00:00:00Z",
            updated_at: "2026-05-19T00:00:00Z",
            messages: [],
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      ),
    );
    vi.stubGlobal("fetch", fetchSpy);

    renderWithClient(<NewChatDialog open mode="create" onOpenChange={() => {}} />);

    const user = userEvent.setup();
    const submit = await screen.findByRole("button", { name: /create chat/i });
    await user.click(submit);

    // Inspect the body of the first POST to /api/chats.
    const fetchCalls = fetchSpy.mock.calls as unknown as Array<
      [string, RequestInit | undefined]
    >;
    const call = fetchCalls.find(
      (c) => String(c[0]).includes("/api/chats") && c[1]?.method === "POST",
    );
    expect(call).toBeDefined();
    const body = JSON.parse(String(call?.[1]?.body ?? "{}"));
    expect(body).toHaveProperty("retriever");
    expect(body).toHaveProperty("collections");
    expect(Array.isArray(body.collections)).toBe(true);
  });
});
