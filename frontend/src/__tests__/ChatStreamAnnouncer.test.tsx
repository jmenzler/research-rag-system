/**
 * Unit tests for the ChatSurface aria-live announcer + citation-cycle hotkey.
 *
 * Behavioural contract (D-12, POLISH-06):
 *   1. An always-mounted sr-only div with aria-live="polite" aria-atomic="false"
 *      exists in ChatSurface regardless of streaming state.
 *   2. The div's text content equals stream.delta whenever it is non-empty —
 *      including after the stream reaches "done", so assistive tech still
 *      announces the final answer. It is the empty string only when delta is
 *      empty (idle).
 *   3. Dispatching hotkey:citation-next moves focus among rendered citation pills
 *      (no-op when none are rendered).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

// ---------------------------------------------------------------------------
// Mock heavy dependencies
// ---------------------------------------------------------------------------

vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }),
}));

vi.mock("@/hooks/useChatStream", () => ({
  useChatStream: vi.fn(),
}));

vi.mock("@/queries/chats", () => ({
  useChatQuery: vi.fn(),
}));

const mockUiStore = {
  registerCitations: vi.fn(),
  scopeDialogOpen: null,
  setScopeDialogOpen: vi.fn(),
  openedCitation: null,
  openCitation: vi.fn(),
  closeCitation: vi.fn(),
  openedAuditQueryId: null,
  openAudit: vi.fn(),
  closeAudit: vi.fn(),
  editingMessageId: null,
  setEditingMessageId: vi.fn(),
  firstSeenSha: null,
  healthFailureStreak: 0,
  noteHealthFailure: vi.fn(),
  noteHealthSuccess: vi.fn(),
  sidebarOpen: true,
  setSidebarOpen: vi.fn(),
  commandPaletteOpen: false,
  setCommandPaletteOpen: vi.fn(),
  helpSheetOpen: false,
  setHelpSheetOpen: vi.fn(),
};

vi.mock("@/state/uiStore", () => ({
  useUiStore: vi.fn((selector: (s: typeof mockUiStore) => unknown) =>
    selector(mockUiStore),
  ),
}));

vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return { ...actual };
});

vi.mock("@/components/chat/CitationSheet", () => ({
  CitationSheet: () => null,
}));

vi.mock("@/components/chat/AuditDetailSheet", () => ({
  AuditDetailSheet: () => null,
}));

vi.mock("@/components/chat/Conversation", () => ({
  Conversation: () => null,
}));

vi.mock("@/components/chat/StageProgressStrip", () => ({
  StageProgressStrip: () => null,
}));

vi.mock("@/components/chat/ChatScopeBar", () => ({
  ChatScopeBar: () => null,
}));

vi.mock("@/components/chat/NewChatDialog", () => ({
  NewChatDialog: () => null,
}));

vi.mock("@/components/chat/ComposeBox", () => ({
  ComposeBox: () => null,
}));

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function renderWithClient(ui: ReactNode): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Helpers to set up mock return values
// ---------------------------------------------------------------------------

async function importMocks() {
  const { useChatStream } = await import("@/hooks/useChatStream");
  const { useChatQuery } = await import("@/queries/chats");
  return { useChatStream: vi.mocked(useChatStream), useChatQuery: vi.mocked(useChatQuery) };
}

function makeChatQueryResult(overrides: Record<string, unknown> = {}) {
  return {
    data: null,
    isLoading: false,
    error: null,
    ...overrides,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}

function makeStreamResult(overrides: Record<string, unknown> = {}) {
  return {
    state: "idle",
    delta: "",
    stagesObserved: [],
    subStages: {},
    citations: [],
    done: null,
    messageUuid: null,
    error: null,
    start: vi.fn(),
    stop: vi.fn(),
    ...overrides,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ChatSurface — aria-live announcer (D-12)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders an always-mounted sr-only aria-live=polite region even when not streaming", async () => {
    const { useChatStream, useChatQuery } = await importMocks();
    useChatQuery.mockReturnValue(makeChatQueryResult());
    useChatStream.mockReturnValue(makeStreamResult({ state: "idle", delta: "" }));

    const { ChatSurface } = await import("@/components/chat/ChatSurface");
    renderWithClient(<ChatSurface chatId="test-chat-1" />);

    const announcer = document.querySelector('[aria-live="polite"][aria-atomic="false"]');
    expect(announcer).not.toBeNull();
    expect(announcer?.textContent).toBe("");
  });

  it("populates the announcer with stream.delta while streaming", async () => {
    const { useChatStream, useChatQuery } = await importMocks();
    useChatQuery.mockReturnValue(makeChatQueryResult());
    useChatStream.mockReturnValue(
      makeStreamResult({ state: "streaming", delta: "Hello world" }),
    );

    const { ChatSurface } = await import("@/components/chat/ChatSurface");
    renderWithClient(<ChatSurface chatId="test-chat-2" />);

    const announcer = document.querySelector('[aria-live="polite"][aria-atomic="false"]');
    expect(announcer).not.toBeNull();
    expect(announcer?.textContent).toBe("Hello world");
  });

  it("retains the final answer in the announcer after the stream reaches done", async () => {
    const { useChatStream, useChatQuery } = await importMocks();
    useChatQuery.mockReturnValue(makeChatQueryResult());
    useChatStream.mockReturnValue(
      makeStreamResult({ state: "done", delta: "The final synthesized answer." }),
    );

    const { ChatSurface } = await import("@/components/chat/ChatSurface");
    renderWithClient(<ChatSurface chatId="test-chat-done" />);

    const announcer = document.querySelector('[aria-live="polite"][aria-atomic="false"]');
    expect(announcer).not.toBeNull();
    expect(announcer?.textContent).toBe("The final synthesized answer.");
  });

  it("dispatching hotkey:citation-next is a no-op when no citation pills exist", async () => {
    const { useChatStream, useChatQuery } = await importMocks();
    useChatQuery.mockReturnValue(makeChatQueryResult());
    useChatStream.mockReturnValue(makeStreamResult({ state: "idle", delta: "" }));

    const { ChatSurface } = await import("@/components/chat/ChatSurface");
    renderWithClient(<ChatSurface chatId="test-chat-3" />);

    // Should not throw when no citation pills are present
    await act(async () => {
      window.dispatchEvent(new CustomEvent("hotkey:citation-next"));
    });

    // Still renders the announcer
    const announcer = document.querySelector('[aria-live="polite"][aria-atomic="false"]');
    expect(announcer).not.toBeNull();
  });
});
