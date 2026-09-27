/**
 * Wave-1 RED test for the Phase-2 extension of useUiStore.
 *
 * Implementation lands in Plan 02-12 (frontend/src/state/uiStore.ts — adds
 * `sidebarOpen` (persisted via Zustand `persist`), `scopeDialogOpen` (ephemeral),
 * `editingMessageId` (ephemeral), wrapped in `persist` with key `"rag-ui"` and
 * a `partialize` that only persists `sidebarOpen`).
 *
 * Behavioural contract:
 *   - sidebarOpen defaults to true
 *   - toggleSidebar flips the boolean
 *   - sidebarOpen persists to localStorage key 'rag-ui'
 *   - scopeDialogOpen does NOT persist
 *   - editingMessageId does NOT persist
 *   - All Phase 1 slots remain functional after extension
 */
import { describe, expect, it, vi, beforeEach } from "vitest";

beforeEach(() => {
  // Reset modules so the store re-imports with a fresh localStorage stub each time.
  vi.resetModules();
  // Provide a deterministic in-memory localStorage mock.
  const storage: Record<string, string> = {};
  const mock = {
    getItem: vi.fn((k: string) => storage[k] ?? null),
    setItem: vi.fn((k: string, v: string) => {
      storage[k] = v;
    }),
    removeItem: vi.fn((k: string) => {
      delete storage[k];
    }),
    clear: vi.fn(() => {
      for (const k of Object.keys(storage)) delete storage[k];
    }),
    key: vi.fn((_i: number) => null),
    length: 0,
  };
  vi.stubGlobal("localStorage", mock);
});

describe("useUiStore Phase-2 extension — RED tests until Plan 02-12", () => {
  it("sidebarOpen defaults to true", async () => {
    const { useUiStore } = await import("@/state/uiStore");
    const state = useUiStore.getState() as unknown as {
      sidebarOpen: boolean;
    };
    expect(state.sidebarOpen).toBe(true);
  });

  it("toggleSidebar flips the boolean", async () => {
    const { useUiStore } = await import("@/state/uiStore");
    const before = (
      useUiStore.getState() as unknown as { sidebarOpen: boolean }
    ).sidebarOpen;
    (
      useUiStore.getState() as unknown as { toggleSidebar: () => void }
    ).toggleSidebar();
    const after = (
      useUiStore.getState() as unknown as { sidebarOpen: boolean }
    ).sidebarOpen;
    expect(after).toBe(!before);
  });

  it("sidebarOpen persists to localStorage under key 'rag-ui'", async () => {
    const { useUiStore } = await import("@/state/uiStore");
    (
      useUiStore.getState() as unknown as {
        setSidebarOpen: (v: boolean) => void;
      }
    ).setSidebarOpen(false);

    const setItem = (
      localStorage as unknown as { setItem: ReturnType<typeof vi.fn> }
    ).setItem;
    const wroteToRagUi = setItem.mock.calls.some(
      (c) => String(c[0]) === "rag-ui",
    );
    expect(wroteToRagUi).toBe(true);

    // The serialized payload contains the sidebarOpen flag.
    const ragUiCall = setItem.mock.calls.find((c) => String(c[0]) === "rag-ui");
    const payload = String(ragUiCall?.[1] ?? "");
    expect(payload).toContain("sidebarOpen");
  });

  it("scopeDialogOpen does NOT persist", async () => {
    const { useUiStore } = await import("@/state/uiStore");
    (
      useUiStore.getState() as unknown as {
        setScopeDialogOpen: (v: "create" | "edit" | null) => void;
      }
    ).setScopeDialogOpen("edit");

    const setItem = (
      localStorage as unknown as { setItem: ReturnType<typeof vi.fn> }
    ).setItem;
    const ragUiCall = setItem.mock.calls.find((c) => String(c[0]) === "rag-ui");
    if (ragUiCall !== undefined) {
      const payload = String(ragUiCall[1] ?? "");
      expect(payload).not.toContain("scopeDialogOpen");
    }
  });

  it("editingMessageId does NOT persist", async () => {
    const { useUiStore } = await import("@/state/uiStore");
    (
      useUiStore.getState() as unknown as {
        setEditingMessageId: (id: string | null) => void;
      }
    ).setEditingMessageId("m-1");

    const setItem = (
      localStorage as unknown as { setItem: ReturnType<typeof vi.fn> }
    ).setItem;
    const ragUiCall = setItem.mock.calls.find((c) => String(c[0]) === "rag-ui");
    if (ragUiCall !== undefined) {
      const payload = String(ragUiCall[1] ?? "");
      expect(payload).not.toContain("editingMessageId");
    }
  });

  it("Phase 1 slots (firstSeenSha, openedCitation, openedAuditQueryId) remain functional", async () => {
    const { useUiStore } = await import("@/state/uiStore");

    const s = useUiStore.getState() as unknown as {
      firstSeenSha: string | null;
      setFirstSeenSha: (sha: string) => void;
      openedCitation: unknown;
      openCitation: (mid: string, marker: number) => void;
      openedAuditQueryId: string | null;
      openAuditDetail: (qid: string) => void;
    };

    s.setFirstSeenSha("abc123");
    expect(
      (useUiStore.getState() as unknown as { firstSeenSha: string | null })
        .firstSeenSha,
    ).toBe("abc123");

    s.openCitation("m1", 1);
    expect(
      (useUiStore.getState() as unknown as { openedCitation: unknown })
        .openedCitation,
    ).toEqual({ messageId: "m1", marker: 1 });

    s.openAuditDetail("q1");
    expect(
      (
        useUiStore.getState() as unknown as { openedAuditQueryId: string | null }
      ).openedAuditQueryId,
    ).toBe("q1");
  });
});
