import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { StateStorage } from "zustand/middleware";

import type { CitationPayload } from "@/lib/sse-client";

export interface OpenedCitation {
  messageId: string;
  marker: number;
  /**
   * Optional parent_id, only present when the caller of openCitation
   * knew the parent_id at dispatch time (e.g. RetrievalTraceInspector's
   * per-card click). Omitted otherwise (CitationToken from Plan 06 still
   * dispatches with 2 args — backwards compat verified by
   * frontend/src/__tests__/CitationToken.test.tsx + RetrievalTraceInspector.test.tsx
   * which use `.toEqual({messageId, marker})` strict deep-equality).
   */
  parentId?: string;
}

export interface CitationRegistryEntry {
  parentId: string | null;
  resolved: boolean;
}

export type ScopeDialogMode = "create" | "edit" | null;

interface UiState {
  // First /api/version SHA seen this session. Used to detect deploys (D-15).
  firstSeenSha: string | null;
  setFirstSeenSha: (sha: string) => void;

  // Phase 0 wires the counter; Phase 6 POLISH-07 surfaces a banner at 3.
  healthFailureStreak: number;
  noteHealthFailure: () => void;
  noteHealthSuccess: () => void;

  // Phase 1 Plan 05 — citation Sheet (D-06 / D-08).
  // Plan 07 extends openCitation with an OPTIONAL third arg `parentId` so the
  // RetrievalTraceInspector card click can hand the parent_id straight to the
  // Sheet without forcing a registry lookup. CitationToken (Plan 06) still
  // calls the 2-arg form — the spread below keeps `openedCitation` strictly
  // deep-equal to `{messageId, marker}` in that path (CitationToken.test.tsx
  // line 72 + RetrievalTraceInspector.test.tsx line 128 both assert toEqual).
  openedCitation: OpenedCitation | null;
  openCitation: (messageId: string, marker: number, parentId?: string) => void;
  closeCitation: () => void;

  // Plan 07 — citation registry. The chat surface pre-populates
  // `(messageId, marker) -> {parentId, resolved}` from the per-turn citation
  // payload (live SSE stream + persisted message rows). CitationSheet falls
  // back to this when openCitation was dispatched without an explicit
  // parentId (e.g. from CitationToken). Synthetic key:
  //   `${messageId}::${marker}` -> CitationRegistryEntry
  citationRegistry: Record<string, CitationRegistryEntry>;
  registerCitations: (messageId: string, cites: CitationPayload[]) => void;

  // Phase 1 Plan 05 — Audit Detail Sheet (D-13).
  openedAuditQueryId: string | null;
  openAuditDetail: (queryId: string) => void;
  closeAuditDetail: () => void;

  // Phase 1 Plan 05 — in-flight stream control (D-04 morph button).
  streamAbortController: AbortController | null;
  setStreamAbortController: (c: AbortController | null) => void;

  // ----------------------------------------------------------------- Phase 2.
  // Plan 02-12 — sidebar open state (D-10). Persisted to localStorage['rag-ui']
  // via Zustand persist middleware + partialize (only sidebarOpen is written).
  // Default = true on desktop, false on viewport <= 768px (matchMedia read
  // once at module load; guarded for jsdom / SSR where matchMedia is missing).
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Plan 02-12 — ephemeral scope-dialog mode (D-19). Plan 02-13/14 sets to
  // "create" on new-chat click and "edit" on ChatScopeBar click. NOT persisted.
  scopeDialogOpen: ScopeDialogMode;
  setScopeDialogOpen: (mode: ScopeDialogMode) => void;

  // Plan 02-12 — ephemeral inline-edit target (HoverPencil flow, D-18).
  // The user-turn <Message> swaps content for <MessageEditor> when this
  // matches the message id. NOT persisted.
  editingMessageId: string | null;
  setEditingMessageId: (id: string | null) => void;

  // Phase 6 Plan 02 — ephemeral command palette + help sheet flags (D-08/D-09).
  // NOT added to partialize (must not persist across reloads — T-06-08).
  commandOpen: boolean;
  setCommandOpen: (open: boolean) => void;
  helpOpen: boolean;
  setHelpOpen: (open: boolean) => void;
}

function defaultSidebarOpen(): boolean {
  // jsdom + Node-side imports lack window.matchMedia. Guard against both
  // missing globals AND a missing matchMedia method (older jsdom versions).
  if (typeof window === "undefined") return true;
  const mql = window.matchMedia?.bind(window);
  if (typeof mql !== "function") return true;
  try {
    return !mql("(max-width: 768px)").matches;
  } catch {
    return true;
  }
}

/**
 * Storage adapter that prefers real localStorage but gracefully no-ops in
 * environments where it is missing or stubbed without setItem (jsdom in
 * some vitest configurations exposes `localStorage` as an object whose
 * `setItem` is undefined). Without this guard the persist middleware
 * crashes Phase 1 tests that touch useUiStore.setState (CitationToken,
 * CitationSheet, RetrievalTraceInspector, etc.) because they don't stub
 * localStorage but DO trigger setState which now flows through persist.
 */
function safeStorage(): StateStorage {
  const noop: StateStorage = {
    getItem: () => null,
    setItem: () => undefined,
    removeItem: () => undefined,
  };
  try {
    if (typeof localStorage === "undefined") return noop;
    if (typeof localStorage.setItem !== "function") return noop;
    return localStorage as StateStorage;
  } catch {
    return noop;
  }
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      // Phase 0 (verbatim).
      firstSeenSha: null,
      setFirstSeenSha: (sha) =>
        set((s) => (s.firstSeenSha === null ? { firstSeenSha: sha } : s)),

      healthFailureStreak: 0,
      noteHealthFailure: () =>
        set((s) => ({ healthFailureStreak: s.healthFailureStreak + 1 })),
      noteHealthSuccess: () => set({ healthFailureStreak: 0 }),

      // Phase 1.
      openedCitation: null,
      openCitation: (messageId, marker, parentId) =>
        set({
          openedCitation:
            parentId !== undefined
              ? { messageId, marker, parentId }
              : { messageId, marker },
        }),
      closeCitation: () => set({ openedCitation: null }),

      citationRegistry: {},
      registerCitations: (messageId, cites) =>
        set((s) => {
          const next = { ...s.citationRegistry };
          for (const c of cites) {
            next[`${messageId}::${c.marker}`] = {
              parentId: c.parent_id,
              resolved: c.resolved,
            };
          }
          return { citationRegistry: next };
        }),

      openedAuditQueryId: null,
      openAuditDetail: (queryId) => set({ openedAuditQueryId: queryId }),
      closeAuditDetail: () => set({ openedAuditQueryId: null }),

      streamAbortController: null,
      setStreamAbortController: (c) => set({ streamAbortController: c }),

      // Phase 2 — Plan 02-12.
      sidebarOpen: defaultSidebarOpen(),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      scopeDialogOpen: null,
      setScopeDialogOpen: (mode) => set({ scopeDialogOpen: mode }),

      editingMessageId: null,
      setEditingMessageId: (id) => set({ editingMessageId: id }),

      commandOpen: false,
      setCommandOpen: (open) => set({ commandOpen: open }),
      helpOpen: false,
      setHelpOpen: (open) => set({ helpOpen: open }),
    }),
    {
      name: "rag-ui",
      storage: createJSONStorage(safeStorage),
      // Only sidebarOpen persists (D-10). scopeDialogOpen + editingMessageId
      // stay ephemeral per Plan 02-12 truths.
      partialize: (state) => ({ sidebarOpen: state.sidebarOpen }),
    },
  ),
);
