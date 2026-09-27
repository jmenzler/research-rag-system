/**
 * /app/_layout — TanStack Router layout route (HIST-01, D-10).
 *
 * Adds a persistent CSS grid with column 1 = sidebar (17rem expanded,
 * 3rem collapsed) and column 2 = the chat outlet. zod-validated search
 * schema lives in FilterChipRow.tsx so the layout and the chip row
 * share the same parser.
 *
 * The sidebar width transitions over 200ms; honoring
 * `prefers-reduced-motion: reduce` is handled by Tailwind's
 * `motion-reduce:transition-none` modifier (D-10).
 *
 * NOTE: this is added alongside the existing /app/chat/$chatId route
 * tree per RESEARCH option A (least disruptive). The existing
 * frontend/src/routes/app.tsx and frontend/src/routes/app/chat.tsx are
 * unchanged.
 *
 * Phase 6 Plan 02: global hotkeys, command palette, keyboard help sheet,
 * and offline banner all mount here — single app shell, single source of truth.
 */
import { createFileRoute, Outlet, useNavigate, useParams } from "@tanstack/react-router";
import { lazy, Suspense, useMemo, type ReactElement } from "react";

import { useUiStore } from "@/state/uiStore";
import { HistorySidebar } from "@/components/history/HistorySidebar";
import { CreateChatDialog } from "@/components/chat/CreateChatDialog";
import {
  chatListSearchSchema,
  normalizeFilters,
} from "@/components/history/FilterChipRow";
import { useGlobalHotkeys } from "@/hooks/useGlobalHotkeys";
import { buildKeymap } from "@/lib/keymap";
import { OfflineBanner } from "@/components/OfflineBanner";

// Modal surfaces kept off the chat-shell critical path: their chunk loads on
// first open, not on app mount, so the eager index bundle stays under budget.
const CommandPalette = lazy(() =>
  import("@/components/CommandPalette").then((m) => ({ default: m.CommandPalette })),
);
const KeyboardHelpSheet = lazy(() =>
  import("@/components/KeyboardHelpSheet").then((m) => ({ default: m.KeyboardHelpSheet })),
);

export const Route = createFileRoute("/app/_layout")({
  component: AppLayout,
  validateSearch: chatListSearchSchema,
});

function AppLayout(): ReactElement {
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const setScopeDialogOpen = useUiStore((s) => s.setScopeDialogOpen);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);
  const commandOpen = useUiStore((s) => s.commandOpen);
  const setCommandOpen = useUiStore((s) => s.setCommandOpen);
  const helpOpen = useUiStore((s) => s.helpOpen);
  const setHelpOpen = useUiStore((s) => s.setHelpOpen);

  const navigate = useNavigate();
  const sidebarWidth = sidebarOpen ? "17rem" : "3rem";

  const search = Route.useSearch();
  const effectiveFilters = normalizeFilters(search);

  // The active chat id (for the row indicator) lives one level deeper
  // in the route tree at /app/chat/$chatId. Read it defensively via
  // useParams with strict:false so the layout still renders on /app
  // itself (no chat selected).
  let activeChatId: string | undefined;
  try {
    const params = useParams({ strict: false }) as { chatId?: string };
    activeChatId = params?.chatId;
  } catch {
    activeChatId = undefined;
  }

  // Build the keymap once per mount — context values are stable Zustand selectors.
  const bindings = useMemo(
    () =>
      buildKeymap({
        navigate,
        setCommandOpen,
        setHelpOpen,
        setScopeDialogOpen,
        toggleSidebar,
        setSidebarOpen,
      }),
    // Stable Zustand action refs + navigate ref — safe to omit from deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useGlobalHotkeys(bindings);

  return (
    <>
      <div className="flex h-screen flex-col overflow-hidden">
        <OfflineBanner />
        <div
          className="grid min-h-0 flex-1 overflow-hidden transition-[grid-template-columns] duration-200 ease-in-out motion-reduce:transition-none"
          style={{ gridTemplateColumns: `${sidebarWidth} 1fr`, gridTemplateRows: "minmax(0, 1fr)" }}
        >
          <HistorySidebar
            activeChatId={activeChatId}
            filters={effectiveFilters}
            searchQuery={search.q}
            onNewChat={() => setScopeDialogOpen("create")}
          />
          <main className="min-h-0 min-w-0 overflow-hidden">
            <Outlet />
          </main>
          {/* Create-mode scope dialog mounted at the layout level so it is
              available on the empty chat index AND every chat surface. The
              in-chat ChatScopeBar still drives the edit-mode dialog from
              ChatSurface; this instance only handles the "create" flow. */}
          <CreateChatDialog />
        </div>
      </div>
      {commandOpen && (
        <Suspense fallback={null}>
          <CommandPalette open onOpenChange={setCommandOpen} bindings={bindings} />
        </Suspense>
      )}
      {helpOpen && (
        <Suspense fallback={null}>
          <KeyboardHelpSheet open onOpenChange={setHelpOpen} bindings={bindings} />
        </Suspense>
      )}
    </>
  );
}
