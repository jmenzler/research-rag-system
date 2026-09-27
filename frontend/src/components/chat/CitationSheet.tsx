/**
 * D-08 / CHAT-04: Citation Sheet — slide-in from right.
 *
 * Drivers:
 *   - opens when `useUiStore.openedCitation` becomes non-null
 *   - fetches the parent chunk via `useChunkQuery` (staleTime: Infinity from
 *     Plan 05 — parent chunks are immutable, cache for the session)
 *   - closes on outside-click / Escape (radix-ui Dialog defaults via shadcn
 *     Sheet primitive — no custom keyboard handling needed)
 *
 * Parent-id resolution order:
 *   1. `openedCitation.parentId` (Plan 07 — RetrievalTraceInspector dispatches
 *      with parentId already known per-chunk)
 *   2. `citationRegistry[messageId::marker].parentId` (Plan 07 — chat surface
 *      pre-populates the registry from live SSE citations + persisted message
 *      rows)
 *   3. Synthetic key `m-${messageId}-${marker}` — fallback for tests and for
 *      historic message rows whose citation rows weren't registered yet.
 *      In production the synthetic key hits a 404 from /api/chunks/{id}, and
 *      the Sheet shows "Failed to load chunk" — which is the correct UX for an
 *      unresolved/missing citation. In test the global fetch stub returns
 *      parent text regardless of URL so the Sheet renders normally.
 *
 * T-01-W4-05 (server is canonical): we never invent a parent_id; we either
 * use the server-emitted one or fall back to a synthetic that triggers a
 * 404 — the Sheet never spoofs into a wrong parent.
 */
import { type ReactElement } from "react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { useUiStore } from "@/state/uiStore";
import { useChunkQuery } from "@/queries/papers";

function resolveParentId(
  opened: { messageId: string; marker: number; parentId?: string } | null,
  registry: Record<string, { parentId: string | null; resolved: boolean }>,
): string | null {
  if (opened === null) return null;
  if (typeof opened.parentId === "string" && opened.parentId.length > 0) {
    return opened.parentId;
  }
  const key = `${opened.messageId}::${opened.marker}`;
  const fromRegistry = registry[key]?.parentId;
  if (typeof fromRegistry === "string" && fromRegistry.length > 0) {
    return fromRegistry;
  }
  // Synthetic fallback so useChunkQuery fires (404 in prod, mocked in tests).
  return `m-${opened.messageId}-${opened.marker}`;
}

export function CitationSheet(): ReactElement {
  const opened = useUiStore((s) => s.openedCitation);
  const registry = useUiStore((s) => s.citationRegistry);
  const closeCitation = useUiStore((s) => s.closeCitation);

  const parentId = resolveParentId(opened, registry);
  const chunkQuery = useChunkQuery(parentId);
  const isOpen = opened !== null;

  return (
    <Sheet
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) closeCitation();
      }}
    >
      <SheetContent side="right" data-testid="citation-sheet">
        <SheetHeader>
          <SheetTitle>
            {opened ? `Citation [${opened.marker}]` : "Citation"}
          </SheetTitle>
          <SheetDescription>
            {chunkQuery.data?.source_file ?? "Citation source"}
          </SheetDescription>
        </SheetHeader>
        <div className="mt-4 space-y-3 px-4 pb-4">
          {chunkQuery.isLoading && (
            <p className="text-muted-foreground text-sm">Loading chunk…</p>
          )}
          {chunkQuery.isError && (
            <p className="text-destructive text-sm">Failed to load chunk.</p>
          )}
          {chunkQuery.data && (
            <>
              <div className="text-muted-foreground flex flex-wrap items-center gap-2 text-xs">
                <Badge variant="outline" className="font-mono">
                  {chunkQuery.data.source_file}
                </Badge>
                <span>page {chunkQuery.data.page_number}</span>
              </div>
              <article
                className="prose dark:prose-invert max-w-none text-sm whitespace-pre-wrap"
                data-testid="citation-sheet-text"
              >
                {chunkQuery.data.text}
              </article>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
