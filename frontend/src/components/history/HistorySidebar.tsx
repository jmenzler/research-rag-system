/**
 * HistorySidebar (HIST-01, HIST-04) — persistent left column.
 *
 * Composition:
 *   SidebarHeader + HistorySearchInput + FilterChipRow + ScrollArea {
 *     DateBucketGroup[] (when q is empty)
 *     | flat search results with <mark> snippets (when q is non-empty)
 *   } + RenameChatDialog + DeleteChatConfirm (mounted once)
 *
 * Data flow:
 *   - When `searchQuery` non-empty → `useChatSearchQuery(q, filters)` and
 *     render flat list with rehype-sanitize'd snippet.
 *   - When empty → `useChatsListQuery()`, apply client-side filters,
 *     bucket via `bucketByDate`, render in BUCKET_ORDER skipping empties.
 *
 * Sidebar is fully prop-driven so it can be tested without a router
 * context: the test passes `activeChatId`, `filters`, `searchQuery`
 * directly. Production code mounts it inside the layout route where the
 * router supplies these via URL search params.
 *
 * FilterChipRow is rendered only when no `filters` prop is supplied —
 * that signal means "use the live URL state" and implies a router
 * context. With an explicit `filters` prop (test scenario), the chip row
 * is hidden so the test does not try to render router-bound UI without
 * a router.
 */
import { useState, useMemo } from "react";
import type { ReactElement } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";

import { toast } from "sonner";

import { useChatsListQuery } from "@/queries/chats";
import { api, isLock409, OPTIMISTIC_409_TOAST } from "@/lib/api";
import type { ChatSummary, ChatSearchPayload } from "@/lib/api";
import { CITATION_SANITIZE_SCHEMA } from "@/lib/sanitize-schema";
import { bucketByDate } from "@/lib/date-buckets";
import {
  ALL_COLLECTIONS,
  ALL_RETRIEVERS,
  FilterChipRow,
} from "@/components/history/FilterChipRow";
import type { ChatRow } from "@/components/history/ChatListItem";
import { DateBucketGroup } from "@/components/history/DateBucketGroup";
import { HistorySearchInput } from "@/components/history/HistorySearchInput";
import { SidebarHeader } from "@/components/history/SidebarHeader";
import { SidebarNav } from "@/components/history/SidebarNav";
import { RenameChatDialog } from "@/components/history/RenameChatDialog";
import { DeleteChatConfirm } from "@/components/history/DeleteChatConfirm";
import { useUpdateChatMutation } from "@/queries/chats";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useUiStore } from "@/state/uiStore";
import { VersionPill } from "@/components/VersionPill";
import { FooterQueuePill } from "@/components/graph/FooterQueuePill";
import { MessageSquare } from "lucide-react";

export interface HistorySidebarFilters {
  collections: string[];
  retrievers: string[];
  archived: boolean;
}

export interface HistorySidebarProps {
  /** Active chat id for the active-row indicator. */
  activeChatId?: string | undefined;
  /**
   * Filter set. When provided, overrides URL-based state AND disables the
   * FilterChipRow render (the caller is responsible for the chip row).
   */
  filters?: HistorySidebarFilters | undefined;
  /** Search query string. When non-empty, triggers FTS5 search. */
  searchQuery?: string | undefined;
  /** Called when the user clicks "New chat". */
  onNewChat?: (() => void) | undefined;
}

function toChatRow(c: ChatSummary): ChatRow {
  // ChatSummary has `archived: boolean | undefined` in Plan 02-04. Map to
  // the row shape consumed by ChatListItem (archived defaults to false).
  const summary = c as ChatSummary & { archived?: boolean };
  return {
    id: summary.id,
    title: summary.title,
    retriever: summary.retriever,
    collections: summary.collections,
    version: summary.version,
    created_at: summary.created_at,
    updated_at: summary.updated_at,
    archived: summary.archived === true,
  };
}

function applyFilters(
  rows: ChatRow[],
  filters: HistorySidebarFilters | undefined,
): ChatRow[] {
  if (filters === undefined) return rows;
  // Empty collections / retrievers means "all" — see normalizeFilters.
  const effectiveCols =
    filters.collections.length > 0 ? filters.collections : [...ALL_COLLECTIONS];
  const effectiveRetrievers =
    filters.retrievers.length > 0 ? filters.retrievers : [...ALL_RETRIEVERS];
  return rows.filter((row) => {
    if (!filters.archived && row.archived === true) return false;
    if (!effectiveRetrievers.includes(row.retriever)) return false;
    // collections is a JSON string per ChatSummary contract — parse defensively.
    let cols: string[] = [];
    try {
      const parsed = JSON.parse(row.collections);
      if (Array.isArray(parsed)) cols = parsed.map(String);
    } catch {
      /* ignore — treat as no collections */
    }
    if (cols.length === 0) return true; // chat has no collection set — keep
    return cols.some((c) => effectiveCols.includes(c));
  });
}

/**
 * Local search query that does NOT depend on a router context. Inline to
 * avoid the chatSearch hook's enabled gate diverging across the test/prod
 * paths — `enabled` is gated on `q !== ""`.
 */
function useSidebarSearch(
  q: string,
  filters: HistorySidebarFilters | undefined,
): { data: ChatSearchPayload | undefined; isFetching: boolean } {
  const filtersKey = {
    collections: filters?.collections ?? [],
    retrievers: filters?.retrievers ?? [],
    archived: filters?.archived ?? false,
  };
  const query = useQuery({
    queryKey: ["chats", "search", { q: q.trim(), ...filtersKey }],
    queryFn: () =>
      api.chats.search(q.trim(), {
        collections: filtersKey.collections,
        retrievers: filtersKey.retrievers,
        archived: filtersKey.archived,
      }),
    enabled: q.trim().length > 0,
    staleTime: 0,
  });
  return { data: query.data, isFetching: query.isFetching };
}

export function HistorySidebar(props: HistorySidebarProps = {}): ReactElement {
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const listQuery = useChatsListQuery();
  const search = useSidebarSearch(props.searchQuery ?? "", props.filters);

  const [renameTarget, setRenameTarget] = useState<ChatRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ChatRow | null>(null);

  // Used by the row menu's Archive item.
  const archiveTargetId = renameTarget?.id ?? "";
  const archiveMutation = useUpdateChatMutation(archiveTargetId);
  void archiveMutation; // type-only — actual archive uses per-row mutations

  const rowsRaw: ChatRow[] = (listQuery.data?.chats ?? []).map(toChatRow);
  // The server returns chats ordered by updated_at DESC, but defensive
  // sort here so we tolerate unsorted payloads (e.g. test fixtures) and
  // preserve MRU semantics in the date-bucket display order.
  const sorted = [...rowsRaw].sort(
    (a, b) =>
      new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
  );
  const filtered = applyFilters(sorted, props.filters);

  const searching =
    typeof props.searchQuery === "string" && props.searchQuery.trim().length > 0;
  const searchResults = search.data?.results ?? [];

  const handleArchiveToggle = (chat: ChatRow): void => {
    // Inline mutation per row — avoids needing the rename mutation's chatId.
    // Re-implements the toggle directly via api.chats.patch since
    // useUpdateChatMutation needs a chatId at hook-call time (rules-of-hooks).
    void api.chats
      .patch(chat.id, {
        archived: !(chat.archived === true),
        expected_version: chat.version,
      })
      .catch((err: unknown) => {
        if (isLock409(err)) {
          toast.error(OPTIMISTIC_409_TOAST);
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        toast.error(`Couldn't update chat: ${message}`);
      })
      .finally(() => {
        void listQuery.refetch();
      });
  };

  const buckets = useMemo(() => bucketByDate<ChatRow>(filtered), [filtered]);

  // ---- Collapsed icon-rail rendering ------------------------------------
  if (!sidebarOpen) {
    return (
      <aside
        className="flex h-full flex-col border-r border-[var(--p3-border)]"
        style={{ background: "#0A0D14" }}
      >
        <SidebarHeader onNewChat={props.onNewChat} />
        <SidebarNav />
      </aside>
    );
  }

  // ---- Expanded rendering ----------------------------------------------
  return (
    <aside
      className="flex h-full flex-col gap-3 border-r border-[var(--p3-border)]"
      style={{ background: "#0A0D14" }}
    >
      <SidebarHeader onNewChat={props.onNewChat} />
      <SidebarNav />
      <HistorySearchInput isFetching={search.isFetching} />
      {props.filters === undefined && <FilterChipRow />}
      <div className="flex items-center gap-1.5 px-4 pt-1 text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--p3-muted-2)]">
        <MessageSquare className="size-3" />
        <span>Chat history</span>
      </div>
      <ScrollArea className="flex-1">
        <div className="flex flex-col gap-4 pb-4">
          {/* Loading skeleton when no previous data. */}
          {listQuery.isLoading && !searching && (
            <div className="flex flex-col gap-1 px-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="bg-muted/40 h-8 animate-pulse rounded"
                />
              ))}
            </div>
          )}

          {/* Search-results branch -------------------------------------- */}
          {searching && searchResults.length > 0 && (
            <ul className="flex flex-col">
              {searchResults.map((r) => (
                <li key={r.id} className="px-3 py-2">
                  <a
                    href={`/app/chat/${r.id}`}
                    className="block text-sm font-medium"
                  >
                    {r.title ?? "Untitled chat"}
                  </a>
                  <div className="text-muted-foreground text-xs">
                    {/* FTS5 snippets contain literal <mark>…</mark> HTML
                        wrapping matched terms. rehype-raw parses the raw
                        HTML into the hast tree; rehype-sanitize then
                        strips anything not in CITATION_SANITIZE_SCHEMA —
                        which allowlists <mark> with no attrs
                        (T-02-12-01 XSS mitigation). */}
                    <ReactMarkdown
                      rehypePlugins={[
                        rehypeRaw,
                        [rehypeSanitize, CITATION_SANITIZE_SCHEMA],
                      ]}
                    >
                      {r.snippet}
                    </ReactMarkdown>
                  </div>
                </li>
              ))}
            </ul>
          )}

          {/* Search empty -- no results for query. */}
          {searching && !search.isFetching && searchResults.length === 0 && (
            <div className="px-6 py-12 text-center">
              <p className="text-sm font-medium">
                No matches for &quot;
                {(props.searchQuery ?? "").length > 40
                  ? (props.searchQuery ?? "").slice(0, 40) + "…"
                  : props.searchQuery}
                &quot;
              </p>
              <p className="text-muted-foreground text-xs">
                Try fewer or different words.
              </p>
            </div>
          )}

          {/* Default (non-search) list. ------------------------------- */}
          {!searching && !listQuery.isLoading && (
            <>
              {filtered.length === 0 ? (
                props.filters !== undefined ? (
                  <div className="px-6 py-12 text-center">
                    <p className="text-sm font-medium">
                      No chats match these filters
                    </p>
                    <p className="text-muted-foreground text-xs">
                      Try removing a filter or searching different terms.
                    </p>
                  </div>
                ) : (
                  <div className="px-6 py-12 text-center">
                    <p className="text-sm font-medium">No chats yet</p>
                    <p className="text-muted-foreground text-xs">
                      Start a new chat to ground your questions in your
                      corpus.
                    </p>
                  </div>
                )
              ) : (
                buckets.map((b) => (
                  <DateBucketGroup
                    key={b.label}
                    label={b.label}
                    items={b.items}
                    activeChatId={props.activeChatId ?? null}
                    onRename={setRenameTarget}
                    onDelete={setDeleteTarget}
                    onArchiveToggle={handleArchiveToggle}
                  />
                ))
              )}
            </>
          )}
        </div>
      </ScrollArea>

      {/* Build SHA — relocated here from the old global app header so the
          version stays visible once the header was removed (VersionPill is
          unchanged; only its mount point moved). */}
      <div className="flex items-center justify-between border-t border-[var(--p3-border-soft)] px-3 py-2 text-[11px] text-[var(--p3-muted-2)]">
        <span className="font-mono">build</span>
        <div className="flex items-center gap-2">
          <FooterQueuePill />
          <VersionPill />
        </div>
      </div>

      <RenameChatDialog
        target={renameTarget}
        onClose={() => setRenameTarget(null)}
      />
      <DeleteChatConfirm
        target={deleteTarget}
        onClose={() => setDeleteTarget(null)}
      />
    </aside>
  );
}

// ChatRow is re-exported from ChatListItem for consumers that want the
// concrete row type (sidebar tests, parent route).
