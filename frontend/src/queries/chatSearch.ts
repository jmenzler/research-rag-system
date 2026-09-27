/**
 * Plan 02-12 — FTS5 chat-content search (HIST-04).
 *
 * Hits GET /api/chats/search (Plan 02-04). Server escapes the query string
 * (`safe_q` per Plan 02-04 contract) and returns up to 50 results with
 * `<mark>…</mark>` highlighting in the `snippet` field.
 *
 * Caching discipline:
 *   - staleTime: 0 — CRIT-3 mandate. When the user returns to a
 *     search-active sidebar (route remount / window focus), refetch.
 *   - placeholderData: keepPreviousData — UI keeps the prior result list
 *     mounted while a new query fires. The spinner in the input's right
 *     padding signals "refreshing"; the list does not flash empty.
 *
 * Debouncing: NOT done here. TanStack Query handles caching only;
 * the calling component (HistorySearchInput) debounces the `q` URL param
 * update via setTimeout / useEffect (UI-SPEC: 200ms after last keystroke).
 *
 * Enable gate: `enabled: q.trim().length > 0`. Empty `q` returns the
 * canonical chat list via useChatsListQuery; the search endpoint is only
 * hit when the user has typed something.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ChatSearchFilters, ChatSearchPayload } from "@/lib/api";

export function useChatSearchQuery(
  q: string,
  filters: ChatSearchFilters = {},
): UseQueryResult<ChatSearchPayload> {
  const trimmed = q.trim();
  return useQuery({
    queryKey: [
      "chats",
      "search",
      {
        q: trimmed,
        collections: filters.collections ?? [],
        retrievers: filters.retrievers ?? [],
        archived: filters.archived ?? false,
      },
    ],
    queryFn: () => api.chats.search(trimmed, filters),
    enabled: trimmed.length > 0,
    placeholderData: keepPreviousData, // RESEARCH §Pattern 3 + CRIT-3
    staleTime: 0,
  });
}
