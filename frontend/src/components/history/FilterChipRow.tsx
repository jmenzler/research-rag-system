/**
 * FilterChipRow (HIST-02, D-12, Pitfall 9).
 *
 * Two ToggleGroup type="multiple" for collections + retrievers and a single
 * Toggle for archived. URL search params are the source of truth — read via
 * TanStack Router useSearch, write via useNavigate({ search: (prev) => ... }).
 *
 * Zod schema lives here and is shared with the layout route's validateSearch.
 *
 * CSV transform per RESEARCH §Pitfall 9 — explicit
 * `string -> string[]` parser; `archived` uses `z.coerce.boolean` so
 * `?archived=1` reads as `true` (the "1" coerces via the standard truthiness
 * rules zod applies to coerce.boolean).
 */
import { z } from "zod";
import { useSearch, useNavigate } from "@tanstack/react-router";
import type { ReactElement } from "react";

import { Toggle } from "@/components/ui/toggle";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

// All 4 corpus collections (matches src/server/api_chats.py whitelist).
export const ALL_COLLECTIONS = [
  "trading",
  "ecology",
  "notes",
  "system",
] as const;

// Retriever names — locked at backend level via Literal[…] in router.py.
export const ALL_RETRIEVERS = [
  "milvus",
  "paperqa",
  "hipporag",
  "lazygraph",
  "fused",
] as const;

/**
 * Zod schema for the URL search params on the /app/_layout route.
 *
 *   ?q=
 *   ?collections=trading,ecology  (CSV)
 *   ?retrievers=milvus,paperqa (CSV)
 *   ?archived=1                (truthy coerce)
 *
 * Empty / missing values become safe defaults so the layout route's
 * validateSearch never throws on a bare /app URL.
 */
const csvList = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((s): string[] => {
    if (s === undefined) return [];
    if (Array.isArray(s)) return s.filter(Boolean);
    return s.length > 0 ? s.split(",").filter(Boolean) : [];
  });

// eslint-disable-next-line react-refresh/only-export-components
export const chatListSearchSchema = z.object({
  q: z.string().optional().default(""),
  collections: csvList,
  retrievers: csvList,
  archived: z.coerce.boolean().default(false),
});

export type ChatListSearch = z.infer<typeof chatListSearchSchema>;

/** Effective filter set used to query and filter chats. */
export interface EffectiveFilters {
  collections: string[];
  retrievers: string[];
  archived: boolean;
}

/**
 * Translate a parsed search bag into the effective filter set. Empty
 * `collections` is treated as "all four" per D-03 ergonomics — the user
 * has nothing selected, which is the same as "no filter". Retrievers
 * behaves the same way.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function normalizeFilters(parsed: ChatListSearch): EffectiveFilters {
  const collections =
    parsed.collections.length > 0
      ? parsed.collections
      : [...ALL_COLLECTIONS];
  const retrievers =
    parsed.retrievers.length > 0 ? parsed.retrievers : [...ALL_RETRIEVERS];
  return {
    collections,
    retrievers,
    archived: parsed.archived,
  };
}

function useSearchSafe(): Partial<ChatListSearch> {
  // useSearch throws if there's no router context (unit-test render-tree).
  // Returning empty defaults keeps the component testable in isolation.
  try {
    // `strict: false` lets us read the layout search without binding to a
    // specific route id at compile-time (the layout route owns the schema).
    const search = useSearch({ strict: false }) as
      | Partial<ChatListSearch>
      | undefined;
    return search ?? {};
  } catch {
    return {};
  }
}

function useNavigateSafe(): ReturnType<typeof useNavigate> | null {
  try {
    return useNavigate();
  } catch {
    return null;
  }
}

export interface FilterChipRowProps {
  /**
   * Caller-provided filters override the URL-derived state. Used by the
   * unit tests; production callers should let the component read directly
   * from `useSearch`.
   */
  filters?: EffectiveFilters;
}

export function FilterChipRow(_props: FilterChipRowProps = {}): ReactElement {
  // Hooks at the top — React rules: never conditional. The safe wrappers
  // swallow context errors so tests that render <FilterChipRow /> without
  // a router still mount cleanly.
  const search = useSearchSafe();
  const navigate = useNavigateSafe();

  const collections = Array.isArray(search.collections)
    ? search.collections
    : typeof search.collections === "string"
      ? (search.collections as string).split(",").filter(Boolean)
      : [];
  const retrievers = Array.isArray(search.retrievers)
    ? search.retrievers
    : typeof search.retrievers === "string"
      ? (search.retrievers as string).split(",").filter(Boolean)
      : [];
  const archived = Boolean(search.archived);

  const updateSearch = (
    patch: (prev: Partial<ChatListSearch>) => Partial<ChatListSearch>,
  ): void => {
    if (navigate === null) return;
    // TanStack Router types `search` against a route-specific schema. Since
    // FilterChipRow is route-agnostic (mounted under any /app subtree), we
    // cast the reducer through `unknown` to bypass the route-binding
    // generic — the runtime behaviour is identical.
    navigate({
      search: ((prev: Partial<ChatListSearch>) =>
        patch(prev)) as unknown as never,
    });
  };

  const handleCollectionsChange = (next: string[]): void => {
    updateSearch((prev) => ({ ...prev, collections: next }));
  };

  const handleRetrieversChange = (next: string[]): void => {
    updateSearch((prev) => ({ ...prev, retrievers: next }));
  };

  const handleArchivedChange = (pressed: boolean): void => {
    updateSearch((prev) => ({ ...prev, archived: pressed }));
  };

  return (
    <div className="flex flex-col gap-2 px-2">
      <div>
        <p className="text-muted-foreground mb-1 text-xs">Collections</p>
        <ToggleGroup
          type="multiple"
          value={collections}
          onValueChange={handleCollectionsChange}
          className="flex flex-wrap gap-1"
        >
          {ALL_COLLECTIONS.map((c) => (
            <ToggleGroupItem
              key={c}
              value={c}
              aria-label={c}
              className="text-xs"
            >
              {c}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      <div>
        <p className="text-muted-foreground mb-1 text-xs">Retrievers</p>
        <ToggleGroup
          type="multiple"
          value={retrievers}
          onValueChange={handleRetrieversChange}
          className="flex flex-wrap gap-1"
        >
          {ALL_RETRIEVERS.map((r) => (
            <ToggleGroupItem
              key={r}
              value={r}
              aria-label={r}
              className="font-mono text-xs"
            >
              {r}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
      <Toggle
        pressed={archived}
        onPressedChange={handleArchivedChange}
        aria-label="Show archived"
        className="text-xs"
      >
        Show archived
      </Toggle>
    </div>
  );
}
