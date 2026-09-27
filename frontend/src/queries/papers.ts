/**
 * Convenience re-export for callers that conceptually want a "papers"
 * namespace.  The actual hook lives in `@/queries/chats` (Plan 05) —
 * CitationHoverCard + CitationSheet (Plan 07) import from here.
 */
export { useChunkQuery } from "@/queries/chats";
