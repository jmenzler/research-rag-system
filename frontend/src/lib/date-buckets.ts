/**
 * Date-bucketing for the chat history sidebar (HIST-01, D-11).
 *
 * Pure function — no React, no globals beyond `new Date()` default.
 *
 * Buckets (in render order, per UI-SPEC Copywriting Contract):
 *   Today / Yesterday / Last 7 days / Last 30 days / Older
 *
 * Empty buckets are filtered from the output so the UI does not render a
 * bucket label with zero children (D-11 truth: "empty buckets hidden").
 *
 * MRU ordering inside a bucket is preserved from the input array — the
 * caller is responsible for sorting `chats` newest-first BEFORE calling
 * this function (server returns chats already ordered by updated_at DESC
 * — see src/server/api_chats.py).
 *
 * DST safety: uses `date-fns#differenceInCalendarDays` which compares
 * calendar dates (not raw 24h ticks), so a 23-hour day across the fall-back
 * boundary still classifies "yesterday" as Yesterday. Phase 2 RED test
 * `date-buckets.test.ts` anchors at 2024-11-03 to verify this.
 */
import { differenceInCalendarDays } from "date-fns";

export const BUCKET_ORDER = [
  "Today",
  "Yesterday",
  "Last 7 days",
  "Last 30 days",
  "Older",
] as const;
export type BucketLabel = (typeof BUCKET_ORDER)[number];

/** Minimum shape consumed — any chat-like object with `id` + `updated_at`. */
export interface ChatLike {
  id: string;
  updated_at: string;
}

export interface Bucket<T extends ChatLike = ChatLike> {
  label: BucketLabel;
  items: T[];
}

function classify(updated: Date, now: Date): BucketLabel {
  const days = differenceInCalendarDays(now, updated);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days <= 7) return "Last 7 days";
  if (days <= 30) return "Last 30 days";
  return "Older";
}

/**
 * Bucket an array of chat-like rows into the 5 date buckets, in BUCKET_ORDER,
 * with empty buckets dropped from the output. Input ordering is preserved
 * inside each bucket.
 */
export function bucketByDate<T extends ChatLike>(
  items: T[],
  now: Date = new Date(),
): Bucket<T>[] {
  const out: Record<BucketLabel, T[]> = {
    Today: [],
    Yesterday: [],
    "Last 7 days": [],
    "Last 30 days": [],
    Older: [],
  };
  for (const item of items) {
    out[classify(new Date(item.updated_at), now)].push(item);
  }
  return BUCKET_ORDER.filter((label) => out[label].length > 0).map((label) => ({
    label,
    items: out[label],
  }));
}

/**
 * Plan-artifact-named alias: returns the same logical buckets keyed by label.
 * Provided for callers that need O(1) bucket lookup; the array form
 * (`bucketByDate`) is the one used by `<HistorySidebar>` because it
 * preserves render order and pre-filters empties.
 */
export function bucketChatsByDate<T extends ChatLike>(
  items: T[],
  now: Date = new Date(),
): Record<BucketLabel, T[]> {
  const out: Record<BucketLabel, T[]> = {
    Today: [],
    Yesterday: [],
    "Last 7 days": [],
    "Last 30 days": [],
    Older: [],
  };
  for (const item of items) {
    out[classify(new Date(item.updated_at), now)].push(item);
  }
  return out;
}
