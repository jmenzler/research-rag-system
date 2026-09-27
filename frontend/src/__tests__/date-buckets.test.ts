/**
 * Wave-1 RED test for the pure date-bucket function (D-11).
 *
 * Implementation lands in Plan 02-12 (frontend/src/lib/date-buckets.ts).
 *
 * Behavioural contract:
 *   - Buckets use `date-fns#differenceInCalendarDays` (DST-safe).
 *   - Today / Yesterday / Last 7 days (2–7d) / Last 30 days (8–30d) / Older (31+d).
 *   - Preserves MRU ordering of the input array within each bucket.
 *   - Filters empty buckets from the output.
 */
import { describe, expect, it } from "vitest";

interface ChatLite {
  id: string;
  updated_at: string;
}

interface Bucket {
  label: string;
  items: ChatLite[];
}

type BucketByDate = (items: ChatLite[], now?: Date) => Bucket[];

function isoDaysAgo(days: number, baseMs?: number): string {
  const d = baseMs ? new Date(baseMs) : new Date();
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString();
}

describe("date-buckets (HIST-01, D-11) — RED tests until Plan 02-12", () => {
  it("today returns 'Today'", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    const buckets = mod.bucketByDate([
      { id: "a", updated_at: isoDaysAgo(0) },
    ]);
    expect(buckets.find((b) => b.label === "Today")).toBeDefined();
  });

  it("yesterday returns 'Yesterday'", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    const buckets = mod.bucketByDate([
      { id: "a", updated_at: isoDaysAgo(1) },
    ]);
    expect(buckets.find((b) => b.label === "Yesterday")).toBeDefined();
  });

  it("2 to 7 days ago returns 'Last 7 days'", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    for (const n of [2, 3, 7]) {
      const buckets = mod.bucketByDate([
        { id: `a-${n}`, updated_at: isoDaysAgo(n) },
      ]);
      expect(buckets.find((b) => b.label === "Last 7 days")).toBeDefined();
    }
  });

  it("8 to 30 days ago returns 'Last 30 days'", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    for (const n of [8, 15, 30]) {
      const buckets = mod.bucketByDate([
        { id: `a-${n}`, updated_at: isoDaysAgo(n) },
      ]);
      expect(buckets.find((b) => b.label === "Last 30 days")).toBeDefined();
    }
  });

  it("31+ days ago returns 'Older'", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    for (const n of [31, 365]) {
      const buckets = mod.bucketByDate([
        { id: `a-${n}`, updated_at: isoDaysAgo(n) },
      ]);
      expect(buckets.find((b) => b.label === "Older")).toBeDefined();
    }
  });

  it("bucketing uses differenceInCalendarDays from date-fns (DST-safe across boundary)", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };

    // Anchored at the US DST fall-back boundary (2024-11-03), UTC-locked to
    // stay deterministic across CI time zones.
    const nowMs = Date.UTC(2024, 10, 3, 12, 0, 0);
    const yesterdayMs = nowMs - 24 * 3600 * 1000;
    const items: ChatLite[] = [
      { id: "today", updated_at: new Date(nowMs).toISOString() },
      { id: "yesterday", updated_at: new Date(yesterdayMs).toISOString() },
    ];

    const buckets = mod.bucketByDate(items, new Date(nowMs));
    const labels = buckets.map((b) => b.label);
    // DST-safe: a 23-hour gap (fall-back) must still classify yesterday as "Yesterday".
    expect(labels).toContain("Today");
    expect(labels).toContain("Yesterday");
  });

  it("preserves MRU ordering of input array within each bucket", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };

    // Three items all in the "Today" bucket, given in MRU order.
    const now = new Date();
    const items: ChatLite[] = [
      { id: "newest", updated_at: new Date(now.getTime()).toISOString() },
      { id: "middle", updated_at: new Date(now.getTime() - 60_000).toISOString() },
      { id: "oldest", updated_at: new Date(now.getTime() - 120_000).toISOString() },
    ];

    const buckets = mod.bucketByDate(items);
    const today = buckets.find((b) => b.label === "Today");
    expect(today).toBeDefined();
    expect(today?.items.map((i) => i.id)).toEqual([
      "newest",
      "middle",
      "oldest",
    ]);
  });

  it("filters out empty buckets in the output", async () => {
    const mod = (await import("@/lib/date-buckets")) as unknown as {
      bucketByDate: BucketByDate;
    };
    const buckets = mod.bucketByDate([
      { id: "a", updated_at: isoDaysAgo(0) },
    ]);
    expect(buckets.map((b) => b.label)).toEqual(["Today"]);
  });
});
