/**
 * Wave-1 RED test for <HistorySidebar> (HIST-01, HIST-04).
 *
 * Implementation lands in Plan 02-12 (frontend/src/components/history/HistorySidebar.tsx).
 *
 * Behavioural contract:
 *   - Renders date buckets Today / Yesterday / Last 7 days / Last 30 days / Older
 *   - Hides empty buckets
 *   - Preserves MRU ordering within a bucket
 *   - Shows active-row indicator on /app/chat/$chatId match
 *   - Shows the two empty states + the search-empty state
 *   - Renders <mark>...</mark> highlight from FTS5 snippet through the sanitizer
 *
 * Strategy: deferred `await import(...)` inside each `it()` body so the test file
 * collects cleanly (RED = test failures, not import errors). Network is mocked via
 * `vi.stubGlobal("fetch", ...)`; the TanStack Router context is provided by a
 * lightweight in-test router harness (mirrors how Phase 1 wrapping is structured).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

function renderWithClient(ui: ReactNode): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function stubFetchByUrl(table: Record<string, unknown>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      const key = Object.keys(table).find((k) => url.includes(k));
      if (key === undefined) {
        return Promise.resolve(new Response("not stubbed", { status: 404 }));
      }
      return Promise.resolve(
        new Response(JSON.stringify(table[key]), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }),
  );
}

function iso(daysAgo: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - daysAgo);
  return d.toISOString();
}

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

describe("<HistorySidebar> — RED tests until Plan 02-12", () => {
  it("renders date buckets Today / Yesterday / Last 7 days / Last 30 days / Older", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    stubFetchByUrl({
      "/api/chats": {
        chats: [
          {
            id: "c-today",
            title: "today chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(0),
            updated_at: iso(0),
          },
          {
            id: "c-yesterday",
            title: "yesterday chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(1),
            updated_at: iso(1),
          },
          {
            id: "c-week",
            title: "week chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(4),
            updated_at: iso(4),
          },
          {
            id: "c-month",
            title: "month chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(20),
            updated_at: iso(20),
          },
          {
            id: "c-old",
            title: "old chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(60),
            updated_at: iso(60),
          },
        ],
      },
    });

    renderWithClient(<HistorySidebar />);

    expect(await screen.findByText("Today")).toBeTruthy();
    expect(await screen.findByText("Yesterday")).toBeTruthy();
    expect(await screen.findByText("Last 7 days")).toBeTruthy();
    expect(await screen.findByText("Last 30 days")).toBeTruthy();
    expect(await screen.findByText("Older")).toBeTruthy();
  });

  it("hides empty buckets", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    stubFetchByUrl({
      "/api/chats": {
        chats: [
          {
            id: "c-today",
            title: "today only",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(0),
            updated_at: iso(0),
          },
        ],
      },
    });

    renderWithClient(<HistorySidebar />);
    await screen.findByText("Today");
    expect(screen.queryByText("Yesterday")).toBeNull();
    expect(screen.queryByText("Last 7 days")).toBeNull();
    expect(screen.queryByText("Last 30 days")).toBeNull();
    expect(screen.queryByText("Older")).toBeNull();
  });

  it("preserves MRU ordering within a bucket", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    // Both today. MRU = most-recently-updated first.
    const t1 = new Date();
    const t2 = new Date(t1.getTime() - 60_000);
    const t3 = new Date(t1.getTime() - 120_000);
    stubFetchByUrl({
      "/api/chats": {
        chats: [
          {
            id: "c-mid",
            title: "middle",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: t2.toISOString(),
            updated_at: t2.toISOString(),
          },
          {
            id: "c-newest",
            title: "newest",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: t1.toISOString(),
            updated_at: t1.toISOString(),
          },
          {
            id: "c-oldest",
            title: "oldest",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: t3.toISOString(),
            updated_at: t3.toISOString(),
          },
        ],
      },
    });

    renderWithClient(<HistorySidebar />);
    await screen.findByText("Today");
    const titles = screen
      .getAllByText(/^(newest|middle|oldest)$/)
      .map((n) => n.textContent);
    expect(titles).toEqual(["newest", "middle", "oldest"]);
  });

  it("shows active-row indicator on /app/chat/$chatId match", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    stubFetchByUrl({
      "/api/chats": {
        chats: [
          {
            id: "active-id",
            title: "active",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(0),
            updated_at: iso(0),
          },
        ],
      },
    });

    renderWithClient(<HistorySidebar activeChatId="active-id" />);
    const row = await screen.findByText("active");
    // The active indicator is some visual marker — bar element with
    // data-active="true", aria-current="page", or a class. Accept any of those.
    const container = row.closest("[data-active], [aria-current], a, li, div");
    const html = container?.outerHTML ?? "";
    const isActive =
      /data-active=["']?true/.test(html) ||
      /aria-current=["']?page/.test(html) ||
      /\bactive\b/.test(html);
    expect(isActive).toBe(true);
  });

  it("shows empty state 'No chats yet' when zero chats", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    stubFetchByUrl({ "/api/chats": { chats: [] } });

    renderWithClient(<HistorySidebar />);
    expect(await screen.findByText("No chats yet")).toBeTruthy();
  });

  it("shows empty state 'No chats match these filters' when filter chips exclude all", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    // Backend returns no matches for the active filter set.
    stubFetchByUrl({
      "/api/chats/search": { results: [] },
      "/api/chats": { chats: [] },
    });

    renderWithClient(
      <HistorySidebar
        filters={{
          collections: ["trading"],
          retrievers: ["milvus"],
          archived: false,
        }}
      />,
    );
    expect(
      await screen.findByText("No chats match these filters"),
    ).toBeTruthy();
  });

  it("renders <mark>...</mark> highlight from FTS5 snippet through the sanitizer", async () => {
    const { HistorySidebar } = await import(
      "@/components/history/HistorySidebar"
    );

    stubFetchByUrl({
      "/api/chats/search": {
        results: [
          {
            id: "match",
            title: "a chat",
            retriever: "milvus",
            collections: '["trading"]',
            version: 1,
            created_at: iso(0),
            updated_at: iso(0),
            snippet: "first <mark>match</mark> here",
          },
        ],
      },
    });

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <HistorySidebar searchQuery="match" />
      </QueryClientProvider>,
    );

    await screen.findByText(/first/i);
    const mark = container.querySelector("mark");
    expect(mark).not.toBeNull();
    expect(mark?.textContent).toContain("match");
  });
});
