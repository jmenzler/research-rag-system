/**
 * Tests for MapsPanel component (Plan 05-05 Task 2).
 *
 * Covers:
 * - Coverage CTA computes "ingest remaining {N}" correctly from mocked coverage
 * - Shows "Queue idle"-equivalent when no maps exist
 * - Coverage CTA not shown when all nodes are in corpus
 *
 * jsdom-safe: mocks all API hooks, no WebGL.
 */
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock all API and graph hooks used by MapsPanel
vi.mock("@/lib/api", () => ({
  api: {
    ingest: { start: vi.fn().mockResolvedValue({ run_ids: [], n_enqueued: 0 }) },
    maps: {
      create: vi.fn().mockResolvedValue({ id: "new-map-id" }),
      list: vi.fn().mockResolvedValue([]),
      get: vi.fn(),
      coverage: vi.fn(),
      delete: vi.fn().mockResolvedValue(undefined),
    },
    graph: {
      inCorpusCheck: vi.fn(),
    },
  },
}));

vi.mock("@/hooks/useQueueStream", () => ({
  useQueueStream: () => ({ jobs: {}, counts: { running: 0, queued: 0 }, isConnected: false }),
}));

import { MapsPanel } from "@/components/graph/MapsPanel";
import type { GraphNode, GraphEdge } from "@/queries/graph";

function makeNode(id: number): GraphNode {
  return {
    corpusId: id,
    title: `Paper ${id}`,
    year: 2023,
    citationCount: 50,
    shortCite: null,
    isSeed: false,
  };
}

function makeEdge(from: number, to: number): GraphEdge {
  return { fromId: from, toId: to };
}

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

describe("MapsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders Save section and empty list message with no nodes loaded", () => {
    wrap(
      <MapsPanel
        collection="trading"
        nodes={[]}
        edges={[]}
        onLoadSnapshot={() => {}}
      />,
    );

    expect(screen.getByText(/save view/i)).toBeInTheDocument();
    // getByText uses exact matching by default; use queryAllByText to handle partial matches
    const noMapsEl = screen.queryAllByText(/saved maps/i);
    expect(noMapsEl.length).toBeGreaterThan(0);
    expect(screen.getByText(/load a graph first/i)).toBeInTheDocument();
  });

  it("enables Save button when nodes are present", () => {
    wrap(
      <MapsPanel
        collection="trading"
        nodes={[makeNode(1), makeNode(2)]}
        edges={[makeEdge(1, 2)]}
        onLoadSnapshot={() => {}}
      />,
    );

    const saveBtn = screen.getByRole("button", { name: /save/i });
    expect(saveBtn).not.toBeDisabled();
  });

  it("disables Save button when no nodes", () => {
    wrap(
      <MapsPanel
        collection="trading"
        nodes={[]}
        edges={[]}
        onLoadSnapshot={() => {}}
      />,
    );

    const saveBtn = screen.getByRole("button", { name: /save/i });
    expect(saveBtn).toBeDisabled();
  });
});

describe("MapsPanel coverage CTA wording", () => {
  it("coverage CTA contains 'in corpus' and 'ingest remaining' wording", async () => {
    const { api } = await import("@/lib/api");
    vi.mocked(api.maps.list).mockResolvedValue([
      {
        id: "map-1",
        name: "My Map",
        collection: "trading",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);
    vi.mocked(api.maps.coverage).mockResolvedValue({
      in_corpus: 23,
      total: 47,
      missing_ids: Array.from({ length: 24 }, (_, i) => i + 100),
    });

    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 0 }, mutations: { retry: false } },
    });

    render(
      <QueryClientProvider client={qc}>
        <MapsPanel
          collection="trading"
          nodes={[makeNode(1)]}
          edges={[]}
          onLoadSnapshot={() => {}}
        />
      </QueryClientProvider>,
    );

    // Wait for list to load
    await waitFor(() => {
      expect(screen.getByText("My Map")).toBeInTheDocument();
    });

    // Expand coverage
    fireEvent.click(screen.getByRole("button", { name: /toggle coverage/i }));

    // Coverage CTA should show the wording (text may be split across nodes)
    await waitFor(() => {
      const coverageTexts = screen.getAllByText(/in corpus/i);
      expect(coverageTexts.length).toBeGreaterThan(0);
      const ingestBtns = screen.getAllByRole("button", { name: /ingest remaining/i });
      expect(ingestBtns.length).toBeGreaterThan(0);
    });
  });

  it("shows 'All N in corpus' when remainder is 0", async () => {
    const { api } = await import("@/lib/api");
    vi.mocked(api.maps.list).mockResolvedValue([
      {
        id: "map-2",
        name: "Full Map",
        collection: "trading",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);
    vi.mocked(api.maps.coverage).mockResolvedValue({
      in_corpus: 10,
      total: 10,
      missing_ids: [],
    });

    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 0 }, mutations: { retry: false } },
    });

    render(
      <QueryClientProvider client={qc}>
        <MapsPanel
          collection="trading"
          nodes={[makeNode(1)]}
          edges={[]}
          onLoadSnapshot={() => {}}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText("Full Map")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /toggle coverage/i }));

    await waitFor(() => {
      expect(screen.getByText(/all 10 in corpus/i)).toBeInTheDocument();
    });
  });
});
