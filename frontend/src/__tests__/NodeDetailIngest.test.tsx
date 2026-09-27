/**
 * Tests for NodeDetail ingest wiring (Plan 05-05 Task 1).
 *
 * Covers:
 * - "Add to corpus" button enabled when status === "missing"
 * - No ingest button rendered when status === "in-other" (D-11)
 * - No ingest button rendered when status === "in"
 * - "Add to corpus (next phase)" placeholder is gone
 *
 * jsdom-safe: no WebGL dependency.
 */
import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock API so TanStack Query hooks don't make real fetch calls
vi.mock("@/lib/api", () => ({
  api: {
    ingest: { start: vi.fn().mockResolvedValue({ run_ids: [], n_enqueued: 0 }) },
    maps: { create: vi.fn(), list: vi.fn(), get: vi.fn(), coverage: vi.fn(), delete: vi.fn() },
    graph: {
      collections: vi.fn(),
      corpusView: vi.fn(),
      seedSearch: vi.fn(),
      walk: vi.fn(),
      expand: vi.fn(),
      inCorpusCheck: vi.fn(),
      node: vi.fn().mockResolvedValue({
        corpus_id: 1,
        title: "Test",
        year: 2021,
        citation_count: 10,
        authors: [],
        abstract: null,
        arxiv_id: null,
        doi: null,
      }),
      delete: vi.fn(),
    },
  },
}));

// Mock useQueueStream to prevent EventSource in tests
vi.mock("@/hooks/useQueueStream", () => ({
  useQueueStream: () => ({ jobs: {}, counts: { running: 0, queued: 0 }, isConnected: false }),
}));

import { NodeDetail } from "@/components/graph/NodeDetail";
import type { GraphNode, InCorpusResult } from "@/queries/graph";

function makeNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    corpusId: 42,
    title: "Test Paper",
    year: 2023,
    citationCount: 100,
    shortCite: null,
    isSeed: false,
    ...overrides,
  };
}

function makeInCorpus(inCorpus: boolean, collection: string | null = null): InCorpusResult {
  return { inCorpus, collection };
}

function renderNodeDetail(node: GraphNode, inCorpus: InCorpusResult | undefined, collection = "trading") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <NodeDetail
        node={node}
        inCorpus={inCorpus}
        collection={collection}
        onExpand={() => {}}
        onWalk={() => {}}
      />
    </QueryClientProvider>,
  );
}

describe("NodeDetail ingest button", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders 'Add to corpus' button enabled when node is missing", () => {
    const node = makeNode({ isSeed: false });
    const inCorpus = makeInCorpus(false);
    renderNodeDetail(node, inCorpus);

    const btn = screen.getByRole("button", { name: /add to corpus/i });
    expect(btn).toBeInTheDocument();
    expect(btn).not.toBeDisabled();
  });

  it("does NOT render an ingest button for in-other (violet) nodes (D-11)", () => {
    const node = makeNode({ isSeed: false });
    const inCorpus = makeInCorpus(true, "ecology");
    renderNodeDetail(node, inCorpus);

    expect(screen.queryByRole("button", { name: /add to corpus/i })).toBeNull();
  });

  it("does NOT render an ingest button for in-corpus nodes", () => {
    const node = makeNode({ isSeed: false });
    const inCorpus = makeInCorpus(true, "trading");
    renderNodeDetail(node, inCorpus);

    expect(screen.queryByRole("button", { name: /add to corpus/i })).toBeNull();
  });

  it("does NOT render an ingest button for seed nodes", () => {
    const node = makeNode({ isSeed: true });
    renderNodeDetail(node, undefined);

    expect(screen.queryByRole("button", { name: /add to corpus/i })).toBeNull();
  });

  it("does NOT contain old placeholder copy 'next phase'", () => {
    const node = makeNode({ isSeed: false });
    const inCorpus = makeInCorpus(false);
    renderNodeDetail(node, inCorpus);

    expect(screen.queryByText(/next phase/i)).toBeNull();
  });
});
