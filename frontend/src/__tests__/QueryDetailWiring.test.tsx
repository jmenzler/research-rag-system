/**
 * Component-level wiring for the Phase 3 query-detail forensic surface.
 *
 * Verifies QLOG-02 (stage payload lazy-fetch) and QLOG-03 (chunk text view):
 * the stage payload and chunk text are fetched ONLY when the user expands the
 * row — never eagerly on page load. Renders QueryDetail through a
 * RouterProvider harness on its real route.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createRouter,
  RouterProvider,
  createRootRoute,
  createRoute,
  createMemoryHistory,
} from "@tanstack/react-router";

const QID = "20260519T211408Z_a1b2c3d4";

const DETAIL_RESPONSE = {
  query_id: QID,
  meta: {
    query: "GLFT vs Avellaneda-Stoikov spread derivation",
    ts_started: "2026-05-19T21:14:08.866354+00:00",
    config: { gen_model: "gemini-3.1-flash" },
    outcome: { crag_retried: false, synthesis_skipped: false, n_chunks_to_synth: 8 },
    totals: {
      cost_usd: 0.011,
      total_latency_ms: 3214,
      n_llm_calls: 3,
      tokens: { total: 18204 },
    },
  },
  report_md: "# Answer\nThe GLFT framework recovers the optimal spread.",
  stages: ["01_decompose.json", "03_milvus.json"],
  chunks: ["01_a3f1.txt", "02_b8c2.txt"],
};

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function textResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/plain" },
  });
}

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

async function buildRouter() {
  const { QueryDetail } = await import("@/components/phase3/QueryDetail");
  const rootRoute = createRootRoute();
  const detailRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/app/queries/$queryId",
    component: () => <QueryDetail />,
  });
  const routeTree = rootRoute.addChildren([detailRoute]);
  const history = createMemoryHistory({ initialEntries: [`/app/queries/${QID}`] });
  return createRouter({ routeTree, history });
}

async function renderDetail(fetchImpl: (url: string) => Response) {
  const spy = vi.fn((url: string) => Promise.resolve(fetchImpl(url)));
  vi.stubGlobal("fetch", spy);
  const router = await buildRouter();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return spy;
}

describe("QueryDetail — lazy stage + chunk fetch", () => {
  it("does NOT fetch stage payloads or chunk text on initial render", async () => {
    const spy = await renderDetail((url) => {
      if (url.includes(`/api/queries/${QID}/stages/`)) return jsonResponse({ payload: {} });
      if (url.includes(`/api/queries/${QID}/chunks/`)) return textResponse("chunk text");
      return jsonResponse(DETAIL_RESPONSE);
    });

    // Detail itself loads.
    expect(
      await screen.findByText("GLFT vs Avellaneda-Stoikov spread derivation"),
    ).toBeTruthy();

    // No stage/chunk endpoint hit yet (stage "01" auto-opens, so allow stage 01;
    // assert no CHUNK fetch fired and stage 03 not fetched).
    const urls = spy.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.includes("/chunks/"))).toBe(false);
    expect(urls.some((u) => u.includes("/stages/03"))).toBe(false);
  });

  it("fetches chunk text when a chunk row is expanded (QLOG-03)", async () => {
    const spy = await renderDetail((url) => {
      if (url.includes(`/api/queries/${QID}/stages/`)) return jsonResponse({ payload: {} });
      if (url.includes(`/api/queries/${QID}/chunks/01_a3f1.txt`))
        return textResponse("PARENT CHUNK BODY 42");
      if (url.includes(`/api/queries/${QID}/chunks/`)) return textResponse("other");
      return jsonResponse(DETAIL_RESPONSE);
    });

    const chunkRow = await screen.findByText("01_a3f1.txt");
    fireEvent.click(chunkRow);

    await waitFor(() => {
      const urls = spy.mock.calls.map((c) => String(c[0]));
      expect(urls.some((u) => u.includes("/chunks/01_a3f1.txt"))).toBe(true);
    });
    expect(await screen.findByText("PARENT CHUNK BODY 42")).toBeTruthy();
  });

  it("fetches stage payload when a stage row is expanded (QLOG-02)", async () => {
    const spy = await renderDetail((url) => {
      if (url.includes(`/api/queries/${QID}/stages/milvus`))
        return jsonResponse({ name: "milvus", payload: { hits: 7 } });
      if (url.includes(`/api/queries/${QID}/stages/`))
        return jsonResponse({ name: "decompose", payload: {} });
      if (url.includes(`/api/queries/${QID}/chunks/`)) return textResponse("c");
      return jsonResponse(DETAIL_RESPONSE);
    });

    // "milvus" appears twice (retriever tag + stage name); the stage-name node
    // lives in a .name div inside the accordion summary.
    await screen.findByText("decompose"); // detail rendered
    const milvusNodes = screen.getAllByText("milvus");
    const stageName = milvusNodes.find((n) => n.className.includes("name"));
    expect(stageName).toBeTruthy();
    fireEvent.click(stageName as HTMLElement);

    await waitFor(() => {
      const urls = spy.mock.calls.map((c) => String(c[0]));
      expect(urls.some((u) => u.includes("/stages/milvus"))).toBe(true);
    });
  });
});
