/**
 * Transport-layer tests for api.ingest.* and api.maps.* (Plan 05-04, Task 1).
 * Uses fetch mock to verify correct paths, methods, and payload shapes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock fetch before importing api so the module picks up the mock
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

// eslint-disable-next-line import/order
import { api } from "@/lib/api";

function mockOk(body: unknown): Response {
  return {
    ok: true,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response;
}

function mockNoContent(): Response {
  return {
    ok: true,
    status: 204,
    json: () => Promise.reject(new Error("no body")),
  } as unknown as Response;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("api.ingest.start", () => {
  it("POSTs to /api/ingest with corpus_ids + collection", async () => {
    const responseBody = {
      run_ids: [{ run_id: "abc", corpus_ids: [1, 2] }],
      n_enqueued: 2,
    };
    mockFetch.mockResolvedValueOnce(mockOk(responseBody));

    const result = await api.ingest.start({ corpus_ids: [1, 2], collection: "trading" });

    expect(mockFetch).toHaveBeenCalledWith(
      "/api/ingest",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ corpus_ids: [1, 2], collection: "trading" }),
      }),
    );
    expect(result).toEqual(responseBody);
  });

  it("returns run_ids array with run_id + corpus_ids", async () => {
    const responseBody = {
      run_ids: [
        { run_id: "run1", corpus_ids: [10] },
        { run_id: "run2", corpus_ids: [20, 30] },
      ],
      n_enqueued: 3,
    };
    mockFetch.mockResolvedValueOnce(mockOk(responseBody));

    const result = await api.ingest.start({ corpus_ids: [10, 20, 30], collection: "ecology" });
    expect(result.run_ids).toHaveLength(2);
    expect(result.n_enqueued).toBe(3);
  });
});

describe("api.maps.*", () => {
  it("create POSTs to /api/maps", async () => {
    mockFetch.mockResolvedValueOnce(mockOk({ id: "map-id-1" }));

    const result = await api.maps.create({ name: "My Map", collection: "trading", snapshot: "{}" });

    expect(mockFetch).toHaveBeenCalledWith(
      "/api/maps",
      expect.objectContaining({ method: "POST" }),
    );
    expect(result).toEqual({ id: "map-id-1" });
  });

  it("list GETs /api/maps?collection=... with encoded collection", async () => {
    mockFetch.mockResolvedValueOnce(mockOk([]));

    await api.maps.list("trading");

    const url = (mockFetch.mock.calls[0] as unknown[])[0] as string;
    expect(url).toContain("/api/maps");
    expect(url).toContain("collection=trading");
  });

  it("get GETs /api/maps/{id}", async () => {
    const mapBody = {
      id: "m1",
      name: "Test",
      collection: "trading",
      snapshot: "{}",
      created_at: "2026-01-01",
      updated_at: "2026-01-01",
    };
    mockFetch.mockResolvedValueOnce(mockOk(mapBody));

    const result = await api.maps.get("m1");

    const url = (mockFetch.mock.calls[0] as unknown[])[0] as string;
    expect(url).toBe("/api/maps/m1");
    expect(result).toEqual(mapBody);
  });

  it("coverage GETs /api/maps/{id}/coverage", async () => {
    mockFetch.mockResolvedValueOnce(mockOk({ in_corpus: 3, total: 5, missing_ids: [10, 11] }));

    const result = await api.maps.coverage("m1");

    const url = (mockFetch.mock.calls[0] as unknown[])[0] as string;
    expect(url).toBe("/api/maps/m1/coverage");
    expect(result).toEqual({ in_corpus: 3, total: 5, missing_ids: [10, 11] });
  });

  it("delete issues DELETE /api/maps/{id}", async () => {
    mockFetch.mockResolvedValueOnce(mockNoContent());

    await api.maps.delete("m1");

    expect(mockFetch).toHaveBeenCalledWith(
      "/api/maps/m1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
