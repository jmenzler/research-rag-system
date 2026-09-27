/**
 * Tests for TanStack Query hooks added in Plan 05-04 Task 1:
 * useIngestMutation, useMapsListQuery, useMapQuery, useMapCoverageQuery,
 * useCreateMapMutation, useDeleteMapMutation.
 *
 * Uses a real QueryClient wrapper and vi.fn() mocks on api.ingest / api.maps.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor, act } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock api module
vi.mock("@/lib/api", () => ({
  api: {
    ingest: {
      start: vi.fn(),
    },
    maps: {
      create: vi.fn(),
      list: vi.fn(),
      get: vi.fn(),
      coverage: vi.fn(),
      delete: vi.fn(),
    },
    graph: {
      collections: vi.fn(),
      corpusView: vi.fn(),
      seedSearch: vi.fn(),
      walk: vi.fn(),
      expand: vi.fn(),
      inCorpusCheck: vi.fn(),
      node: vi.fn(),
    },
  },
}));

// eslint-disable-next-line import/order
import { api } from "@/lib/api";
// eslint-disable-next-line import/order
import {
  useIngestMutation,
  useMapsListQuery,
  useMapQuery,
  useMapCoverageQuery,
  useCreateMapMutation,
  useDeleteMapMutation,
} from "@/queries/graph";

function makeWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  return { Wrapper, qc };
}

describe("useIngestMutation", () => {
  it("calls api.ingest.start with corpus_ids + collection", async () => {
    const { Wrapper } = makeWrapper();
    vi.mocked(api.ingest.start).mockResolvedValueOnce({
      run_ids: [{ run_id: "r1", corpus_ids: [1] }],
      n_enqueued: 1,
    });

    const { result } = renderHook(() => useIngestMutation(), { wrapper: Wrapper });
    await act(async () => {
      result.current.mutate({ corpus_ids: [1], collection: "trading" });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.ingest.start).toHaveBeenCalledWith({ corpus_ids: [1], collection: "trading" });
    expect(result.current.data).toEqual({
      run_ids: [{ run_id: "r1", corpus_ids: [1] }],
      n_enqueued: 1,
    });
  });
});

describe("useMapsListQuery", () => {
  it("fetches maps list for a collection", async () => {
    const { Wrapper } = makeWrapper();
    vi.mocked(api.maps.list).mockResolvedValueOnce([
      { id: "m1", name: "Map 1", collection: "trading", created_at: "", updated_at: "" },
    ]);

    const { result } = renderHook(() => useMapsListQuery("trading"), { wrapper: Wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.maps.list).toHaveBeenCalledWith("trading");
    expect(result.current.data).toHaveLength(1);
  });
});

describe("useMapQuery", () => {
  it("fetches a single map by id", async () => {
    const { Wrapper } = makeWrapper();
    const mapData = {
      id: "m1",
      name: "Map 1",
      collection: "trading",
      snapshot: "{}",
      created_at: "",
      updated_at: "",
    };
    vi.mocked(api.maps.get).mockResolvedValueOnce(mapData);

    const { result } = renderHook(() => useMapQuery("m1"), { wrapper: Wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.maps.get).toHaveBeenCalledWith("m1");
    expect(result.current.data).toEqual(mapData);
  });
});

describe("useMapCoverageQuery", () => {
  it("fetches coverage for a map", async () => {
    const { Wrapper } = makeWrapper();
    vi.mocked(api.maps.coverage).mockResolvedValueOnce({
      in_corpus: 3,
      total: 5,
      missing_ids: [10, 11],
    });

    const { result } = renderHook(() => useMapCoverageQuery("m1"), { wrapper: Wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.maps.coverage).toHaveBeenCalledWith("m1");
    expect(result.current.data?.in_corpus).toBe(3);
  });
});

describe("useCreateMapMutation", () => {
  it("calls api.maps.create", async () => {
    const { Wrapper } = makeWrapper();
    vi.mocked(api.maps.create).mockResolvedValueOnce({ id: "new-map" });

    const { result } = renderHook(() => useCreateMapMutation(), { wrapper: Wrapper });
    await act(async () => {
      result.current.mutate({ name: "Test", collection: "trading", snapshot: "{}" });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.maps.create).toHaveBeenCalledWith({
      name: "Test",
      collection: "trading",
      snapshot: "{}",
    });
  });
});

describe("useDeleteMapMutation", () => {
  it("calls api.maps.delete", async () => {
    const { Wrapper } = makeWrapper();
    vi.mocked(api.maps.delete).mockResolvedValueOnce(undefined);

    const { result } = renderHook(() => useDeleteMapMutation(), { wrapper: Wrapper });
    await act(async () => {
      result.current.mutate("m1");
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.maps.delete).toHaveBeenCalledWith("m1");
  });
});
