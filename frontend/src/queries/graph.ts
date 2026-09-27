/**
 * TanStack Query hooks and queryOptions for the Phase 4 graph explorer.
 *
 * Maps snake_case API response shapes onto camelCase component shapes.
 * Server state (nodes/edges/in-corpus status) stays in TanStack Query.
 * Transient UI state (selectedNodeId, listOpen, ceilingOverride) stays
 * in graphStore.ts (Zustand).
 */
import {
  keepPreviousData,
  queryOptions,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import type { UseMutationResult, UseQueryResult } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  CollectionInfoApi,
  CreateMapBody,
  ExpandRequest,
  GraphEdgeApi,
  GraphNodeApi,
  InCorpusResultApi,
  IngestRequestApi,
  IngestResponseApi,
  MapApi,
  MapCoverageApi,
  MapMetaApi,
  NodeDetailResponseApi,
  SeedSearchHitApi,
  WalkRequest,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Component-facing camelCase shapes
// ---------------------------------------------------------------------------

export interface GraphNode {
  corpusId: number;
  title: string;
  year: number | null;
  citationCount: number | null;
  abstract?: string | null;
  shortCite: string | null;
  isSeed: boolean;
}

export interface GraphEdge {
  fromId: number;
  toId: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SeedSearchHit {
  corpusId: number;
  title: string;
  arxivId: string | null;
  doi: string | null;
  confidence: number;
  source: string;
}

export interface CollectionInfo {
  name: string;
  hasGraphData: boolean;
}

export interface InCorpusResult {
  inCorpus: boolean;
  collection: string | null;
}

export interface NodeDetail {
  corpusId: number;
  title: string;
  year: number | null;
  citationCount: number | null;
  authors: string[] | null;
  abstract: string | null;
  arxivId: string | null;
  doi: string | null;
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

function toNodeDetail(r: NodeDetailResponseApi): NodeDetail {
  return {
    corpusId: r.corpus_id,
    title: r.title,
    year: r.year,
    citationCount: r.citation_count,
    authors: r.authors,
    abstract: r.abstract,
    arxivId: r.arxiv_id,
    doi: r.doi,
  };
}

function toGraphNode(n: GraphNodeApi): GraphNode {
  return {
    corpusId: n.corpus_id,
    title: n.title,
    year: n.year,
    citationCount: n.citation_count,
    abstract: n.abstract ?? null,
    shortCite: n.short_cite ?? null,
    isSeed: n.is_seed,
  };
}

function toGraphEdge(e: GraphEdgeApi): GraphEdge {
  return { fromId: e.from_id, toId: e.to_id };
}

function toSeedSearchHit(h: SeedSearchHitApi): SeedSearchHit {
  return {
    corpusId: h.corpus_id,
    title: h.title,
    arxivId: h.arxiv_id,
    doi: h.doi,
    confidence: h.confidence,
    source: h.source,
  };
}

function toCollectionInfo(c: CollectionInfoApi): CollectionInfo {
  return { name: c.name, hasGraphData: c.has_graph_data };
}

function toInCorpusResult(r: InCorpusResultApi): InCorpusResult {
  return { inCorpus: r.in_corpus, collection: r.collection };
}

// ---------------------------------------------------------------------------
// Query options + hooks
// ---------------------------------------------------------------------------

export const corpusViewQuery = (collection: string) =>
  queryOptions({
    queryKey: ["graph", "corpus-view", collection],
    queryFn: async (): Promise<GraphData> => {
      const raw = await api.graph.corpusView(collection);
      return {
        nodes: raw.nodes.map(toGraphNode),
        edges: raw.edges.map(toGraphEdge),
      };
    },
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  });

export const collectionsQuery = () =>
  queryOptions({
    queryKey: ["graph", "collections"],
    queryFn: async (): Promise<CollectionInfo[]> => {
      const raw = await api.graph.collections();
      return raw.map(toCollectionInfo);
    },
    staleTime: 60_000,
  });

export const seedSearchQuery = (q: string) =>
  queryOptions({
    queryKey: ["graph", "seed-search", q],
    queryFn: async (): Promise<SeedSearchHit> => {
      const raw = await api.graph.seedSearch(q);
      return toSeedSearchHit(raw);
    },
    staleTime: 0,
    enabled: q.length > 0,
  });

export const useInCorpusCheck = (corpusIds: number[]) =>
  queryOptions({
    queryKey: ["graph", "in-corpus", corpusIds],
    queryFn: async (): Promise<Map<number, InCorpusResult>> => {
      const raw = await api.graph.inCorpusCheck({ corpus_ids: corpusIds });
      const result = new Map<number, InCorpusResult>();
      for (const [key, val] of Object.entries(raw)) {
        result.set(Number(key), toInCorpusResult(val as InCorpusResultApi));
      }
      return result;
    },
    staleTime: 0,
    enabled: corpusIds.length > 0,
  });

export function useWalkMutation(): UseMutationResult<GraphData, Error, WalkRequest> {
  return useMutation({
    mutationFn: async (params: WalkRequest): Promise<GraphData> => {
      const raw = await api.graph.walk(params);
      return {
        nodes: raw.nodes.map(toGraphNode),
        edges: raw.edges.map(toGraphEdge),
      };
    },
  });
}

export function useExpandMutation(): UseMutationResult<GraphData, Error, ExpandRequest> {
  return useMutation({
    mutationFn: async (params: ExpandRequest): Promise<GraphData> => {
      const raw = await api.graph.expand(params);
      return {
        nodes: raw.nodes.map(toGraphNode),
        edges: raw.edges.map(toGraphEdge),
      };
    },
  });
}

export const nodeDetailQuery = (corpusId: number | null) =>
  queryOptions({
    queryKey: ["graph", "node-detail", corpusId],
    queryFn: async (): Promise<NodeDetail> => {
      const raw = await api.graph.node(corpusId!);
      return toNodeDetail(raw);
    },
    staleTime: 5 * 60_000,
    enabled: corpusId !== null,
  });

export function useCorpusViewQuery(collection: string): UseQueryResult<GraphData> {
  return useQuery(corpusViewQuery(collection));
}

// ---------------------------------------------------------------------------
// Ingest + Maps hooks
// ---------------------------------------------------------------------------

export function useIngestMutation(): UseMutationResult<IngestResponseApi, Error, IngestRequestApi> {
  return useMutation({
    mutationFn: (params: IngestRequestApi): Promise<IngestResponseApi> =>
      api.ingest.start(params),
  });
}

export function useMapsListQuery(collection: string): UseQueryResult<MapMetaApi[]> {
  return useQuery(
    queryOptions({
      queryKey: ["maps", collection],
      queryFn: (): Promise<MapMetaApi[]> => api.maps.list(collection),
      staleTime: 30_000,
      placeholderData: keepPreviousData,
    }),
  );
}

export function useMapQuery(id: string): UseQueryResult<MapApi> {
  return useQuery(
    queryOptions({
      queryKey: ["maps", id],
      queryFn: (): Promise<MapApi> => api.maps.get(id),
      staleTime: 30_000,
      enabled: id.length > 0,
    }),
  );
}

export function useMapCoverageQuery(id: string): UseQueryResult<MapCoverageApi> {
  return useQuery(
    queryOptions({
      queryKey: ["maps", id, "coverage"],
      queryFn: (): Promise<MapCoverageApi> => api.maps.coverage(id),
      staleTime: 0,
      enabled: id.length > 0,
    }),
  );
}

export function useCreateMapMutation(): UseMutationResult<{ id: string }, Error, CreateMapBody> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateMapBody): Promise<{ id: string }> => api.maps.create(body),
    onSuccess: (): void => {
      void queryClient.invalidateQueries({ queryKey: ["maps"] });
    },
  });
}

export function useDeleteMapMutation(): UseMutationResult<void, Error, string> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string): Promise<void> => api.maps.delete(id),
    onSuccess: (): void => {
      void queryClient.invalidateQueries({ queryKey: ["maps"] });
    },
  });
}
