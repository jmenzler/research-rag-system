/**
 * TanStack Query hooks for the Phase 3 corpus browser surfaces.
 *
 * Fetches the real /api/papers endpoints (03-02-PLAN: corpus backend) and maps
 * the snake_case API response shapes onto the camelCase component prop shapes
 * defined in phase3Fixtures.ts.  No fixture data path remains — all fixture
 * DATA imports are removed (types stay).
 *
 * keepPreviousData + staleTime: 0 — the UI keeps the prior result list
 * mounted while a new query fires, preventing flash-of-empty (CRIT-3).
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { UseQueryResult } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  PaperRowApi,
  PaperListApi,
  PaperDetailApi,
} from "@/lib/api";
import type {
  Collection,
  Paper,
  PaperDetail,
  PaperListResult,
  ProvenanceEntry,
} from "@/queries/phase3Fixtures";

export interface PaperListParams {
  q: string;
  collections: Collection[];
  yearFrom: string;
  yearTo: string;
  ingested: "any" | "7d" | "30d" | "1y";
  hasArxiv: boolean;
  hasDoi: boolean;
  sort: "title" | "year" | "ingested";
  dir: "asc" | "desc";
  /** Cross-link scope: narrow to papers cited in a given chat. */
  scopeChatId?: string | null;
  cursor?: string | null;
  limit?: number;
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

/** Parse backend authors (JSON array string or null) to a display string. */
function parseAuthors(raw: string | null): string {
  if (!raw) return "";
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.join(", ") : String(parsed);
  } catch {
    return raw;
  }
}

/** Format an epoch-seconds timestamp to "YYYY-MM-DD". */
function epochToDate(epoch: number | null): string {
  if (epoch == null) return "";
  return new Date(epoch * 1000).toISOString().slice(0, 10);
}

/** Map a PaperRowApi from the backend to the component's Paper shape. */
function toPaper(row: PaperRowApi): Paper {
  return {
    id: row.paper_id,
    title: row.title ?? "",
    authors: parseAuthors(row.authors),
    year: row.year ?? 0,
    collection: row.collection as Collection,
    chunks: row.chunk_count ?? 0,
    arxiv: row.arxiv_id,
    doi: row.doi,
    ingested: epochToDate(row.ingested_at),
    shortCite: row.short_cite ?? "",
    degradedChunks: row.chunk_count === null,
  };
}

/** Format the total label (em-dash when null, locale-formatted otherwise). */
function formatTotal(n: number | null): string {
  if (n === null) return "—";
  return n.toLocaleString();
}

/** Build the ProvenanceEntry[] for CorpusDetail from the detail response. */
function buildProvenance(row: PaperDetailApi): ProvenanceEntry[] {
  return [
    { k: "paper_id", v: row.paper_id, mono: true },
    { k: "short cite", v: row.short_cite ?? null },
    { k: "fetch source", v: row.fetch_source ?? null },
    {
      k: "arXiv",
      v: row.arxiv_id,
      mono: true,
      ...(row.arxiv_id ? {} : { degradedReason: "no arXiv id" }),
    },
    {
      k: "DOI",
      v: row.doi,
      mono: true,
      ...(row.doi ? {} : { degradedReason: "no DOI" }),
    },
    {
      k: "extraction quality",
      v: row.extraction_quality,
      kind: "extraction-quality" as const,
    },
    { k: "parser version", v: row.parser_version ?? null, mono: true },
    { k: "collection", v: row.collection, kind: "coltag" as const },
    {
      k: "chunks in milvus",
      v: row.chunk_count === null ? null : String(row.chunk_count),
      kind: "chunks" as const,
      ...(row.chunk_count === null
        ? { degradedReason: "Milvus unreachable" }
        : {}),
    },
    {
      k: "ingested",
      v: epochToDate(row.ingested_at) || null,
      mono: true,
    },
    {
      k: "file hash",
      v: null,
      degradedReason: "pending backend",
      mono: true,
    },
  ];
}

/** Map nullable extraction_quality to the triple union the component expects. */
function toExtractionQuality(raw: string | null): "high" | "medium" | "low" {
  if (raw === "high") return "high";
  if (raw === "medium") return "medium";
  return "low";
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export function usePaperListQuery(
  params: PaperListParams,
): UseQueryResult<PaperListResult> {
  return useQuery({
    queryKey: ["papers", "list", params],
    queryFn: async (): Promise<PaperListResult> => {
      const sp = new URLSearchParams();
      if (params.q) sp.set("q", params.q);
      if (params.collections.length > 0) {
        sp.set("collection", params.collections.join(","));
      }
      if (params.yearFrom) sp.set("yearFrom", params.yearFrom);
      if (params.yearTo) sp.set("yearTo", params.yearTo);
      if (params.ingested && params.ingested !== "any") {
        sp.set("ingested", params.ingested);
      }
      if (params.hasArxiv) sp.set("hasArxiv", "1");
      if (params.hasDoi) sp.set("hasDoi", "1");
      if (params.sort) sp.set("sort", params.sort);
      if (params.dir) sp.set("dir", params.dir);
      if (params.scopeChatId) sp.set("citedInChat", params.scopeChatId);
      if (params.cursor) sp.set("cursor", params.cursor);
      if (params.limit) sp.set("limit", String(params.limit));

      const raw: PaperListApi = await api.papers.list(sp);
      return {
        papers: raw.papers.map(toPaper),
        totalLabel: formatTotal(raw.total),
        nextCursor: raw.next_cursor,
      };
    },
    placeholderData: keepPreviousData,
    staleTime: 0,
  });
}

export function usePaperDetailQuery(
  paperId: string | null,
): UseQueryResult<PaperDetail> {
  return useQuery({
    queryKey: ["papers", "detail", paperId],
    queryFn: async (): Promise<PaperDetail> => {
      const raw: PaperDetailApi = await api.papers.detail(paperId!);
      const base = toPaper(raw);
      return {
        ...base,
        provenance: buildProvenance(raw),
        queriesForPaper: [],
        chatsForPaper: [],
        extractionQuality: toExtractionQuality(raw.extraction_quality),
        parentChunks: 0,
        fileHash: "",
      };
    },
    enabled: paperId !== null,
    placeholderData: keepPreviousData,
    staleTime: 0,
  });
}
