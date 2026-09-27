// Re-export the canonical RetrieverId from api.ts so callers can import
// the type from one place. The two definitions stay in sync because the
// server's Literal[…] is the source of truth for both PATCH validation
// (api_chats.PatchChatRequest) and the availability map.
export type { RetrieverId } from "@/lib/api";
import type { RetrieverId } from "@/lib/api";

export const FUSED_POPOVER_BODY =
  "Combines available retriever results and reranks the merged candidates.";

export const STUB_UNAVAILABLE_HOVERCARD =
  "This retriever isn't installed on the server. Pick another.";

interface RetrieverMeta {
  id: RetrieverId;
  label: string;
  perfCostLabel: string;
}

export const RETRIEVERS: Record<RetrieverId, RetrieverMeta> = {
  milvus: {
    id: "milvus",
    label: "milvus",
    perfCostLabel: "milvus · hybrid retrieval",
  },
  paperqa: {
    id: "paperqa",
    label: "paperqa",
    perfCostLabel: "paperqa · optional adapter",
  },
  hipporag: {
    id: "hipporag",
    label: "hipporag",
    perfCostLabel: "hipporag · optional adapter",
  },
  lazygraph: {
    id: "lazygraph",
    label: "lazygraph",
    perfCostLabel: "lazygraph · optional adapter",
  },
  fused: {
    id: "fused",
    label: "fused",
    perfCostLabel: "fused · merged retrieval",
  },
};

export const RETRIEVER_ORDER: RetrieverId[] = [
  "milvus",
  "paperqa",
  "hipporag",
  "lazygraph",
  "fused",
];

export type CollectionId = "trading" | "ecology" | "notes" | "system";

export const COLLECTION_ORDER: CollectionId[] = [
  "trading",
  "ecology",
  "notes",
  "system",
];
