/**
 * <ChatScopeBar> (Plan 02-13, D-01).
 *
 * Above-message-list bar showing the current chat's retriever badge +
 * collection count (or comma-list when ≤ 2 selected). Clicking the bar
 * opens NewChatDialog in edit-mode via useUiStore.setScopeDialogOpen("edit").
 *
 * Visually disabled while streaming (opacity-60 pointer-events-none) so
 * the user cannot change scope mid-turn — the in-flight pipeline is bound
 * to the retriever+collections that started it (D-03).
 *
 * Layout: sits inside the chat surface's max-w-3xl container, above the
 * message list, with the same horizontal padding as the messages so the
 * left edge of the badges aligns with the assistant turns.
 */
import { type ReactElement } from "react";
import { ChevronDown } from "lucide-react";

import { useUiStore } from "@/state/uiStore";
import type { RetrieverId } from "@/lib/api";
import type { CollectionId } from "@/lib/retrievers";
import { cn } from "@/lib/cn";

export interface ChatScopeBarProps {
  chatId: string;
  retriever: RetrieverId | string;
  collections: CollectionId[] | string[];
  streaming: boolean;
}

function formatCollections(cols: CollectionId[] | string[]): string {
  // Default of all 4 → "4 collections" reads cleaner than the comma list.
  if (cols.length >= 4) return `${cols.length} collections`;
  if (cols.length === 0) return "0 collections";
  return cols.join(", ");
}

export function ChatScopeBar(props: ChatScopeBarProps): ReactElement {
  const { retriever, collections, streaming } = props;
  const setScopeDialogOpen = useUiStore((s) => s.setScopeDialogOpen);

  return (
    <button
      type="button"
      data-slot="chat-scope-bar"
      onClick={() => setScopeDialogOpen("edit")}
      disabled={streaming}
      aria-label="edit chat scope"
      className={cn(
        "mb-4 flex items-center gap-2 self-start rounded-md border border-transparent px-2 py-1 text-left text-xs transition duration-150 hover:border-[var(--p3-border)] hover:bg-[var(--p3-surface)] motion-reduce:transition-none",
        streaming && "opacity-60 pointer-events-none",
      )}
    >
      <span
        className={cn(
          "inline-flex h-[22px] items-center gap-1.5 rounded-sm border px-2 font-mono text-[11px] font-medium",
          retrieverTagClass(retriever),
        )}
      >
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-current opacity-85"
        />
        {retriever}
      </span>
      <span className="text-[var(--p3-muted-2)]">·</span>
      <span className="font-mono text-[11px] text-[var(--p3-fg-2)]">
        {formatCollections(collections)}
      </span>
      {!streaming && (
        <ChevronDown className="ml-1 size-3 text-[var(--p3-muted)]" />
      )}
    </button>
  );
}

// Per-retriever tint, mirroring the .tag.r-* fills in phase3.css. Falls back
// to the neutral "unknown" treatment for retriever ids without a tint.
function retrieverTagClass(retriever: string): string {
  switch (retriever) {
    case "milvus":
      return "text-[var(--p3-tag-milvus-fg)] bg-[var(--p3-tag-milvus-bg)] border-[var(--p3-tag-milvus-edge)]";
    case "paperqa":
      return "text-[var(--p3-tag-paperqa-fg)] bg-[var(--p3-tag-paperqa-bg)] border-[var(--p3-tag-paperqa-edge)]";
    case "lightrag":
    case "hipporag":
      return "text-[var(--p3-tag-lightrag-fg)] bg-[var(--p3-tag-lightrag-bg)] border-[var(--p3-tag-lightrag-edge)]";
    case "graphrag":
    case "lazygraph":
      return "text-[var(--p3-tag-graphrag-fg)] bg-[var(--p3-tag-graphrag-bg)] border-[var(--p3-tag-graphrag-edge)]";
    case "fused":
      return "text-[var(--p3-tag-fused-fg)] bg-[var(--p3-tag-fused-bg)] border-[var(--p3-tag-fused-edge)]";
    default:
      return "text-[var(--p3-muted)] bg-transparent border-[var(--p3-border)]";
  }
}
