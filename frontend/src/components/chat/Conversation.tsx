/**
 * Plan 07 surface composer — extended by Plan 02-14 with per-turn
 * `<MessageActions>` (D-19), per-turn retriever badge (D-02),
 * hover-pencil edit-last (D-17/D-18), and inline `<MessageEditor>` swap.
 *
 * Composition rules:
 *   - User turn:
 *       • Renders the bare <Message> + optional <HoverPencil> on the
 *         LAST user turn (props.isLastUserTurn === true).
 *       • If useUiStore.editingMessageId === this.message.id, swap the
 *         markdown body for <MessageEditor>; the rest of the row
 *         (timestamp, etc) stays.
 *   - Assistant turn:
 *       • <Message> + <AssistantTurnFooter> + <RetrievalTraceInspector>
 *         (unchanged from Plan 07) + new:
 *           - Per-turn retriever badge next to <CostLatencyBadge>
 *             (reads from `message.retriever` — Phase 2 column added
 *             by Plan 02-03's migration).
 *           - <MessageActions> trigger at the right end of the footer.
 *
 * Row-level group class: `group` wraps the user-turn row so
 * <HoverPencil>'s `group-hover:opacity-100` fades the pencil in.
 */
import { type ReactElement } from "react";

import { Message } from "@/components/chat/Message";
import { CostLatencyBadge } from "@/components/chat/CostLatencyBadge";
import { Badge } from "@/components/ui/badge";
import {
  RetrievalTraceInspector,
  type ChunkLike,
} from "@/components/chat/RetrievalTraceInspector";
import { MessageActions } from "@/components/chat/MessageActions";
import { HoverPencil } from "@/components/chat/HoverPencil";
import { MessageEditor } from "@/components/chat/MessageEditor";
import { useUiStore } from "@/state/uiStore";
import type { ChatMessage } from "@/lib/api";
import type { CitationPayload, DonePayload } from "@/lib/sse-client";

export interface ConversationProps {
  message: ChatMessage;
  citations: CitationPayload[];
  /** Pre-hydrated chunks (short_cite + preview if available). Falls back to
   *  citations[] when this is omitted. */
  chunks?: ChunkLike[];
  liveDone?: DonePayload | null;
  isLive?: boolean;
  /** Plan 02-14 — true ONLY for the last user turn in the merged list,
   *  and only when no stream is active. Drives <HoverPencil> mount. */
  isLastUserTurn?: boolean;
  /** Plan 02-14 — chat id needed by <MessageActions> + <MessageEditor>
   *  for the regenerate-last / edit-last POST. */
  chatId?: string;
  /** Plan 02-14 — chat-level optimistic-lock version, forwarded to
   *  <MessageEditor> so its POST body carries the correct
   *  expected_version. */
  expectedVersion?: number;
  /** Plan 02-14 — whether ANY stream is active in this chat (used by
   *  <MessageActions> to disable Regenerate, and by <HoverPencil> to
   *  hide the pencil mid-stream). */
  streaming?: boolean;
  /** Plan 02-14 — whether THIS specific turn is the in-flight assistant
   *  turn (hides the per-turn ⋯ menu trigger). */
  streamingThisTurn?: boolean;
  /** Plan 02-14 — fires the regenerate-last POST. Owned by the parent
   *  surface (Plan 02-14 wires this in ChatSurface). */
  onRegenerate?: () => void;
  /** Plan 02-14 — per-turn retriever provenance (D-02). Reads
   *  `message.retriever` if set; falls back to chat-level retriever
   *  passed via prop. */
  retriever?: string | null;
}

/**
 * Normalise SSE CitationPayload[] into the ChunkLike[] shape the inspector
 * accepts. Used as the fallback when the caller does not supply pre-hydrated
 * `chunks`. The preview / short_cite fields are left undefined — the inspector
 * degrades gracefully (falls back to paper_id / `marker [N]`).
 */
function citationsToChunks(citations: CitationPayload[]): ChunkLike[] {
  return citations.map((c) => ({
    marker: c.marker,
    child_id: c.child_id,
    parent_id: c.parent_id,
    paper_id: c.paper_id,
    score_dense: c.score_dense,
    score_sparse: c.score_sparse,
    score_rerank: c.score_rerank,
    resolved: c.resolved,
  }));
}

export function AssistantTurnFooter(props: ConversationProps): ReactElement {
  const retriever = props.retriever ?? null;
  return (
    <div className="mt-2 flex items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <CostLatencyBadge
          queryId={props.message.query_id}
          liveDone={props.isLive ? (props.liveDone ?? null) : null}
        />
        {/* Plan 02-14 — per-turn retriever badge (D-02). Reads `retriever`
            prop, falling back to `null` when the per-turn column is absent
            (older Phase 1 chats). The badge MUST live next to the cost +
            latency badge inside the footer row (UI-SPEC § Interaction
            Contracts / Per-turn ⋯ menu — order in row:
            `[Cost·Latency·Tokens]  [retriever]  [⋯]`). */}
        {retriever ? (
          <Badge
            variant="secondary"
            data-testid="retriever-badge"
            className="font-mono text-xs"
          >
            {retriever}
          </Badge>
        ) : null}
      </div>
      {/* Plan 02-14 — per-turn ⋯ menu (D-19). Mount on every assistant
          turn; the component itself decides whether to render based on
          `streamingThisTurn` (hides the trigger on the in-flight turn). */}
      {props.chatId ? (
        <MessageActions
          chatId={props.chatId}
          messageId={props.message.id}
          queryId={props.message.query_id}
          streaming={props.streaming ?? false}
          streamingThisTurn={props.streamingThisTurn ?? false}
          onRegenerate={props.onRegenerate}
          content={props.message.content}
          citations={props.citations.map((c) => ({
            marker: c.marker,
            paper_id: c.paper_id,
            // CitationPayload doesn't include short_cite / arxiv_id /
            // doi / local_pdf_path on the live SSE event (Phase 1
            // limitation noted in citationsToChunks above). Copy-as-
            // markdown falls back to "paper {paper_id}" — better than
            // nothing and matches the test contract for the empty
            // short_cite case.
          }))}
        />
      ) : null}
    </div>
  );
}

export function Conversation(props: ConversationProps): ReactElement {
  // Read the editing state at the top so user-turn rows can swap their
  // body for <MessageEditor>. Reading via the hook here (not inside the
  // user branch) keeps hooks order stable across the conditional.
  const editingMessageId = useUiStore((s) => s.editingMessageId);
  const isEditingThis = editingMessageId === props.message.id;

  if (props.message.role === "user") {
    const showPencil = props.isLastUserTurn === true;
    return (
      <div
        data-testid={`conversation-turn-${props.message.id}`}
        className="group relative"
      >
        {isEditingThis && props.chatId ? (
          <MessageEditor
            chatId={props.chatId}
            messageId={props.message.id}
            originalMessageUuid={props.message.message_uuid}
            initialContent={props.message.content}
            expectedVersion={props.expectedVersion ?? props.message.version}
          />
        ) : (
          <Message message={props.message} citations={[]} />
        )}
        {showPencil ? (
          <div className="absolute right-0 top-2">
            <HoverPencil
              messageId={props.message.id}
              isLastUserTurn={props.isLastUserTurn ?? false}
              streaming={props.streaming ?? false}
            />
          </div>
        ) : null}
      </div>
    );
  }

  const chunks = props.chunks ?? citationsToChunks(props.citations);
  return (
    <div data-testid={`conversation-turn-${props.message.id}`}>
      <Message message={props.message} citations={props.citations} />
      <AssistantTurnFooter {...props} />
      <RetrievalTraceInspector
        messageId={props.message.id}
        queryId={props.message.query_id}
        chunks={chunks}
      />
    </div>
  );
}
