/**
 * SSE client wrapper around @microsoft/fetch-event-source.
 *
 * Phase 1 D-20 event taxonomy: queued -> stage* -> citations -> delta -> done | error.
 * Keepalive: server emits `:hb ...\n\n` comment frames every 15s (D-21).
 *
 * `openWhenHidden: true` keeps the stream open while the tab is backgrounded.
 *
 * `onerror` re-throws the error to disable @microsoft/fetch-event-source's
 * automatic reconnect on terminal failures. Reconnect / retry is the consuming
 * hook's responsibility (Plan 06 `useChatStream`).
 */
import { fetchEventSource } from "@microsoft/fetch-event-source";

export interface CitationPayload {
  marker: number;
  child_id: string | null;
  parent_id: string | null;
  paper_id: string | null;
  score_dense: number | null;
  score_sparse: number | null;
  score_rerank: number | null;
  resolved: boolean;
}

export interface StagePayload {
  stage: string;
  latency_ms: number;
}

export interface DonePayload {
  query_id: string | null;
  usage: {
    input?: number;
    output?: number;
    total?: number;
    reasoning?: number;
  };
  cost_usd: number | null;
  latency_ms: number;
  idempotent_replay?: boolean;
}

export interface StreamEvents {
  onQueued?: (data: { chat_id: string; message_uuid: string }) => void;
  onStage?: (stage: string, latencyMs: number) => void;
  onCitations?: (cites: CitationPayload[]) => void;
  onDelta?: (text: string) => void;
  onDone?: (meta: DonePayload) => void;
  onError?: (msg: string, status?: number) => void;
}

export interface PostMessageBody {
  content: string;
  message_uuid: string;
  expected_version: number;
}

export async function streamChatTurn(
  chatId: string,
  body: PostMessageBody,
  signal: AbortSignal,
  events: StreamEvents,
): Promise<void> {
  await fetchEventSource(`/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
    // Keep background tabs connected.
    openWhenHidden: true,
    onmessage(ev) {
      if (!ev.event) return;
      switch (ev.event) {
        case "queued":
          events.onQueued?.(JSON.parse(ev.data));
          break;
        case "stage": {
          const d = JSON.parse(ev.data) as StagePayload;
          events.onStage?.(d.stage, d.latency_ms);
          break;
        }
        case "citations":
          events.onCitations?.(JSON.parse(ev.data) as CitationPayload[]);
          break;
        case "delta": {
          const d = JSON.parse(ev.data) as { text: string };
          events.onDelta?.(d.text);
          break;
        }
        case "done":
          events.onDone?.(JSON.parse(ev.data) as DonePayload);
          break;
        case "error": {
          const d = JSON.parse(ev.data) as { message: string; status?: number };
          events.onError?.(d.message, d.status);
          break;
        }
        default:
          // unknown event — ignore (forward-compat with Phase 2+ additions)
          break;
      }
    },
    onerror(err) {
      const msg = err instanceof Error ? err.message : String(err);
      events.onError?.(msg);
      // Re-throw to disable auto-reconnect on terminal errors
      // (per @microsoft/fetch-event-source contract).
      throw err;
    },
  });
}
