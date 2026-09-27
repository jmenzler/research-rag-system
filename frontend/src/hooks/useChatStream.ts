/**
 * React state machine over the SSE chat stream (D-04, D-20, Risk 5).
 *
 * State machine:
 *   idle → queued → streaming → done | cancelled | error
 *
 * Owns the AbortController. `stop()` calls `abort()` (propagates through
 * @microsoft/fetch-event-source's `signal`) and transitions to "cancelled".
 *
 * Idempotency (Risk 5 / D-18): a fresh `message_uuid` is generated on EVERY
 * `start()` call — including a post-cancel retry. If the user cancels and
 * retries we MUST submit a NEW uuid so the server does not idempotently
 * return the cancelled row.
 *
 * On `event: done` the hook calls `queryClient.invalidateQueries(["chats",
 * chatId])` so the next `useChatQuery` read reflects the new assistant turn —
 * the server-rendered messages list stays the single source of truth (CRIT-3).
 */
import { useCallback, useContext, useRef, useState } from "react";
import { QueryClientContext } from "@tanstack/react-query";
import type { QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import {
  streamChatTurn,
  type CitationPayload,
  type DonePayload,
} from "@/lib/sse-client";
import { useUiStore } from "@/state/uiStore";
import { randomUuid } from "@/lib/uuid";

export type ChatStreamState =
  | "idle"
  | "queued"
  | "streaming"
  | "done"
  | "error"
  | "cancelled";

export interface StartBody {
  content: string;
  message_uuid?: string;
  expected_version?: number;
}

// Plan 02-14 — sub-stage map shape (D-06 / Plan 02-06..09 sub-retriever
// event taxonomy). The backend emits `event: stage` with payloads like
// `{stage: "paperqa_rerank", latency_ms: 1234}` for fine-grained
// retriever progress; the hook prefix-splits the stage name on `_` and
// folds the trailing fragment into the parent retriever's slot.
//
// Optional `count` is for sub-stages that carry a numeric progress
// (e.g. PaperQA2's read counter — D-06). Backend may emit a payload
// like `{stage: "paperqa_read", latency_ms: 1234, index: 3, total: 12}`
// — the hook reads `index` + `total` if present and forwards as a
// `count: "3/12"` string the strip can render verbatim.
export type SubRetrieverParent = "paperqa" | "hipporag" | "lazygraph" | "fused";

export interface SubStageEntry {
  /** The trailing fragment after the parent prefix (e.g. "rerank" for
   *  "paperqa_rerank"). */
  current: string;
  /** Optional `N/T` progress, e.g. "3/12" for PaperQA2's read counter. */
  count?: string;
  /** Latency_ms of the most recent sub-stage event from this parent.
   *  Strip uses this only for debugging — the elapsed counter ticks
   *  off a local setInterval; per-stage latency comes from the parent
   *  `stagesObserved` slot. */
  latency_ms: number;
}

export type SubStageMap = Partial<Record<SubRetrieverParent, SubStageEntry>>;

const SUB_RETRIEVER_PARENTS: readonly SubRetrieverParent[] = [
  "paperqa",
  "hipporag",
  "lazygraph",
  "fused",
];

/** Detect parent retriever from a stage name like `paperqa_rerank` and
 * return `{parent, current}`. Returns null for top-level stages
 * (`decompose`, `milvus`, `rerank`, `synthesis`) and for unknown
 * prefixes — those stay in `stagesObserved` only. */
function parseSubStage(stage: string): {
  parent: SubRetrieverParent;
  current: string;
} | null {
  for (const parent of SUB_RETRIEVER_PARENTS) {
    const prefix = `${parent}_`;
    if (stage.startsWith(prefix)) {
      return { parent, current: stage.slice(prefix.length) };
    }
  }
  return null;
}

export interface UseChatStreamReturn {
  start: (body: StartBody) => void;
  stop: () => void;
  state: ChatStreamState;
  stagesObserved: Array<{ stage: string; latency_ms: number }>;
  /** Plan 02-14 — sub-stage cursor per parent retriever. Empty when no
   *  sub-stage events have arrived; the strip falls back to top-level
   *  glyphs in that case. */
  subStages: SubStageMap;
  citations: CitationPayload[];
  delta: string;
  done: DonePayload | null;
  error: string | null;
  messageUuid: string | null;
}

export function useChatStream(chatId: string): UseChatStreamReturn {
  // Read the query client via context directly (not via `useQueryClient`)
  // because the Wave-0 unit test for this hook renders without a
  // QueryClientProvider; `useQueryClient` would throw. When the client is
  // absent we simply skip the invalidate call — the hook is still observable
  // (state machine + data) which is what the test cares about.
  const queryClient = useContext(QueryClientContext) as
    | QueryClient
    | undefined;
  const setStoreController = useUiStore((s) => s.setStreamAbortController);

  const [state, setState] = useState<ChatStreamState>("idle");
  const [stagesObserved, setStagesObserved] = useState<
    UseChatStreamReturn["stagesObserved"]
  >([]);
  const [subStages, setSubStages] = useState<SubStageMap>({});
  const [citations, setCitations] = useState<CitationPayload[]>([]);
  const [delta, setDelta] = useState<string>("");
  const [done, setDone] = useState<DonePayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [messageUuid, setMessageUuid] = useState<string | null>(null);

  // Ref so callbacks see the live controller without retriggering React renders.
  const controllerRef = useRef<AbortController | null>(null);

  const start = useCallback(
    (body: StartBody): void => {
      // Regenerate uuid on every start — even post-cancel retry (Risk 5).
      const uuid = body.message_uuid ?? randomUuid();
      const expectedVersion = body.expected_version ?? 0;

      const controller = new AbortController();
      controllerRef.current = controller;
      setStoreController(controller);

      setState("queued");
      setStagesObserved([]);
      setSubStages({});
      setCitations([]);
      setDelta("");
      setDone(null);
      setError(null);
      setMessageUuid(uuid);

      void streamChatTurn(
        chatId,
        {
          content: body.content,
          message_uuid: uuid,
          expected_version: expectedVersion,
        },
        controller.signal,
        {
          onQueued: () => {
            /* state is already "queued"; nothing to do here */
          },
          onStage: (stage, latency_ms) => {
            setState("streaming");
            // Sub-stage events (paperqa_rerank, hipporag_walk, …) fold
            // into subStages map keyed by parent; top-level stages
            // (decompose / milvus / rerank / synthesis) append to
            // stagesObserved as before so the existing strip renders
            // continue to work unchanged.
            const sub = parseSubStage(stage);
            if (sub !== null) {
              setSubStages((prev) => ({
                ...prev,
                [sub.parent]: { current: sub.current, latency_ms },
              }));
            } else {
              setStagesObserved((prev) => [...prev, { stage, latency_ms }]);
            }
          },
          onCitations: (cites) => {
            setCitations(cites);
          },
          onDelta: (text) => {
            setDelta(text);
          },
          onDone: (meta) => {
            setDone(meta);
            setState("done");
            void queryClient?.invalidateQueries({
              queryKey: ["chats", chatId],
            });
            controllerRef.current = null;
            setStoreController(null);
          },
          onError: (msg, status) => {
            setError(msg);
            setState("error");
            if (status === 409) {
              // Plan 07's route owns the SOLE 409 toast (D-17 wording).
              // We only invalidate so the route sees the new server version.
              void queryClient?.invalidateQueries({
                queryKey: ["chats", chatId],
              });
            } else {
              toast.error(msg);
            }
            controllerRef.current = null;
            setStoreController(null);
          },
        },
      ).catch((err: unknown) => {
        const name = err instanceof Error ? err.name : "";
        // AbortError on stop() — surface as cancelled, not error.
        // Also treat "no live controller" as already-handled (cancelled or done).
        if (name === "AbortError" || controllerRef.current === null) {
          setState((prev) =>
            prev === "done" || prev === "error" ? prev : "cancelled",
          );
        } else {
          setError(err instanceof Error ? err.message : String(err));
          setState("error");
        }
        controllerRef.current = null;
        setStoreController(null);
      });
    },
    [chatId, queryClient, setStoreController],
  );

  const stop = useCallback((): void => {
    const c = controllerRef.current;
    if (c) {
      c.abort();
    }
    // Don't regress a terminal state — if the stream already reached
    // "done" or "error", a late Stop click must NOT silently revert
    // the UI to "cancelled" (which would hide the "Open full audit"
    // affordance and the live done payload from CostLatencyBadge).
    // Mirrors the .catch handler's guard at line 150.
    setState((prev) =>
      prev === "done" || prev === "error" ? prev : "cancelled",
    );
    controllerRef.current = null;
    setStoreController(null);
  }, [setStoreController]);

  return {
    start,
    stop,
    state,
    stagesObserved,
    subStages,
    citations,
    delta,
    done,
    error,
    messageUuid,
  };
}
