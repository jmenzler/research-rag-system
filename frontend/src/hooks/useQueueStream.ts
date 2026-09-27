/**
 * EventSource-backed queue state hook.
 *
 * Connects to GET /api/queue/stream with native EventSource and dispatches the five
 * typed events emitted by api_ingest.py:
 *
 *   position   → job state "queued" with position + queue_depth
 *   running    → job state "running" with stage + elapsed_s; adds corpus_ids to in-flight
 *   done       → re-calls api.graph.inCorpusCheck (flip trigger, D-07); clears in-flight
 *   crashed    → marks job failed, surfaces toast.error + Retry action (D-08); clears in-flight
 *   cancelled  → marks job cancelled; clears in-flight
 *
 * Cleanup: es.close() on unmount.
 */
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { useGraphStore } from "@/state/graphStore";

export type QueueJobState = "queued" | "running" | "done" | "crashed" | "cancelled";

export interface QueueJob {
  run_id: string;
  state: QueueJobState;
  position?: number;
  queue_depth?: number;
  stage?: string;
  elapsed_s?: number;
  note?: string;
  corpus_ids?: number[];
  n_imported?: number;
}

export interface QueueCounts {
  running: number;
  queued: number;
}

export interface UseQueueStreamReturn {
  jobs: Record<string, QueueJob>;
  counts: QueueCounts;
  isConnected: boolean;
}

export interface UseQueueStreamOptions {
  onRetry?: (job: QueueJob) => void;
}

export function useQueueStream(options: UseQueueStreamOptions = {}): UseQueueStreamReturn {
  const [jobs, setJobs] = useState<Record<string, QueueJob>>({});
  const [isConnected, setIsConnected] = useState(false);
  const { onRetry } = options;

  const addInFlight = useGraphStore((s) => s.addInFlight);
  const removeInFlight = useGraphStore((s) => s.removeInFlight);

  // onRetry from callers (e.g. TanStack mutations) is a fresh reference every
  // render. Keep it in a ref so the EventSource mounts ONCE — listing it in the
  // effect deps recreated the stream every render, thrashing isConnected and
  // re-rendering the graph at screen-refresh rate (the white-flash bug).
  const onRetryRef = useRef(onRetry);
  onRetryRef.current = onRetry;

  // The stream re-emits a job's terminal state every poll (done/crashed/cancelled
  // sit in a retention window). Run the side effects — inCorpusCheck, removeInFlight,
  // the failure toast — exactly ONCE per run_id, else the toast respawns un-dismissibly
  // and inCorpusCheck refires each poll (a flash source).
  const terminalHandledRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (typeof EventSource === "undefined") return;
    const es = new EventSource("/api/queue/stream");

    es.onopen = (): void => {
      setIsConnected(true);
    };

    es.onerror = (): void => {
      setIsConnected(false);
    };

    es.addEventListener("position", (ev: MessageEvent): void => {
      const parsed = JSON.parse(ev.data as string) as {
        run_id: string;
        position: number;
        queue_depth: number;
      };
      setJobs((prev) => ({
        ...prev,
        [parsed.run_id]: {
          ...prev[parsed.run_id],
          run_id: parsed.run_id,
          state: "queued",
          position: parsed.position,
          queue_depth: parsed.queue_depth,
        },
      }));
    });

    es.addEventListener("running", (ev: MessageEvent): void => {
      const parsed = JSON.parse(ev.data as string) as {
        run_id: string;
        elapsed_s: number;
        stage: string;
        corpus_ids?: number[];
      };
      if (parsed.corpus_ids) {
        addInFlight(parsed.corpus_ids);
      }
      setJobs((prev) => {
        const existing = prev[parsed.run_id];
        const updated: QueueJob = {
          run_id: parsed.run_id,
          state: "running",
          stage: parsed.stage,
          elapsed_s: parsed.elapsed_s,
        };
        if (parsed.corpus_ids !== undefined) {
          updated.corpus_ids = parsed.corpus_ids;
        } else if (existing?.corpus_ids !== undefined) {
          updated.corpus_ids = existing.corpus_ids;
        }
        if (existing?.position !== undefined) updated.position = existing.position;
        if (existing?.queue_depth !== undefined) updated.queue_depth = existing.queue_depth;
        return { ...prev, [parsed.run_id]: updated };
      });
    });

    es.addEventListener("done", (ev: MessageEvent): void => {
      const parsed = JSON.parse(ev.data as string) as {
        run_id: string;
        corpus_ids: number[];
        n_imported: number;
      };
      if (!terminalHandledRef.current.has(parsed.run_id)) {
        terminalHandledRef.current.add(parsed.run_id);
        void api.graph.inCorpusCheck({ corpus_ids: parsed.corpus_ids });
        removeInFlight(parsed.corpus_ids);
      }
      setJobs((prev) => ({
        ...prev,
        [parsed.run_id]: {
          ...prev[parsed.run_id],
          run_id: parsed.run_id,
          state: "done",
          corpus_ids: parsed.corpus_ids,
          n_imported: parsed.n_imported,
        },
      }));
    });

    es.addEventListener("crashed", (ev: MessageEvent): void => {
      const parsed = JSON.parse(ev.data as string) as {
        run_id: string;
        note: string;
        corpus_ids: number[];
      };
      const crashedJob: QueueJob = {
        run_id: parsed.run_id,
        state: "crashed",
        note: parsed.note,
        corpus_ids: parsed.corpus_ids,
      };
      setJobs((prev) => ({
        ...prev,
        [parsed.run_id]: {
          run_id: parsed.run_id,
          state: "crashed",
          note: parsed.note,
          corpus_ids: parsed.corpus_ids,
        },
      }));
      if (!terminalHandledRef.current.has(parsed.run_id)) {
        terminalHandledRef.current.add(parsed.run_id);
        removeInFlight(parsed.corpus_ids);
        // Stable id → repeated crashed frames update one toast instead of stacking.
        // dismissible so the user can clear it; the Queue tab keeps the durable record.
        toast.error(`Ingest failed: ${parsed.note}`, {
          id: `ingest-fail-${parsed.run_id}`,
          action: {
            label: "Retry",
            onClick: () => onRetryRef.current?.(crashedJob),
          },
          duration: Infinity,
          dismissible: true,
        });
      }
    });

    es.addEventListener("cancelled", (ev: MessageEvent): void => {
      const parsed = JSON.parse(ev.data as string) as {
        run_id: string;
        corpus_ids: number[];
      };
      if (!terminalHandledRef.current.has(parsed.run_id)) {
        terminalHandledRef.current.add(parsed.run_id);
        removeInFlight(parsed.corpus_ids);
      }
      setJobs((prev) => ({
        ...prev,
        [parsed.run_id]: {
          ...prev[parsed.run_id],
          run_id: parsed.run_id,
          state: "cancelled",
          corpus_ids: parsed.corpus_ids,
        },
      }));
    });

    return (): void => {
      es.close();
      setIsConnected(false);
    };
  }, [addInFlight, removeInFlight]); // onRetry via ref; addInFlight/removeInFlight stable (zustand)

  const counts: QueueCounts = {
    running: Object.values(jobs).filter((j) => j.state === "running").length,
    queued: Object.values(jobs).filter((j) => j.state === "queued").length,
  };

  return { jobs, counts, isConnected };
}
