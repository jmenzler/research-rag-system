/**
 * QueueJobRow — a single row in the Queue tab of GraphRail.
 *
 * Renders: amber arc swatch + truncated job title + coarse state label +
 * 4-stage strip when running (fetching→parsing→chunking→embedding, current
 * stage highlighted) + Retry button when crashed/cancelled.
 *
 * All styling uses var(--p3-*) tokens per PATTERNS.md.
 * No WebGL / canvas dependency — safe for jsdom tests.
 */
import type { ReactElement } from "react";
import type { QueueJob } from "@/hooks/useQueueStream";

// ---------------------------------------------------------------------------
// Stage strip
// ---------------------------------------------------------------------------

const STAGES = ["fetching", "parsing", "chunking", "embedding"] as const;
type Stage = (typeof STAGES)[number];

function stageLabel(s: string): string {
  switch (s) {
    case "fetching": return "Fetch";
    case "parsing": return "Parse";
    case "chunking": return "Chunk";
    case "embedding": return "Embed";
    default: return s;
  }
}

function isKnownStage(s: string | undefined): s is Stage {
  return STAGES.includes(s as Stage);
}

// ---------------------------------------------------------------------------
// State label
// ---------------------------------------------------------------------------

const STATE_LABELS: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  crashed: "Failed",
  cancelled: "Cancelled",
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface QueueJobRowProps {
  job: QueueJob;
  onRetry: (job: QueueJob) => void;
}

// ---------------------------------------------------------------------------
// QueueJobRow
// ---------------------------------------------------------------------------

export function QueueJobRow({ job, onRetry }: QueueJobRowProps): ReactElement {
  const isRunning = job.state === "running";
  const isTerminal = job.state === "crashed" || job.state === "cancelled";
  const currentStage = job.stage && isKnownStage(job.stage) ? job.stage : null;

  const jobTitle = job.corpus_ids && job.corpus_ids.length > 0
    ? `${job.corpus_ids.length} paper${job.corpus_ids.length === 1 ? "" : "s"} (${job.run_id.slice(0, 8)})`
    : job.run_id.slice(0, 12);

  return (
    <div
      className="flex flex-col gap-2 rounded-md px-2.5 py-2"
      style={{
        backgroundColor: "var(--p3-surface)",
        border: "1px solid var(--p3-border)",
      }}
      data-testid="queue-job-row"
    >
      {/* Row header: swatch + title + state label */}
      <div className="flex items-center gap-2 min-w-0">
        {/* Amber arc swatch */}
        <span
          className="shrink-0 inline-block size-2.5 rounded-full"
          style={{
            border: `2px solid ${
              job.state === "crashed"
                ? "var(--p3-err)"
                : job.state === "done"
                ? "var(--p3-ok)"
                : "var(--p3-amber, #F59E0B)"
            }`,
            backgroundColor: "transparent",
          }}
          aria-hidden
        />
        {/* Job title — truncated */}
        <span
          className="flex-1 truncate text-xs text-[var(--p3-fg-2)]"
          title={jobTitle}
        >
          {jobTitle}
        </span>
        {/* Coarse state label */}
        <span
          className="shrink-0 text-[10px] font-medium"
          style={{
            color:
              job.state === "crashed"
                ? "var(--p3-err)"
                : job.state === "done"
                ? "var(--p3-ok)"
                : "var(--p3-muted)",
          }}
        >
          {STATE_LABELS[job.state] ?? job.state}
        </span>
      </div>

      {/* 4-stage strip — only when running */}
      {isRunning && (
        <div className="flex gap-1" role="group" aria-label="Ingest stages">
          {STAGES.map((stage) => {
            const isCurrent = stage === currentStage;
            const isDone =
              currentStage !== null &&
              STAGES.indexOf(stage) < STAGES.indexOf(currentStage);
            return (
              <div
                key={stage}
                className="flex-1 rounded-sm px-1 py-0.5 text-center text-[9px] font-medium transition-colors"
                style={{
                  backgroundColor: isCurrent
                    ? "rgba(245,158,11,0.18)"
                    : isDone
                    ? "rgba(245,158,11,0.06)"
                    : "transparent",
                  border: `1px solid ${
                    isCurrent
                      ? "var(--p3-amber, #F59E0B)"
                      : "var(--p3-border)"
                  }`,
                  color: isCurrent
                    ? "var(--p3-amber, #F59E0B)"
                    : isDone
                    ? "var(--p3-muted)"
                    : "var(--p3-muted-2, #6B7890)",
                  fontFamily: "var(--p3-font-mono)",
                }}
                aria-current={isCurrent ? "step" : undefined}
              >
                {stageLabel(stage)}
              </div>
            );
          })}
        </div>
      )}

      {/* Indeterminate spinner when running with unknown stage */}
      {isRunning && currentStage === null && (
        <div className="text-[10px] text-[var(--p3-muted)]" aria-live="polite">
          Processing…
        </div>
      )}

      {/* Failure note */}
      {job.state === "crashed" && job.note && (
        <p className="text-[10px] text-[var(--p3-err)] leading-snug line-clamp-2">
          {job.note}
        </p>
      )}

      {/* Retry button for terminal states */}
      {isTerminal && (
        <button
          type="button"
          onClick={() => onRetry(job)}
          className="self-start rounded border px-2 py-0.5 text-[10px] font-medium transition-colors hover:opacity-80"
          style={{
            borderColor: "var(--p3-border)",
            color: "var(--p3-fg-2)",
            backgroundColor: "transparent",
          }}
        >
          Retry
        </button>
      )}
    </div>
  );
}
