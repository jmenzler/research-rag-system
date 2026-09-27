/**
 * Inline progress strip (D-05, extended in Plan 02-14 with sub-stage
 * cursor + elapsed counter + error / cancelled copy).
 *
 * Phase 1 baseline: 4 top-level glyphs (decompose / milvus / rerank /
 * synthesis) keyed by ✓ ◐ · derived from `event: stage` events.
 *
 * Plan 02-14 additions:
 *   - Sub-stage cursor: when the live `subStages` map carries a parent
 *     retriever's current sub-stage (e.g. `subStages.paperqa = {current:
 *     "rerank"}`), render it inline via <SubStageBlock>. Only the
 *     active parent block is shown — strip width stays bounded.
 *   - Elapsed counter: when a parent stage exceeds 5s wallclock, the
 *     parent's glyph gets an inline `34s` counter. Updates every 1s
 *     via local setInterval (cleaned up on unmount — T-02-14-03
 *     mitigation).
 *   - Error state: `"Retrieval failed — try regenerate"` (UI-SPEC).
 *   - Cancelled state: `"Cancelled"` (UI-SPEC).
 *
 * Layout: same `font-mono text-xs text-muted-foreground` row, glyphs
 * remain inline, sub-stage cursor expands the row but stays on one
 * line (overflow-hidden / truncate handled by parent).
 */
import { useEffect, useRef, useState, type ReactElement } from "react";

import { SubStageBlock } from "@/components/chat/SubStageBlock";
import type { SubStageMap } from "@/hooks/useChatStream";

export interface StageProgressStripProps {
  stagesObserved: Array<{ stage: string; latency_ms: number }>;
  state: "idle" | "queued" | "streaming" | "done" | "error" | "cancelled";
  totalLatencyMs?: number;
  /** Plan 02-14 — sub-stage cursor map from useChatStream. Optional so
   *  callers that haven't migrated still render the Phase 1 top-level
   *  glyph row. */
  subStages?: SubStageMap;
}

const STAGE_ORDER = ["decompose", "milvus", "rerank", "synthesis"] as const;

// Plan 02-14 — elapsed counter threshold (UI-SPEC § States Required for
// `<StageProgressStrip>` State 4: "Streaming, elapsed > 5s on current
// parent stage — append elapsed counter `paperqa 34s ◐`").
const ELAPSED_THRESHOLD_MS = 5_000;

function glyphFor(observed: boolean, isCurrent: boolean): string {
  if (observed) return "✓";
  if (isCurrent) return "◐";
  return "·";
}

export function StageProgressStrip(
  props: StageProgressStripProps,
): ReactElement | null {
  const { state, stagesObserved, totalLatencyMs, subStages } = props;

  // Plan 02-14: elapsed counter ticks once a second while a parent
  // stage exceeds 5s. We anchor the wallclock from FIRST RENDER under
  // the current `currentIdx`; switching parent stages resets the
  // anchor. This is UX-only — actual latency is in `meta.totals` on
  // the server (T-02-14-04 accepted disposition).
  const [now, setNow] = useState<number>(() => Date.now());
  const stageAnchorRef = useRef<{ key: string; startedAt: number } | null>(
    null,
  );

  // Identify the "current" parent stage so the elapsed counter has a key.
  const observedNames = new Set(stagesObserved.map((s) => s.stage));
  const currentIdx = STAGE_ORDER.findIndex((s) => !observedNames.has(s));
  const currentStage =
    currentIdx >= 0 ? STAGE_ORDER[currentIdx] : null;

  // Re-anchor whenever the current parent stage changes. We use the
  // stage name itself as the anchor key so flipping from `decompose`
  // → `milvus` resets the elapsed counter cleanly.
  const anchorKey =
    state === "streaming" || state === "queued" ? (currentStage ?? "") : "";

  useEffect(() => {
    if (anchorKey === "") {
      stageAnchorRef.current = null;
      return;
    }
    if (stageAnchorRef.current?.key !== anchorKey) {
      stageAnchorRef.current = { key: anchorKey, startedAt: Date.now() };
      setNow(Date.now());
    }
  }, [anchorKey]);

  // 1-Hz interval — only mounted during streaming so the timer is
  // cleaned up cleanly when the stream lands (T-02-14-03 mitigation).
  useEffect(() => {
    if (state !== "streaming" && state !== "queued") return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [state]);

  if (state === "idle") return null;

  if (state === "error") {
    return (
      <div
        data-testid="stage-progress-strip"
        className="text-muted-foreground text-xs"
      >
        Retrieval failed — try regenerate
      </div>
    );
  }

  if (state === "cancelled") {
    return (
      <div
        data-testid="stage-progress-strip"
        className="text-muted-foreground text-xs"
      >
        Cancelled
      </div>
    );
  }

  if (state === "done" && totalLatencyMs !== undefined) {
    const seconds = (totalLatencyMs / 1000).toFixed(1);
    return (
      <div
        data-testid="stage-progress-strip"
        className="text-muted-foreground text-xs"
      >
        Retrieved in {seconds}s
      </div>
    );
  }

  const anchor = stageAnchorRef.current;
  const elapsedMs =
    anchor && anchor.key === currentStage ? now - anchor.startedAt : 0;
  const showElapsed = elapsedMs >= ELAPSED_THRESHOLD_MS;
  const elapsedLabel = `${Math.floor(elapsedMs / 1000)}s`;

  return (
    <div
      data-testid="stage-progress-strip"
      role="status"
      aria-live="polite"
      className="ml-12 mt-0.5 inline-flex items-center rounded-full border border-[var(--p3-border)] bg-[var(--p3-surface)] px-3.5 py-1.5 font-mono text-[11px] text-[var(--p3-muted)]"
    >
      {STAGE_ORDER.map((stage, i) => {
        const isCurrent = i === currentIdx;
        const isObserved = observedNames.has(stage);
        const glyph = glyphFor(isObserved, isCurrent);
        // Only the active parent gets the elapsed counter (UI-SPEC
        // State 4). Past stages already have their per-stage latency
        // captured in `stagesObserved`; future stages have not started.
        const parentLabel =
          isCurrent && showElapsed ? `${stage} ${elapsedLabel}` : stage;
        return (
          <span key={stage} className="mr-3">
            {parentLabel}{" "}
            <span className={isCurrent ? "text-foreground" : undefined}>
              {glyph}
            </span>
          </span>
        );
      })}
      {/* Plan 02-14 — sub-stage cursor for active parent retriever, if any. */}
      {subStages &&
        Object.entries(subStages).map(([parent, entry]) =>
          entry ? (
            <SubStageBlock
              key={parent}
              parent={parent as Parameters<typeof SubStageBlock>[0]["parent"]}
              entry={entry}
            />
          ) : null,
        )}
    </div>
  );
}
