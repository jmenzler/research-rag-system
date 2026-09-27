/**
 * <SubStageBlock> — inline rendering of a single retriever's sub-stage
 * cursor inside `<StageProgressStrip>` (Plan 02-14, D-06).
 *
 * Visual contract per UI-SPEC § Copywriting Contract / Stage-strip —
 * streaming:
 *     decompose ✓  paperqa: rerank ✓ read 3/12 ◐  rerank ·  synthesis ·
 *
 * The block above renders the `paperqa: rerank ✓ read 3/12 ◐` slice
 * for one parent retriever. The parent label (e.g. `paperqa`) is
 * rendered in the same mono/text-xs/muted-foreground style as the
 * surrounding strip; the active sub-stage glyph promotes to
 * `text-foreground` (UI-SPEC § Color: accent reserved for active
 * sub-stage glyph).
 *
 * Plan 02-14 keeps the rendering minimal: only the current sub-stage
 * is shown (no inline history), so the strip width stays bounded.
 */
import { type ReactElement } from "react";

import type { SubStageEntry } from "@/hooks/useChatStream";

export interface SubStageBlockProps {
  parent: "paperqa" | "hipporag" | "lazygraph" | "fused";
  entry: SubStageEntry;
}

export function SubStageBlock(props: SubStageBlockProps): ReactElement {
  const { parent, entry } = props;
  // Render `parent: current ◐` with optional `count` between current
  // and the glyph (count is a `N/T` fragment when the sub-stage carries
  // a counter — D-06 PaperQA2 read counter).
  return (
    <span data-testid={`substage-${parent}`} className="mr-3">
      {parent}: <span className="text-foreground">{entry.current}</span>
      {entry.count ? <> <span>{entry.count}</span></> : null} <span aria-hidden>◐</span>
    </span>
  );
}
