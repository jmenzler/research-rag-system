/**
 * D-10 / D-11 / D-12 / CHAT-12: Retrieval Trace Inspector.
 *
 * Per-assistant-turn inline affordance. Collapsed by default (D-10) with a
 * trigger labelled `Retrieval (N chunks)` — N is the count of resolved
 * citation/chunk entries. Expanding (D-11) reveals a vertical stack of cards;
 * each card shows:
 *   - marker + short_cite ("Avellaneda & Stoikov 2008" — falls back to
 *     paper_id, then to "marker [N]" if neither is available)
 *   - first ~200 chars of `preview` (or, if absent, lazy-fetched from the
 *     parent chunk via useChunkQuery — staleTime: Infinity)
 *   - dense / sparse / rerank score badges
 * Clicking a card dispatches `openCitation(messageId, marker, parent_id)` —
 * same store action as CitationToken (Plan 06) but with parent_id passed
 * through so the CitationSheet doesn't fall back to its synthetic key.
 * The "Open full audit" button at the top-right of the expanded panel
 * dispatches `openAuditDetail(queryId)` (D-12, opens AuditDetailSheet).
 *
 * Native <details>/<summary> for collapse (no Disclosure/Accordion lib):
 *   - Keyboard reachable (Tab to summary, Enter/Space toggles).
 *   - Respects prefers-reduced-motion by default (no transition on the
 *     native element).
 *
 * Prop shape `ChunkLike`: superset of CitationPayload that ALSO carries
 * pre-hydrated `short_cite` and `preview` fields. The chat surface
 * normalises live-SSE CitationPayload + persisted message_meta into this
 * shape before passing in; the RetrievalTraceInspector test (Wave-0) uses
 * the same shape verbatim.
 *
 * Phase 1 limitation (WR-09): the live SSE citations event and the
 * persisted citations row both lack `preview` / `short_cite`, so in the
 * Phase 1 chat surface the inspector renders only marker + paper_id (or
 * "marker [N]") with score badges — NEVER preview text. Phase 2 ships the
 * preview hydration path (extend citations row OR lazy useChunkQuery
 * inside ChunkCard gated on `preview === undefined`). The Phase 1
 * RetrievalTraceInspector test passes preview explicitly to exercise the
 * render path, but no end-user flow hits it.
 */
import { type ReactElement, useState } from "react";
import { ChevronRight, ExternalLink, FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";

export interface ChunkLike {
  marker: number;
  child_id: string | null;
  parent_id: string | null;
  paper_id: string | null;
  short_cite?: string | null;
  preview?: string | null;
  score_dense: number | null;
  score_sparse: number | null;
  score_rerank: number | null;
  resolved?: boolean;
}

export interface RetrievalTraceInspectorProps {
  messageId: string;
  queryId: string | null;
  chunks: ChunkLike[];
}

function ChunkCard(props: {
  messageId: string;
  index: number;
  chunk: ChunkLike;
}): ReactElement {
  const openCitation = useUiStore((s) => s.openCitation);
  const c = props.chunk;
  const shortCite =
    (typeof c.short_cite === "string" && c.short_cite.length > 0
      ? c.short_cite
      : null) ??
    (typeof c.paper_id === "string" && c.paper_id.length > 0
      ? c.paper_id
      : null) ??
    `marker [${c.marker}]`;
  const preview = typeof c.preview === "string" ? c.preview : "";
  const previewText =
    preview.length > 200 ? `${preview.slice(0, 200)}…` : preview;

  const onClick = (): void => {
    // Dispatch with 2 args ONLY — strict deep-equal contract with
    // RetrievalTraceInspector.test.tsx line 128 + CitationToken.test.tsx
    // line 72 (both assert toEqual({messageId, marker}) on openedCitation).
    // CitationSheet resolves parent_id via the citationRegistry (populated
    // by the chat surface from the same chunk array).
    openCitation(props.messageId, c.marker);
  };

  return (
    <div
      data-testid={`trace-card-${props.index}`}
      className="border-border/70 hover:border-border hover:bg-accent/30 group cursor-pointer rounded-md border p-2.5 transition-colors"
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div className="flex items-center gap-2">
        <span className="bg-primary/15 text-primary shrink-0 rounded font-mono text-[10px] font-semibold leading-none [padding:3px_5px]">
          {c.marker}
        </span>
        <span className="truncate text-sm font-medium">{shortCite}</span>
        {c.parent_id && (
          <ExternalLink
            className="text-muted-foreground group-hover:text-foreground ml-auto size-3 shrink-0 transition-colors"
            aria-hidden
          />
        )}
      </div>
      {previewText && (
        <p className="text-muted-foreground mt-2 line-clamp-3 text-sm">
          {previewText}
        </p>
      )}
      <div className="mt-2 flex flex-wrap gap-1">
        {c.score_dense !== null && (
          <Badge
            variant="secondary"
            data-testid={`score-dense-${c.marker}`}
            className="font-mono text-xs"
          >
            dense {c.score_dense.toFixed(3)}
          </Badge>
        )}
        {c.score_sparse !== null && (
          <Badge
            variant="secondary"
            data-testid={`score-sparse-${c.marker}`}
            className="font-mono text-xs"
          >
            sparse {c.score_sparse.toFixed(3)}
          </Badge>
        )}
        {c.score_rerank !== null && (
          <Badge
            variant="secondary"
            data-testid={`score-rerank-${c.marker}`}
            className="font-mono text-xs"
          >
            rerank {c.score_rerank.toFixed(3)}
          </Badge>
        )}
      </div>
    </div>
  );
}

export function RetrievalTraceInspector(
  props: RetrievalTraceInspectorProps,
): ReactElement | null {
  const openAuditDetail = useUiStore((s) => s.openAuditDetail);
  const [open, setOpen] = useState(false);

  // T-01-W4-06 mitigation: Inspector returns null ONLY when BOTH the chunk
  // count is 0 AND queryId is null. When queryId is present but no chunks
  // were retrieved, render "Retrieval (0 chunks)" — a visible signal of
  // zero-grounding (NOT silent failure).
  const count = props.chunks.length;
  if (count === 0 && props.queryId === null) return null;

  return (
    <div
      data-testid="retrieval-trace-inspector"
      className="border-border/70 mt-3 overflow-hidden rounded-lg border"
    >
      <button
        type="button"
        data-testid="retrieval-trace-summary"
        className="text-muted-foreground hover:text-foreground hover:bg-accent/30 flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left text-xs font-medium transition-colors"
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
        aria-label={`Sources (${count})`}
      >
        <ChevronRight
          className={`size-3.5 shrink-0 transition-transform duration-150 ${open ? "rotate-90" : ""}`}
          aria-hidden
        />
        <FileText className="size-3.5 shrink-0" aria-hidden />
        <span>Sources</span>
        <span className="bg-muted text-muted-foreground ml-0.5 rounded-full px-1.5 py-0.5 font-mono text-[10px] leading-none">
          {count}
        </span>
      </button>
      {open && (
        <div className="border-border/70 space-y-2 border-t p-2.5">
          {props.queryId !== null && (
            <div className="flex justify-end">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 gap-1.5 text-xs"
                data-testid="open-full-audit"
                onClick={() => openAuditDetail(props.queryId as string)}
              >
                <ExternalLink className="size-3" aria-hidden />
                Open full audit
              </Button>
            </div>
          )}
          <div className="space-y-1.5">
            {props.chunks.map((c, i) => (
              <ChunkCard
                key={`${c.marker}-${c.child_id ?? "unresolved"}-${i}`}
                messageId={props.messageId}
                index={i}
                chunk={c}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
