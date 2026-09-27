/**
 * D-13 / D-14a / CHAT-12: Audit Detail Sheet.
 *
 * Opens on `useUiStore.openedAuditQueryId` change. Wider variant (sm:max-w-2xl)
 * than CitationSheet. Reuses the markdown pipeline + sanitize SCHEMA from
 * Message.tsx (Plan 06) — single source of allowlist (T-01-W4-01: never
 * trust report.md as "safe because server-built"; LLM text bleeds into it).
 *
 * Sections rendered when audit data arrives:
 *   1. meta.totals summary (cost / latency / tokens via CostLatencyBadge
 *      formatters — same look as the per-turn badge)
 *   2. report.md rendered through sanitised markdown
 *   3. lazy stage drill-down (StageRow per `stages[i]`, fetch fires only on
 *      expand — T-01-W4-03 mitigation)
 *
 * Stage entry normalisation: the backend (src/server/api_queries.py line 83)
 * returns filenames as strings like "01_decompose.json". Tests stub the
 * structured shape `{name, prefix}`. AuditDetailSheet normalises both into
 * `{display, name}` where `display` is what the user sees on the row and
 * `name` is what gets passed to /api/queries/{qid}/stages/{name}.
 */
import { type ReactElement, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import { useUiStore } from "@/state/uiStore";
import { useQueryAuditQuery, useQueryStageQuery } from "@/queries/queries";
import { CITATION_SANITIZE_SCHEMA } from "@/components/chat/Message";
import { remarkCitations } from "@/lib/remark-citations";
import { CostLatencyBadge } from "@/components/chat/CostLatencyBadge";
import type { StageEntry } from "@/lib/api";

interface NormalisedStage {
  /** What we render on the row (e.g. "01_decompose"). */
  display: string;
  /** What we POST to /api/queries/{qid}/stages/{name} (e.g. "decompose"). */
  name: string;
}

/**
 * Normalise either a filename ("01_decompose.json") or a structured entry
 * ({name: "decompose", prefix: "01"}) into a uniform render shape.
 */
function normaliseStage(entry: StageEntry): NormalisedStage {
  if (typeof entry === "string") {
    // Strip trailing .json extension; "01_decompose" is the display form.
    const display = entry.replace(/\.json$/, "");
    // Strip optional leading two-digit prefix + underscore to get the name.
    const name = display.replace(/^[0-9]{2}_/, "");
    return { display, name };
  }
  const name = entry.name;
  const prefix = entry.prefix;
  const display =
    typeof prefix === "string" && prefix.length > 0
      ? `${prefix}_${name}`
      : name;
  return { display, name };
}

function StageRow(props: {
  queryId: string;
  stage: NormalisedStage;
}): ReactElement {
  const [open, setOpen] = useState(false);
  // T-01-W4-03 (lazy fetch): query only fires when the user expands this row.
  // Eager-fetching all 8 stage payloads on Sheet open is FORBIDDEN
  // (must_haves truth in 01-07-PLAN.md).
  const stage = useQueryStageQuery(props.queryId, props.stage.name, {
    enabled: open,
  });
  return (
    <div data-testid={`stage-row-${props.stage.name}`} className="py-1">
      <button
        type="button"
        className="text-muted-foreground hover:text-foreground cursor-pointer font-mono text-xs"
        onClick={() => setOpen((p) => !p)}
        aria-expanded={open}
      >
        {open ? "▾" : "▸"} {props.stage.display}
      </button>
      {open && (
        <pre className="bg-muted mt-2 max-h-64 overflow-auto rounded p-2 text-xs">
          {stage.isLoading && "Loading stage payload…"}
          {stage.isError && `Error: ${String(stage.error)}`}
          {stage.data !== undefined &&
            JSON.stringify((stage.data as { payload?: unknown } | unknown), null, 2)}
        </pre>
      )}
    </div>
  );
}

export function AuditDetailSheet(): ReactElement {
  const queryId = useUiStore((s) => s.openedAuditQueryId);
  const close = useUiStore((s) => s.closeAuditDetail);
  const audit = useQueryAuditQuery(queryId);

  return (
    <Sheet
      open={queryId !== null}
      onOpenChange={(open) => {
        if (!open) close();
      }}
    >
      <SheetContent
        side="right"
        className="sm:max-w-2xl"
        data-testid="audit-detail-sheet"
      >
        <SheetHeader>
          <SheetTitle>Audit trail</SheetTitle>
          <SheetDescription>
            {queryId ? (
              <code className="font-mono text-xs">{queryId}</code>
            ) : null}
          </SheetDescription>
        </SheetHeader>
        <div className="space-y-4 overflow-y-auto px-4 pb-4">
          {audit.isLoading && (
            <p className="text-muted-foreground text-sm">Loading audit…</p>
          )}
          {audit.isError && (
            <p className="text-destructive text-sm">
              Failed to load audit: {String(audit.error)}
            </p>
          )}
          {audit.data && (
            <>
              {/*
                meta.totals summary — reuse the CostLatencyBadge formatters by
                rendering the badge itself with queryId. Since `audit.data` is
                already in the cache, the badge's useQueryAuditQuery does NOT
                re-fetch.
              */}
              <CostLatencyBadge queryId={audit.data.query_id ?? queryId} />
              <Separator />
              {/* report.md through the SAME pipeline as Message.tsx (T-01-W4-01). */}
              <article
                className="prose dark:prose-invert max-w-none text-sm"
                data-testid="audit-report-md"
              >
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkCitations]}
                  rehypePlugins={[[rehypeSanitize, CITATION_SANITIZE_SCHEMA]]}
                >
                  {audit.data.report_md}
                </ReactMarkdown>
              </article>
              {audit.data.stages && audit.data.stages.length > 0 && (
                <>
                  <Separator />
                  <div>
                    <h3 className="mb-2 text-sm font-semibold">
                      Stage payloads
                    </h3>
                    <div className="space-y-1">
                      {audit.data.stages.map((entry, i) => {
                        const norm = normaliseStage(entry);
                        return (
                          <StageRow
                            key={`${i}-${norm.name}`}
                            queryId={audit.data!.query_id ?? queryId ?? ""}
                            stage={norm}
                          />
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
