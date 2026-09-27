/**
 * D-07: Hover-card preview that renders inside the shadcn <HoverCardContent>.
 * Shows the first ~200 chars of the parent chunk text + source/page metadata
 * + optional score badges from the SSE citation payload.
 *
 * resolved=false (hallucinated marker) — chunkId is null → render an
 * explanatory empty state (D-09) instead of triggering the chunk fetch.
 */
import { type ReactElement } from "react";
import { Badge } from "@/components/ui/badge";
import { useChunkQuery } from "@/queries/papers";
import type { CitationPayload } from "@/lib/sse-client";

export interface CitationHoverCardProps {
  chunkId: string | null;
  citation?: CitationPayload;
}

export function CitationHoverCard(props: CitationHoverCardProps): ReactElement {
  // Always call the hook; passing null disables it (queries/chats.ts).
  const { data, isLoading, isError } = useChunkQuery(props.chunkId);

  if (props.chunkId === null) {
    return (
      <div data-testid="citation-hover-card" className="max-w-sm p-3 text-sm">
        <p className="text-muted-foreground italic">
          This citation could not be resolved — the model referenced a source
          that was not retrieved.
        </p>
      </div>
    );
  }

  return (
    <div
      data-testid="citation-hover-card"
      className="max-w-sm space-y-2 p-3 text-sm"
    >
      {isLoading && (
        <p className="text-muted-foreground">Loading chunk…</p>
      )}
      {isError && <p className="text-destructive">Failed to load chunk.</p>}
      {data && (
        <>
          <p className="font-mono text-xs text-muted-foreground">
            {data.source_file} (page {data.page_number})
          </p>
          <p className="line-clamp-4 text-foreground">
            {data.text.slice(0, 200)}
            {data.text.length > 200 ? "…" : ""}
          </p>
          {props.citation && (
            <div className="flex flex-wrap gap-1">
              {props.citation.score_dense !== null && (
                <Badge variant="secondary">
                  dense {props.citation.score_dense.toFixed(3)}
                </Badge>
              )}
              {props.citation.score_sparse !== null && (
                <Badge variant="secondary">
                  sparse {props.citation.score_sparse.toFixed(3)}
                </Badge>
              )}
              {props.citation.score_rerank !== null && (
                <Badge variant="secondary">
                  rerank {props.citation.score_rerank.toFixed(3)}
                </Badge>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
