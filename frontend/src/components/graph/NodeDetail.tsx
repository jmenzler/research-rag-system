/**
 * NodeDetail panel — GRAPH-07.
 *
 * Shows full metadata for the selected node: status swatch, title, authors,
 * year, citations, abstract (or empty state), S2 link, and actions.
 * Authors + abstract are fetched on-click via nodeDetailQuery (GET
 * /api/graph/node/{corpus_id} — 04-02 endpoint). The base GraphNode from
 * the walk may not carry them. Never fires during a walk (GRAPH-08 preserved
 * because nodeDetailQuery is only enabled when a node is selected).
 *
 * Threat: T-04-08 — all text rendered as JSX text nodes; no dangerouslySetInnerHTML.
 * T-04-11 — ingest button disabled and wired to no endpoint.
 * T-04-12 — S2 link uses rel="noopener noreferrer".
 */
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import type { ReactElement } from "react";

import { nodeDetailQuery, useIngestMutation } from "@/queries/graph";
import type { GraphNode, InCorpusResult } from "@/queries/graph";
import { NODE_COLORS } from "@/components/graph/GraphCanvas";
import { deriveShortLabel } from "@/lib/graphLabels";
import { useGraphStore } from "@/state/graphStore";

// ---------------------------------------------------------------------------
// Router-safe useNavigate wrapper (no throw in unit tests)
// ---------------------------------------------------------------------------

function useNavigateSafe(): ReturnType<typeof useNavigate> | null {
  try {
    return useNavigate();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function nodeStatus(
  node: GraphNode,
  inCorpus: InCorpusResult | undefined,
): "seed" | "in" | "in-other" | "missing" {
  if (node.isSeed) return "seed";
  if (!inCorpus?.inCorpus) return "missing";
  return inCorpus.collection === null || inCorpus.collection === "trading"
    ? "in"
    : "in-other";
}

const STATUS_LABELS: Record<string, string> = {
  seed: "Seed",
  in: "In corpus",
  "in-other": "In corpus (other)",
  missing: "Missing",
};

// ---------------------------------------------------------------------------
// NodeDetailProps
// ---------------------------------------------------------------------------

export interface NodeDetailProps {
  node: GraphNode;
  inCorpus: InCorpusResult | undefined;
  collection: string;
  onExpand: (corpusId: number) => void;
  onWalk: (corpusId: number) => void;
}

// ---------------------------------------------------------------------------
// NodeDetail
// ---------------------------------------------------------------------------

export function NodeDetail({
  node,
  inCorpus,
  collection,
  onExpand,
  onWalk,
}: NodeDetailProps): ReactElement {
  const navigate = useNavigateSafe();
  const ingestMutation = useIngestMutation();
  const addInFlight = useGraphStore((s) => s.addInFlight);
  const removeInFlight = useGraphStore((s) => s.removeInFlight);
  const isInFlight = useGraphStore((s) => s.inFlightCorpusIds.has(node.corpusId));

  const detailResult = useQuery(nodeDetailQuery(node.corpusId));
  const detail = detailResult.data;

  const status = nodeStatus(node, inCorpus);
  const statusColor = NODE_COLORS[status] ?? "#475569";

  const authorsText =
    detail?.authors && detail.authors.length > 0
      ? detail.authors.join(", ")
      : "—";

  const abstractText = detail?.abstract ?? null;

  const s2Url = `https://www.semanticscholar.org/paper/CorpusID:${node.corpusId}`;

  const handleSendToChat = (): void => {
    if (navigate === null) return;
    void navigate({ to: "/app/chat" } as unknown as never);
  };

  const handleOpenInCorpus = (): void => {
    if (navigate === null || !inCorpus?.inCorpus) return;
    void navigate({ to: "/app/corpus" } as unknown as never);
  };

  const handleAddToCorpus = (): void => {
    if (status !== "missing" || isInFlight || ingestMutation.isPending) return;
    // Mark in-flight optimistically so a rapid second click is blocked before the
    // POST returns (prevents the duplicate "1 processing + 1 queued" double-enqueue).
    addInFlight([node.corpusId]);
    ingestMutation.mutate(
      { corpus_ids: [node.corpusId], collection },
      { onError: () => removeInFlight([node.corpusId]) },
    );
  };

  return (
    <div className="flex flex-col gap-3 p-3 text-sm">
      {/* Status row */}
      <div className="flex items-center gap-2">
        <span
          className="inline-block size-2.5 rounded-full shrink-0"
          style={{ backgroundColor: statusColor }}
          aria-hidden
        />
        <span className="text-xs text-[var(--p3-muted)]">
          {STATUS_LABELS[status]}
          {inCorpus?.collection !== null && inCorpus?.collection !== undefined
            ? ` · ${inCorpus.collection}`
            : ""}
        </span>
      </div>

      {/* Title */}
      <div>
        <p className="font-medium leading-snug text-[var(--p3-fg)]">
          {node.title}
        </p>
        {/* Short cite / year secondary line (D-03) */}
        <p
          className="font-mono text-[11px] text-[var(--p3-muted)] mt-0.5"
        >
          {node.shortCite ?? deriveShortLabel(null, node.title, node.year ?? null)}
        </p>
      </div>

      {/* Metadata rows */}
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
        <dt className="text-[var(--p3-muted)]">Authors</dt>
        <dd className="text-[var(--p3-fg-2)]">{authorsText}</dd>

        <dt className="text-[var(--p3-muted)]">Year</dt>
        <dd className="font-mono text-[var(--p3-fg-2)]">
          {node.year !== null ? node.year : "—"}
        </dd>

        <dt className="text-[var(--p3-muted)]">Citations</dt>
        <dd className="font-mono text-[var(--p3-fg-2)]">
          {node.citationCount !== null
            ? node.citationCount.toLocaleString()
            : "—"}
        </dd>
      </dl>

      {/* Abstract */}
      <div>
        <p className="mb-1 text-xs text-[var(--p3-muted)]">Abstract</p>
        {detailResult.isPending ? (
          <p className="text-xs italic text-[var(--p3-muted)]">Loading…</p>
        ) : abstractText !== null ? (
          <p className="text-xs leading-relaxed text-[var(--p3-fg-2)] line-clamp-6">
            {abstractText}
          </p>
        ) : (
          <p className="text-xs italic text-[var(--p3-muted)]">
            No abstract available
          </p>
        )}
      </div>

      {/* S2 link */}
      <a
        href={s2Url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs text-[var(--p3-primary-text)] underline hover:no-underline"
      >
        CorpusID:{node.corpusId} on Semantic Scholar ↗
      </a>

      {/* Actions */}
      <div className="flex flex-col gap-1.5 pt-1 border-t border-[var(--p3-border)]">
        <button
          type="button"
          onClick={handleSendToChat}
          className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-fg-2)] hover:bg-[var(--p3-bg-2)] hover:text-[var(--p3-fg)] text-left"
        >
          Send to chat
        </button>

        <button
          type="button"
          onClick={handleOpenInCorpus}
          disabled={!inCorpus?.inCorpus}
          className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-fg-2)] hover:bg-[var(--p3-bg-2)] hover:text-[var(--p3-fg)] text-left disabled:cursor-not-allowed disabled:opacity-40"
        >
          Open in corpus
        </button>

        <button
          type="button"
          onClick={() => onExpand(node.corpusId)}
          className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-fg-2)] hover:bg-[var(--p3-bg-2)] hover:text-[var(--p3-fg)] text-left"
        >
          Expand neighbors
        </button>

        <button
          type="button"
          onClick={() => onWalk(node.corpusId)}
          className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-fg-2)] hover:bg-[var(--p3-bg-2)] hover:text-[var(--p3-fg)] text-left"
        >
          Walk from here
        </button>

        {/* Ingest affordance — live only for missing nodes (D-11).
            Other-collection violet nodes (in-other) get no ingest button. */}
        {status === "missing" && (
          <button
            type="button"
            onClick={handleAddToCorpus}
            disabled={ingestMutation.isPending || isInFlight}
            className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-fg-2)] hover:bg-[var(--p3-bg-2)] hover:text-[var(--p3-fg)] text-left disabled:cursor-not-allowed disabled:opacity-50"
          >
            {ingestMutation.isPending || isInFlight ? "Ingesting…" : "Add to corpus"}
          </button>
        )}
      </div>
    </div>
  );
}
