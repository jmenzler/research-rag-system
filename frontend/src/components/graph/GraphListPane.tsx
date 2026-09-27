/**
 * GraphListPane — GRAPH-06 accessible list-view alternative to the WebGL canvas.
 *
 * Shows the FULL node set (not ceiling-capped) so above-ceiling nodes remain
 * reachable via keyboard/screen-reader (GRAPH-05, Pitfall 6).
 *
 * ARIA contract:
 *   container:  role="region" aria-label="Node list"
 *   inner list: role="listbox" aria-label="Citation graph nodes" tabIndex={0}
 *               aria-activedescendant pointing to the active option id
 *   each row:   role="option" aria-selected={corpus_id === selectedCorpusId}
 *               id="glp-option-{corpus_id}"
 *
 * Keyboard: ArrowDown/ArrowUp move the active option; Enter calls onSelect.
 * Selecting an above-ceiling node: onSelect fires; "Show on canvas" CTA sets
 * ceilingOverride via graphStore (caller wires this).
 */
import { useCallback, useMemo, useRef, useState, type KeyboardEvent, type ReactElement } from "react";

import { useGraphStore } from "@/state/graphStore";
import { NODE_COLORS } from "@/components/graph/GraphCanvas";

// ---------------------------------------------------------------------------
// Node shape accepted by GraphListPane.
// Uses snake_case to match the test fixture and the raw API shape; GraphPage
// adapts from its camelCase GraphNode[] via the adapter below.
// ---------------------------------------------------------------------------

export interface ListNode {
  corpus_id: number;
  title: string;
  in_corpus: boolean;
  year: number | null;
  citationcount: number | null;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface GraphListPaneProps {
  nodes: ListNode[];
  selectedCorpusId: number | null;
  onSelect: (corpusId: number) => void;
}

// ---------------------------------------------------------------------------
// ListDot — 10×10 status dot; dashed border for missing nodes
// ---------------------------------------------------------------------------

interface ListDotProps {
  inCorpus: boolean;
}

function ListDot({ inCorpus }: ListDotProps): ReactElement {
  const fillColor = inCorpus ? (NODE_COLORS["in"] ?? "#3B82F6") : "transparent";
  const strokeColor = inCorpus ? "#93C5FD" : "#6B7890";

  return (
    <svg
      width={10}
      height={10}
      viewBox="0 0 10 10"
      aria-hidden="true"
      style={{ flexShrink: 0, marginTop: 3 }}
    >
      <circle
        cx={5}
        cy={5}
        r={4}
        fill={fillColor}
        stroke={strokeColor}
        strokeWidth={1.4}
        strokeDasharray={inCorpus ? undefined : "2.5 2"}
      />
    </svg>
  );
}

function optionId(corpusId: number): string {
  return `glp-option-${corpusId}`;
}

// ---------------------------------------------------------------------------
// GraphListPane
// ---------------------------------------------------------------------------

export function GraphListPane({
  nodes,
  selectedCorpusId,
  onSelect,
}: GraphListPaneProps): ReactElement {
  const setCeilingOverride = useGraphStore((s) => s.setCeilingOverride);

  // Sort by citation count descending (UI-SPEC §10).
  const sortedNodes = useMemo(
    () =>
      [...nodes].sort(
        (a, b) => (b.citationcount ?? 0) - (a.citationcount ?? 0),
      ),
    [nodes],
  );

  const selectedNode = sortedNodes.find((n) => n.corpus_id === selectedCorpusId) ?? null;

  // Track the "focused" option index for keyboard navigation.
  const initialIndex = selectedCorpusId !== null
    ? Math.max(0, sortedNodes.findIndex((n) => n.corpus_id === selectedCorpusId))
    : 0;
  const [activeIndex, setActiveIndex] = useState(initialIndex);
  const listRef = useRef<HTMLDivElement>(null);

  const activeNode = sortedNodes[activeIndex] ?? null;
  const activeDescendant =
    activeNode !== null ? optionId(activeNode.corpus_id) : undefined;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLDivElement>): void => {
      if (sortedNodes.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((prev) => Math.min(prev + 1, sortedNodes.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (activeNode !== null) {
          onSelect(activeNode.corpus_id);
        }
      }
    },
    [sortedNodes, activeNode, onSelect],
  );

  return (
    <div role="region" aria-label="Node list" className="flex flex-col h-full">
      {/* Header */}
      <div
        className="flex h-9 shrink-0 items-center justify-between px-3"
        style={{ borderBottom: "1px solid var(--p3-border)" }}
      >
        <span
          className="text-[11px] font-medium uppercase tracking-wider"
          style={{ color: "var(--p3-muted)", letterSpacing: "0.08em" }}
        >
          Nodes
        </span>
        <span
          className="font-mono text-[11px] tabular-nums"
          style={{ color: "var(--p3-muted)" }}
        >
          {sortedNodes.length}
        </span>
      </div>

      {/* Listbox */}
      <div
        ref={listRef}
        role="listbox"
        aria-label="Citation graph nodes"
        tabIndex={0}
        aria-activedescendant={activeDescendant}
        onKeyDown={handleKeyDown}
        className="flex-1 overflow-y-auto outline-none focus-visible:ring-2 focus-visible:ring-[var(--p3-primary)] focus-visible:ring-inset"
      >
        {sortedNodes.length === 0 && (
          <p className="p-3 text-xs text-[var(--p3-muted)]">No nodes to display.</p>
        )}
        {sortedNodes.map((node, idx) => {
          const isSel = node.corpus_id === selectedCorpusId;
          const isActive = idx === activeIndex;

          return (
            <div
              key={node.corpus_id}
              id={optionId(node.corpus_id)}
              role="option"
              aria-selected={isSel}
              onClick={() => {
                setActiveIndex(idx);
                onSelect(node.corpus_id);
              }}
              onMouseEnter={() => setActiveIndex(idx)}
              className="cursor-pointer border-b last:border-0"
              style={{
                display: "grid",
                gridTemplateColumns: "12px 1fr auto",
                gap: "0 8px",
                alignItems: "start",
                padding: "6px 12px",
                borderColor: "var(--p3-border)",
                background: isSel
                  ? "var(--p3-primary-soft)"
                  : isActive
                    ? "var(--p3-bg-2)"
                    : undefined,
                boxShadow: isSel ? "inset 2px 0 0 var(--p3-primary)" : undefined,
                color: isSel
                  ? "var(--p3-primary)"
                  : isActive
                    ? "var(--p3-fg)"
                    : "var(--p3-fg-2)",
              }}
            >
              {/* Status dot */}
              <ListDot inCorpus={node.in_corpus} />

              {/* Label */}
              <p
                className="min-w-0 truncate text-[12px] leading-snug"
                style={{ paddingTop: 1 }}
              >
                {node.title}
              </p>

              {/* Citation count */}
              <p
                className="font-mono text-[10.5px] tabular-nums"
                style={{ color: "var(--p3-muted)", paddingTop: 1, whiteSpace: "nowrap" }}
              >
                {node.citationcount !== null
                  ? node.citationcount.toLocaleString()
                  : "—"}
              </p>
            </div>
          );
        })}
      </div>
      {/* Show on canvas CTA — outside the listbox so no nested-interactive (axe rule) */}
      {selectedNode !== null && !selectedNode.in_corpus && (
        <div className="border-t border-[var(--p3-border)] p-2">
          <button
            type="button"
            onClick={() => setCeilingOverride(true)}
            className="w-full rounded border border-[var(--p3-border)] px-2.5 py-1.5 text-xs text-[var(--p3-muted)] hover:text-[var(--p3-fg)] hover:bg-[var(--p3-bg-2)]"
          >
            Show on canvas
          </button>
        </div>
      )}
    </div>
  );
}
