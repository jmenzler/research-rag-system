/**
 * GraphRail — tabbed right-rail for the graph explorer.
 *
 * Two tabs:
 *   "Node detail" — renders NodeDetail for the currently selected node.
 *   "Queue"       — inert Phase-5 placeholder (visible but no live wiring).
 *
 * Supports collapse to a 36px icon column (UI-SPEC §8).
 */
import { useState, useCallback, type ReactElement } from "react";
import { PanelRight, PanelRightClose, Target, X } from "lucide-react";

import { useGraphStore } from "@/state/graphStore";
import { NodeDetail } from "@/components/graph/NodeDetail";
import { QueueJobRow } from "@/components/graph/QueueJobRow";
import { MapsPanel } from "@/components/graph/MapsPanel";
import { useQueueStream } from "@/hooks/useQueueStream";
import type { QueueJob } from "@/hooks/useQueueStream";
import { useIngestMutation } from "@/queries/graph";
import type { GraphNode, GraphEdge, InCorpusResult } from "@/queries/graph";

// ---------------------------------------------------------------------------
// GraphRailProps
// ---------------------------------------------------------------------------

export interface GraphRailProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  inCorpus: Map<number, InCorpusResult>;
  collection: string;
  onExpand: (corpusId: number) => void;
  onWalk: (corpusId: number) => void;
  onLoadSnapshot: (nodes: GraphNode[], edges: GraphEdge[]) => void;
}

// ---------------------------------------------------------------------------
// Tab types
// ---------------------------------------------------------------------------

type Tab = "detail" | "queue" | "maps";

// ---------------------------------------------------------------------------
// Status dot color for collapsed column
// ---------------------------------------------------------------------------

function statusDotColor(node: GraphNode | null, inCorpus: Map<number, InCorpusResult>): string {
  if (node === null) return "var(--p3-muted-2)";
  const entry = inCorpus.get(node.corpusId);
  if (entry?.inCorpus) return "#3B82F6";
  return "#6B7890";
}

// ---------------------------------------------------------------------------
// GraphRail
// ---------------------------------------------------------------------------

export function GraphRail({
  nodes,
  edges,
  inCorpus,
  collection,
  onExpand,
  onWalk,
  onLoadSnapshot,
}: GraphRailProps): ReactElement {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useGraphStore((s) => s.setSelectedNodeId);
  const [activeTab, setActiveTab] = useState<Tab>("detail");
  const [collapsed, setCollapsed] = useState(false);

  const ingestMutation = useIngestMutation();
  const { jobs } = useQueueStream({
    onRetry: useCallback(
      (job: QueueJob) => {
        if (job.corpus_ids && job.corpus_ids.length > 0) {
          ingestMutation.mutate({ corpus_ids: job.corpus_ids, collection });
        }
      },
      [ingestMutation, collection],
    ),
  });

  const jobList = Object.values(jobs);

  const selectedNode = selectedNodeId !== null
    ? nodes.find((n) => n.corpusId === selectedNodeId) ?? null
    : null;

  const selectedInCorpus = selectedNodeId !== null
    ? inCorpus.get(selectedNodeId)
    : undefined;

  const tabs: Array<{ id: Tab; label: string }> = [
    { id: "detail", label: "Node detail" },
    { id: "queue", label: "Queue" },
    { id: "maps", label: "Maps" },
  ];

  // ---- Collapsed state: 36px icon column ----
  if (collapsed) {
    const dotColor = statusDotColor(selectedNode, inCorpus);
    return (
      <div
        className="flex shrink-0 flex-col items-center border-l pt-2 gap-3"
        style={{
          width: 36,
          background: "var(--p3-bg)",
          borderColor: "var(--p3-border)",
        }}
      >
        <button
          type="button"
          aria-label="Open node detail"
          onClick={() => setCollapsed(false)}
          className="flex items-center justify-center rounded"
          style={{
            width: 28,
            height: 28,
            color: "var(--p3-muted)",
          }}
        >
          <PanelRight size={14} aria-hidden="true" />
        </button>
        {/* Selected-node status dot */}
        <div
          className="rounded-full"
          style={{
            width: 8,
            height: 8,
            background: dotColor,
            flexShrink: 0,
          }}
          title={selectedNode !== null ? selectedNode.title : "No node selected"}
        />
      </div>
    );
  }

  // ---- Open state: 340px full rail ----
  return (
    <div
      className="flex shrink-0 flex-col border-l"
      style={{
        width: 340,
        background: "var(--p3-bg)",
        borderColor: "var(--p3-border)",
      }}
    >
      {/* Rail header */}
      <div
        className="flex h-9 shrink-0 items-center justify-between px-3"
        style={{ borderBottom: "1px solid var(--p3-border)" }}
      >
        <span
          className="text-[11px] font-medium uppercase tracking-wider"
          style={{ color: "var(--p3-muted)", letterSpacing: "0.08em" }}
        >
          Node Detail
        </span>
        <div className="flex items-center gap-1">
          {selectedNodeId !== null && (
            <button
              type="button"
              aria-label="Clear selection"
              onClick={() => setSelectedNodeId(null)}
              className="flex items-center justify-center rounded"
              style={{
                width: 24,
                height: 24,
                color: "var(--p3-muted)",
              }}
            >
              <X size={12} aria-hidden="true" />
            </button>
          )}
          <button
            type="button"
            aria-label="Collapse panel"
            onClick={() => setCollapsed(true)}
            className="flex items-center justify-center rounded"
            style={{
              width: 24,
              height: 24,
              color: "var(--p3-muted)",
            }}
          >
            <PanelRightClose size={13} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b" style={{ borderColor: "var(--p3-border)" }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={
              "flex-1 px-3 py-2 text-xs font-medium transition-colors " +
              (activeTab === tab.id
                ? "border-b-2 border-[var(--p3-primary)] text-[var(--p3-primary-text)]"
                : "text-[var(--p3-muted)] hover:text-[var(--p3-fg)]")
            }
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "detail" && (
          <>
            {selectedNode !== null ? (
              <NodeDetail
                node={selectedNode}
                inCorpus={selectedInCorpus}
                collection={collection}
                onExpand={onExpand}
                onWalk={onWalk}
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
                <div
                  className="flex items-center justify-center rounded-full"
                  style={{
                    width: 40,
                    height: 40,
                    background: "var(--p3-surface)",
                    border: "1px solid var(--p3-border)",
                  }}
                >
                  <Target size={16} style={{ color: "var(--p3-muted)" }} aria-hidden="true" />
                </div>
                <div>
                  <p
                    className="text-[13px] font-medium"
                    style={{ color: "var(--p3-fg)" }}
                  >
                    No node selected
                  </p>
                  <p
                    className="mt-1 text-xs leading-relaxed"
                    style={{ color: "var(--p3-muted)", maxWidth: 220 }}
                  >
                    Click any node on the canvas to see its provenance and actions.
                  </p>
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === "queue" && (
          <div className="flex flex-col gap-2 p-3">
            {jobList.length === 0 ? (
              <p className="py-8 text-center text-xs text-[var(--p3-muted)]">
                Queue idle
              </p>
            ) : (
              jobList.map((job) => (
                <QueueJobRow
                  key={job.run_id}
                  job={job}
                  onRetry={(j) => {
                    if (j.corpus_ids && j.corpus_ids.length > 0) {
                      ingestMutation.mutate({ corpus_ids: j.corpus_ids, collection });
                    }
                  }}
                />
              ))
            )}
          </div>
        )}

        {activeTab === "maps" && (
          <MapsPanel
            collection={collection}
            nodes={nodes}
            edges={edges}
            onLoadSnapshot={onLoadSnapshot}
          />
        )}
      </div>
    </div>
  );
}
