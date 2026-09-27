/**
 * Top-level graph explorer surface — three-mode orchestration:
 *
 *   Mode 1 corpus-view (default): corpusViewQuery for the active collection.
 *   Mode 3 seed search: user picks a seed hit → useWalkMutation seeds a walk.
 *   Mode 2 expand-from-node: onExpand(corpusId) → useExpandMutation merges
 *     new nodes/edges into the current set without replacing (GRAPH-09 pinning).
 *
 * After every node-set change exactly ONE batched useInCorpusCheck call runs
 * (GRAPH-04). Its result drives the inCorpus Map passed to GraphCanvas.
 *
 * Soft-ceiling "+N more" banner renders above the canvas when the node set
 * exceeds 2500 (D-09). graphStore.ceilingOverride escapes the cap.
 *
 * GraphRail and GraphListPane are placeholder slots filled in 04-05.
 */
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from "react";
import { Settings, Target } from "lucide-react";
import { toast } from "sonner";

import { Route } from "@/routes/app/_layout/graph";
import {
  corpusViewQuery,
  useInCorpusCheck,
  useWalkMutation,
  useExpandMutation,
  useIngestMutation,
} from "@/queries/graph";
import type { GraphNode, GraphEdge, SeedSearchHit } from "@/queries/graph";
import { GraphCanvas } from "@/components/graph/GraphCanvas";
import type { InCorpusEntry } from "@/components/graph/GraphCanvas";
import { GraphToolbar } from "@/components/graph/GraphToolbar";
import { GraphLegend } from "@/components/graph/GraphLegend";
import { GraphRail } from "@/components/graph/GraphRail";
import { GraphListPane } from "@/components/graph/GraphListPane";
import type { ListNode } from "@/components/graph/GraphListPane";
import { GraphMetadataHeader } from "@/components/graph/GraphMetadataHeader";
import { useShallow } from "zustand/react/shallow";

import { useGraphStore } from "@/state/graphStore";
import { useGraphDisplayStore } from "@/state/graphDisplayStore";
import { deriveShortLabel } from "@/lib/graphLabels";
import type { InCorpusResult } from "@/queries/graph";
import type { ExpandRequest, WalkRequest } from "@/lib/api";
import { GraphControlPanel } from "@/components/graph/GraphControlPanel";

// ---------------------------------------------------------------------------
// Router-safe hooks
// ---------------------------------------------------------------------------

type GraphSearch = {
  node?: number;
  collection: string;
  depth: number;
  direction: "references" | "citers" | "both";
  yearFrom?: number;
  yearTo?: number;
  minCites: number;
  listOpen: boolean;
};

function useSearchSafe(): Partial<GraphSearch> {
  try {
    return Route.useSearch() as Partial<GraphSearch>;
  } catch {
    return {};
  }
}

function useNavigateSafe(): ReturnType<typeof useNavigate> | null {
  try {
    return useNavigate();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Type mapping
// ---------------------------------------------------------------------------

function toInCorpusEntry(result: InCorpusResult): InCorpusEntry {
  return { inCorpus: result.inCorpus, collection: result.collection };
}

const SOFT_CEILING = 2500;

// ---------------------------------------------------------------------------
// GraphPage
// ---------------------------------------------------------------------------

export default function GraphPage(): ReactElement {
  const search = useSearchSafe();
  const navigate = useNavigateSafe();

  const collection = search.collection ?? "trading";
  const depth = search.depth ?? 2;
  const direction = search.direction ?? "both";
  const yearFrom = search.yearFrom;
  const yearTo = search.yearTo;
  const minCites = search.minCites ?? 0;

  const { selectedNodeId, setSelectedNodeId, focusedNodeId, listOpen, ceilingOverride, setCeilingOverride, addInFlight,
    overlayNodes, overlayEdges, overflowCount, setOverlayNodes, setOverlayEdges, setOverflowCount,
    setWalkSeedId, clearHoveredPath } =
    useGraphStore(
      useShallow((s) => ({
        selectedNodeId: s.selectedNodeId,
        setSelectedNodeId: s.setSelectedNodeId,
        focusedNodeId: s.focusedNodeId,
        listOpen: s.listOpen,
        ceilingOverride: s.ceilingOverride,
        setCeilingOverride: s.setCeilingOverride,
        addInFlight: s.addInFlight,
        overlayNodes: s.overlayNodes,
        overlayEdges: s.overlayEdges,
        overflowCount: s.overflowCount,
        setOverlayNodes: s.setOverlayNodes,
        setOverlayEdges: s.setOverlayEdges,
        setOverflowCount: s.setOverflowCount,
        setWalkSeedId: s.setWalkSeedId,
        clearHoveredPath: s.clearHoveredPath,
      })),
    );
  const displaySettings = useGraphDisplayStore();

  // Gear modal open state
  const [settingsOpen, setSettingsOpen] = useState(false);

  // D-01: track the expand-source node for centroid placement of new nodes.
  const [expandSourceId, setExpandSourceId] = useState<number | null>(null);

  // Multi-selection for bulk ingest (INGEST-02). shift/marquee wiring is HUMAN-UAT.
  const [selectedMissingIds, setSelectedMissingIds] = useState<Set<number>>(new Set());

  const ingestMutation = useIngestMutation();

  // Overlay node set (walk/expand results) lives in graphStore so it survives
  // tab unmount/remount — null means use corpus-view.

  // Mode 1: corpus-view query
  const corpusResult = useQuery(corpusViewQuery(collection));

  // Mode 3 + 2 mutations
  const walkMutation = useWalkMutation();
  const expandMutation = useExpandMutation();

  // Reset overlay ONLY on an actual collection change (D-03: collection switch →
  // corpus-view). The ref guard skips the mount run so a tab unmount/remount on
  // the same collection keeps the persisted overlay instead of wiping it.
  const overlayCollectionRef = useRef(collection);
  useEffect(() => {
    if (overlayCollectionRef.current !== collection) {
      overlayCollectionRef.current = collection;
      setOverlayNodes(null);
      setOverlayEdges(null);
      setOverflowCount(0);
      setWalkSeedId(null);
      clearHoveredPath();
    }
  }, [collection, setOverlayNodes, setOverlayEdges, setOverflowCount, setWalkSeedId, clearHoveredPath]);

  // Effective node/edge set: overlay (walk/expand) takes precedence over corpus-view.
  // Memoized so downstream useMemo/useCallback deps don't change on every render.
  const activeNodes = useMemo<GraphNode[]>(
    () => overlayNodes ?? corpusResult.data?.nodes ?? [],
    [overlayNodes, corpusResult.data?.nodes],
  );
  const activeEdges = useMemo<GraphEdge[]>(
    () => overlayEdges ?? corpusResult.data?.edges ?? [],
    [overlayEdges, corpusResult.data?.edges],
  );

  // Batched in-corpus-check (GRAPH-04): ONE query for ALL corpus_ids per render.
  const allCorpusIds = useMemo(
    () => activeNodes.map((n) => n.corpusId),
    [activeNodes],
  );
  const inCorpusResult = useQuery(useInCorpusCheck(allCorpusIds));

  // Build the inCorpus Map to pass to GraphCanvas.
  const inCorpusMap = useMemo(() => {
    const m = new Map<number, InCorpusEntry>();
    if (inCorpusResult.data) {
      for (const [cid, result] of inCorpusResult.data.entries()) {
        m.set(cid, toInCorpusEntry(result));
      }
    }
    return m;
  }, [inCorpusResult.data]);

  // Adapter: convert GraphNode[] (camelCase) to ListNode[] (snake_case) for GraphListPane.
  const listNodes = useMemo<ListNode[]>(
    () =>
      activeNodes.map((n) => ({
        corpus_id: n.corpusId,
        title: n.title,
        in_corpus: inCorpusResult.data?.get(n.corpusId)?.inCorpus ?? false,
        year: n.year,
        citationcount: n.citationCount,
      })),
    [activeNodes, inCorpusResult.data],
  );

  // Mode 3 seed walk: called when the toolbar's seed search selects a hit.
  const handleSeedSelect = useCallback(
    (hit: SeedSearchHit): void => {
      const params: WalkRequest = {
        corpus_ids: [hit.corpusId],
        depth,
        direction,
        year_from: yearFrom ?? null,
        year_to: yearTo ?? null,
        min_citations: minCites,
        budget: 500,
      };
      walkMutation.mutate(params, {
        onSuccess(data) {
          setOverlayNodes(data.nodes);
          setOverlayEdges(data.edges);
          const seed = data.nodes.find((n) => n.isSeed);
          setWalkSeedId(seed !== undefined ? String(seed.corpusId) : null);
          clearHoveredPath();
          toast.success(`Walk complete — ${data.nodes.length} nodes loaded`);
        },
        onError(err) {
          console.error("Walk mutation error:", err);
          toast.error("Walk failed: citation graph unavailable");
        },
      });
    },
    [depth, direction, yearFrom, yearTo, minCites, walkMutation, setWalkSeedId, clearHoveredPath],
  );

  // Mode 2 expand: called when a node's "Expand" action fires (wired from rail).
  const handleExpand = useCallback(
    (corpusId: number): void => {
      setExpandSourceId(corpusId);
      const params: ExpandRequest = {
        corpus_ids: [corpusId],
        direction,
        limit: 50,
        min_citations: minCites,
      };
      expandMutation.mutate(params, {
        onSuccess(data) {
          let newCount = 0;
          setOverlayNodes((prev) => {
            const base = prev ?? activeNodes;
            const existingIds = new Set(base.map((n) => n.corpusId));
            const newNodes = data.nodes.filter((n) => !existingIds.has(n.corpusId));
            newCount = newNodes.length;
            return [...base, ...newNodes];
          });
          setOverlayEdges((prev) => {
            const base = prev ?? activeEdges;
            const existingEdgeKeys = new Set(
              base.map((e) => `${e.fromId}->${e.toId}`),
            );
            const newEdges = data.edges.filter(
              (e) => !existingEdgeKeys.has(`${e.fromId}->${e.toId}`),
            );
            return [...base, ...newEdges];
          });
          toast.success(`Expanded — ${newCount} new nodes added`);
        },
        onError(err) {
          console.error("Expand mutation error:", err);
          toast.error("Expand failed: could not reach citation source");
        },
      });
    },
    [direction, minCites, expandMutation, activeNodes, activeEdges],
  );

  // List selection: syncs graphStore (no camera pan for list-only nodes — Pitfall 6).
  const handleListSelect = useCallback(
    (corpusId: number): void => {
      setSelectedNodeId(corpusId);
      if (navigate !== null) {
        navigate({
          search: ((prev: Partial<GraphSearch>) =>
            ({ ...prev, node: corpusId })) as unknown as never,
        });
      }
    },
    [setSelectedNodeId, navigate],
  );

  // Node selection: update graphStore + ?node= URL param.
  const handleNodeClick = useCallback(
    (corpusId: number): void => {
      setSelectedNodeId(corpusId);
      if (navigate !== null) {
        navigate({
          search: ((prev: Partial<GraphSearch>) =>
            ({ ...prev, node: corpusId })) as unknown as never,
        });
      }
    },
    [setSelectedNodeId, navigate],
  );

  // Stable so MapsPanel's snapshot-restore effect doesn't re-fire every render
  // (its deps include this callback).
  const handleLoadSnapshot = useCallback(
    (loadedNodes: GraphNode[], loadedEdges: GraphEdge[]): void => {
      setOverlayNodes(loadedNodes);
      setOverlayEdges(loadedEdges);
    },
    [setOverlayNodes, setOverlayEdges],
  );

  // Walk from a node: starts a fresh walk (used by NodeDetail "Walk from here").
  const handleWalk = useCallback(
    (corpusId: number): void => {
      const params: WalkRequest = {
        corpus_ids: [corpusId],
        depth,
        direction,
        year_from: yearFrom ?? null,
        year_to: yearTo ?? null,
        min_citations: minCites,
        budget: 500,
      };
      walkMutation.mutate(params, {
        onSuccess(data) {
          setOverlayNodes(data.nodes);
          setOverlayEdges(data.edges);
          const seed = data.nodes.find((n) => n.isSeed);
          setWalkSeedId(seed !== undefined ? String(seed.corpusId) : null);
          clearHoveredPath();
          toast.success(`Walk complete — ${data.nodes.length} nodes loaded`);
        },
        onError(err) {
          console.error("Walk mutation error:", err);
          toast.error("Walk failed: citation graph unavailable");
        },
      });
    },
    [depth, direction, yearFrom, yearTo, minCites, walkMutation, setWalkSeedId, clearHoveredPath],
  );

  // Restore selectedNodeId from URL ?node= on first mount.
  useEffect(() => {
    if (search.node !== undefined && selectedNodeId === null) {
      setSelectedNodeId(search.node);
    }
    // Only run on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Determine if the ceiling is exceeded (the canvas caps render internally;
  // we show the banner here based on untruncated activeNodes.length).
  const exceededCeiling =
    !ceilingOverride && activeNodes.length > SOFT_CEILING;
  const ceilingBannerCount = activeNodes.length - SOFT_CEILING;

  const isLoading = corpusResult.isLoading && overlayNodes === null;
  const isError = corpusResult.isError && overlayNodes === null;
  const isWalkPending = walkMutation.isPending;
  const isExpandPending = expandMutation.isPending;
  const isMutating = isWalkPending || isExpandPending;

  // Derive the selected node label for the metadata header breadcrumb.
  const selectedNode = selectedNodeId !== null
    ? activeNodes.find((n) => n.corpusId === selectedNodeId) ?? null
    : null;
  const selectedLabel = selectedNode !== null
    ? deriveShortLabel(selectedNode.shortCite, selectedNode.title, selectedNode.year)
    : null;

  // Corpus stats for the metadata header.
  const inCorpusCount = inCorpusResult.data
    ? Array.from(inCorpusResult.data.values()).filter((r) => r.inCorpus).length
    : 0;
  const totalCount = activeNodes.length;

  // No graph loaded: corpus-view returned empty and no overlay set.
  const hasNoGraph =
    !isLoading &&
    !isMutating &&
    overlayNodes === null &&
    (corpusResult.data?.nodes.length ?? 0) === 0;

  if (isLoading) {
    return (
      <div className="flex h-full w-full flex-col overflow-hidden">
        <GraphToolbar onSeedSelect={handleSeedSelect} />
        <GraphMetadataHeader
          collection={collection}
          inCorpus={0}
          total={0}
          depth={depth}
          direction={direction}
          selectedLabel={null}
        />
        <div
          className="flex flex-1 items-center justify-center"
          role="status"
          aria-label="Loading citation graph…"
        >
          <div
            className="w-80 h-80 rounded-full border-2 border-dashed"
            style={{
              borderColor: "var(--p3-border)",
              animation: "p3-pulse 1.6s ease-in-out infinite",
            }}
          />
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-[var(--p3-err)]">
        Failed to load graph data.
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col overflow-hidden">
      {/* Toolbar */}
      <GraphToolbar
        onSeedSelect={handleSeedSelect}
        onSettingsOpen={() => setSettingsOpen(true)}
      />

      {/* Metadata header strip */}
      <GraphMetadataHeader
        collection={collection}
        inCorpus={inCorpusCount}
        total={totalCount}
        depth={depth}
        direction={direction}
        selectedLabel={selectedLabel}
      />

      {/* Mutation loading indicator */}
      {isMutating && (
        <div className="flex items-center gap-2 border-b border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-3 py-1.5 text-xs text-[var(--p3-muted)]">
          <span style={{ animation: "p3-pulse 1.4s ease-in-out infinite" }}>Computing graph…</span>
        </div>
      )}

      {/* Soft-ceiling banner (D-09) */}
      {exceededCeiling && (
        <div className="flex items-center gap-3 border-b border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-3 py-1.5">
          <span className="text-xs text-[var(--p3-fg-2)]">
            +{ceilingBannerCount.toLocaleString()} more nodes — tighten year /{" "}
            min-citations / depth to narrow
          </span>
          <button
            type="button"
            onClick={() => setCeilingOverride(true)}
            className="ml-auto text-xs text-[var(--p3-primary-text)] underline hover:no-underline"
          >
            Show all
          </button>
        </div>
      )}

      {/* Main body: canvas + optional list pane + right rail */}
      <div className="relative flex flex-1 overflow-hidden">
        {/* List pane — shown when listOpen */}
        {listOpen && (
          <div className="w-72 shrink-0 border-r border-[var(--p3-border)] bg-[var(--p3-bg)] overflow-hidden flex flex-col">
            <GraphListPane
              nodes={listNodes}
              selectedCorpusId={selectedNodeId}
              onSelect={handleListSelect}
            />
          </div>
        )}

        {/* Graph canvas */}
        <div className="relative flex-1 overflow-hidden" data-testid="graph-canvas-region">
          {hasNoGraph ? (
            <div className="flex h-full items-center justify-center">
              <div className="empty">
                <span className="glyph">
                  <Target size={20} aria-hidden="true" />
                </span>
                <h3>No graph loaded</h3>
                <p>Search a seed paper to start a walk.</p>
              </div>
            </div>
          ) : isWalkPending ? (
            <div
              className="flex h-full items-center justify-center"
              role="status"
              aria-label="Loading citation graph…"
            >
              <div
                className="w-80 h-80 rounded-full border-2 border-dashed"
                style={{
                  borderColor: "var(--p3-border)",
                  animation: "p3-pulse 1.6s ease-in-out infinite",
                }}
              />
            </div>
          ) : (
            <GraphCanvas
              nodes={activeNodes}
              edges={activeEdges}
              inCorpus={inCorpusMap}
              activeCollection={collection}
              onNodeClick={handleNodeClick}
              onOverflow={setOverflowCount}
              selectedNodeId={selectedNodeId}
              focusedNodeId={focusedNodeId}
              displaySettings={displaySettings}
              sourceNodeId={expandSourceId}
            />
          )}
          {isExpandPending && (
            <div
              className="absolute bottom-3 right-3 rounded-md px-2 py-1 text-xs"
              role="status"
              aria-label="Expanding graph…"
              style={{
                backgroundColor: "var(--p3-surface)",
                border: "1px solid var(--p3-border)",
                color: "var(--p3-muted)",
                animation: "p3-pulse 1.4s ease-in-out infinite",
              }}
            >
              Expanding…
            </div>
          )}

          {/* Settings cog — top-right overlay, opens Display & Forces panel */}
          {!hasNoGraph && (
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              aria-label="Display and forces settings"
              className="absolute top-3 right-3 z-10 flex h-8 w-8 items-center justify-center rounded-md shadow-xl transition-colors hover:bg-white/5"
              style={{
                backgroundColor: "var(--p3-surface)",
                border: "1px solid var(--p3-border)",
                color: "var(--p3-muted)",
              }}
            >
              <Settings size={14} />
            </button>
          )}

          {/* Legend — bottom-right overlay (self-positioned) */}
          {!hasNoGraph && <GraphLegend />}
        </div>

        {/* Right rail */}
        <GraphRail
          nodes={activeNodes}
          edges={activeEdges}
          inCorpus={inCorpusResult.data ?? new Map()}
          collection={collection}
          onExpand={handleExpand}
          onWalk={handleWalk}
          onLoadSnapshot={handleLoadSnapshot}
        />

        {/* Bulk action bar — floating, appears when ≥1 missing nodes are selected (INGEST-02) */}
        {selectedMissingIds.size > 0 && (
          <div
            className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 rounded-lg px-4 py-2.5 shadow-2xl"
            style={{
              backgroundColor: "var(--p3-surface)",
              border: "1px solid var(--p3-border)",
            }}
            role="toolbar"
            aria-label="Bulk ingest actions"
          >
            <span className="text-xs text-[var(--p3-fg-2)]">
              {selectedMissingIds.size} paper{selectedMissingIds.size === 1 ? "" : "s"} selected
            </span>
            <button
              type="button"
              onClick={() => {
                const ids = Array.from(selectedMissingIds);
                ingestMutation.mutate(
                  { corpus_ids: ids, collection },
                  {
                    onSuccess: () => {
                      addInFlight(ids);
                      setSelectedMissingIds(new Set());
                    },
                  },
                );
              }}
              disabled={ingestMutation.isPending}
              className="rounded-md px-3 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              style={{
                backgroundColor: "var(--p3-primary-soft)",
                border: "1px solid var(--p3-primary-edge, rgba(59,130,246,.32))",
                color: "var(--p3-fg)",
              }}
            >
              Ingest {selectedMissingIds.size} paper{selectedMissingIds.size === 1 ? "" : "s"}
            </button>
            <button
              type="button"
              onClick={() => setSelectedMissingIds(new Set())}
              className="text-xs text-[var(--p3-muted)] hover:text-[var(--p3-fg)]"
              aria-label="Clear selection"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Overflow count forwarded for aria (accessible list in 04-05 can use this) */}
      {overflowCount > 0 && !exceededCeiling && (
        <div className="sr-only" aria-live="polite">
          {overflowCount} nodes hidden (soft ceiling reached). Use list view to see
          all.
        </div>
      )}

      {/* Display & Forces modal (D-07, D-08) — gear-triggered */}
      {settingsOpen && (
        <GraphControlPanel onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  );
}
