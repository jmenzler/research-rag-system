/**
 * MapsPanel — save/load/delete graph views + coverage CTA.
 *
 * Mounted in GraphRail (Maps tab) and receives the live graph state
 * as props so it can build a full snapshot without prop-drilling from
 * the page down through the store.
 *
 * Features:
 *   - Save: snapshot {nodes, edges, corpus_ids} → useCreateMapMutation
 *   - List: useMapsListQuery(collection) + load + delete
 *   - Load: useMapQuery(id) → parse snapshot → call onLoadSnapshot callback
 *     so GraphPage can restore overlay nodes/edges (INGEST-06 nodes intact)
 *   - Coverage CTA: useMapCoverageQuery(id) → "{in_corpus} of {total} in corpus
 *     — ingest remaining {remainder}?" → useIngestMutation(missing_ids)
 *
 * Styling: var(--p3-*) tokens only. No dangerouslySetInnerHTML.
 * jsdom-safe: no WebGL dependency.
 */
import { useEffect, useState, type ReactElement } from "react";
import { Trash2 } from "lucide-react";

import {
  useMapsListQuery,
  useMapQuery,
  useMapCoverageQuery,
  useCreateMapMutation,
  useDeleteMapMutation,
  useIngestMutation,
} from "@/queries/graph";
import type { GraphNode, GraphEdge } from "@/queries/graph";
import { useGraphStore } from "@/state/graphStore";

// ---------------------------------------------------------------------------
// Snapshot shape stored in the maps.snapshot JSON column
// ---------------------------------------------------------------------------

interface MapSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
  corpus_ids: number[];
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface MapsPanelProps {
  collection: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  onLoadSnapshot: (nodes: GraphNode[], edges: GraphEdge[]) => void;
}

// ---------------------------------------------------------------------------
// CoverageRow — shows coverage CTA for a loaded map
// ---------------------------------------------------------------------------

function CoverageRow({
  mapId,
  collection,
}: {
  mapId: string;
  collection: string;
}): ReactElement {
  const coverage = useMapCoverageQuery(mapId);
  const ingestMutation = useIngestMutation();
  const addInFlight = useGraphStore((s) => s.addInFlight);

  if (coverage.isPending) {
    return (
      <p className="text-[10px] text-[var(--p3-muted)]">Checking coverage…</p>
    );
  }

  if (coverage.isError || !coverage.data) {
    return (
      <p className="text-[10px] text-[var(--p3-err)]">Coverage unavailable</p>
    );
  }

  const { in_corpus, total, missing_ids } = coverage.data;
  const remainder = total - in_corpus;

  if (remainder <= 0) {
    return (
      <p className="text-[10px] text-[var(--p3-ok)]">
        All {total} in corpus
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <p className="text-[10px] text-[var(--p3-muted)]">
        {in_corpus} of {total} in corpus &mdash; ingest remaining {remainder}?
      </p>
      <button
        type="button"
        disabled={ingestMutation.isPending}
        onClick={() => {
          ingestMutation.mutate(
            { corpus_ids: missing_ids, collection },
            { onSuccess: () => addInFlight(missing_ids) },
          );
        }}
        className="self-start rounded px-2 py-0.5 text-[10px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          backgroundColor: "var(--p3-primary-soft)",
          border: "1px solid var(--p3-primary-edge, rgba(59,130,246,.32))",
          color: "var(--p3-fg)",
        }}
      >
        {ingestMutation.isPending
          ? "Ingesting…"
          : `Ingest remaining ${remainder}`}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MapsPanel
// ---------------------------------------------------------------------------

export function MapsPanel({
  collection,
  nodes,
  edges,
  onLoadSnapshot,
}: MapsPanelProps): ReactElement {
  const [mapName, setMapName] = useState("");
  const [loadingMapId, setLoadingMapId] = useState<string | null>(null);
  const [expandedCoverageId, setExpandedCoverageId] = useState<string | null>(null);

  const mapsListResult = useMapsListQuery(collection);
  const createMap = useCreateMapMutation();
  const deleteMap = useDeleteMapMutation();
  const loadedMapResult = useMapQuery(loadingMapId ?? "");

  const handleSave = (): void => {
    const name = mapName.trim() || `Map ${new Date().toLocaleTimeString()}`;
    const corpus_ids = nodes.map((n) => n.corpusId);
    const snapshot: MapSnapshot = { nodes, edges, corpus_ids };
    createMap.mutate(
      {
        name,
        collection,
        snapshot: JSON.stringify(snapshot),
      },
      {
        onSuccess: () => setMapName(""),
      },
    );
  };

  const handleLoad = (mapId: string): void => {
    setLoadingMapId(mapId);
  };

  // When the loadedMapResult resolves, restore graph state and clear loading id.
  // Runs in an effect (not the render body) so the parent state updates from
  // onLoadSnapshot land after commit — updating the parent mid-render throws
  // "Cannot update a component while rendering a different component".
  const loadedMap = loadedMapResult.data;
  useEffect(() => {
    if (loadingMapId === null || !loadedMap || loadedMap.id !== loadingMapId) {
      return;
    }
    try {
      const snapshot = JSON.parse(loadedMap.snapshot) as MapSnapshot;
      if (snapshot.nodes && snapshot.edges) {
        onLoadSnapshot(snapshot.nodes, snapshot.edges);
      }
    } catch {
      // Invalid snapshot JSON — ignore
    }
    setLoadingMapId(null);
  }, [loadingMapId, loadedMap, onLoadSnapshot]);

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* Save current view */}
      <div className="flex flex-col gap-1.5">
        <p className="text-[11px] font-medium text-[var(--p3-muted)] uppercase tracking-wider">
          Save view
        </p>
        <div className="flex gap-1.5">
          <input
            type="text"
            value={mapName}
            onChange={(e) => setMapName(e.target.value)}
            placeholder="Map name…"
            maxLength={80}
            className="flex-1 rounded border px-2 py-1 text-xs outline-none focus:ring-1"
            style={{
              backgroundColor: "var(--p3-surface)",
              borderColor: "var(--p3-border)",
              color: "var(--p3-fg)",
            }}
            aria-label="Map name"
          />
          <button
            type="button"
            onClick={handleSave}
            disabled={createMap.isPending || nodes.length === 0}
            className="rounded border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
            style={{
              backgroundColor: "var(--p3-primary-soft)",
              borderColor: "var(--p3-primary-edge, rgba(59,130,246,.32))",
              color: "var(--p3-fg)",
            }}
          >
            Save
          </button>
        </div>
        {nodes.length === 0 && (
          <p className="text-[10px] text-[var(--p3-muted)]">
            Load a graph first to save a map.
          </p>
        )}
      </div>

      {/* Saved maps list */}
      <div className="flex flex-col gap-1">
        <p className="text-[11px] font-medium text-[var(--p3-muted)] uppercase tracking-wider">
          Saved maps
        </p>

        {mapsListResult.isPending && (
          <p className="text-xs text-[var(--p3-muted)]">Loading…</p>
        )}
        {mapsListResult.isError && (
          <p className="text-xs text-[var(--p3-err)]">Failed to load maps</p>
        )}
        {mapsListResult.data && mapsListResult.data.length === 0 && (
          <div className="empty">
            <h3>No maps saved</h3>
            <p>Save the current node set to revisit it later.</p>
          </div>
        )}

        {mapsListResult.data?.map((map) => (
          <div
            key={map.id}
            className="flex flex-col gap-1.5 rounded-md p-2"
            style={{
              backgroundColor: "var(--p3-surface)",
              border: "1px solid var(--p3-border)",
            }}
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="flex-1 truncate text-xs text-[var(--p3-fg)]" title={map.name}>
                {map.name}
              </span>
              <button
                type="button"
                onClick={() => handleLoad(map.id)}
                className="shrink-0 text-[10px] text-[var(--p3-primary-text)] hover:underline"
              >
                Load
              </button>
              <button
                type="button"
                onClick={() => {
                  if (expandedCoverageId === map.id) {
                    setExpandedCoverageId(null);
                  } else {
                    setExpandedCoverageId(map.id);
                  }
                }}
                className="shrink-0 text-[10px] text-[var(--p3-muted)] hover:text-[var(--p3-fg)]"
                title="Show coverage"
                aria-label="Toggle coverage"
              >
                Coverage
              </button>
              <button
                type="button"
                onClick={() => deleteMap.mutate(map.id)}
                disabled={deleteMap.isPending}
                className="shrink-0 text-[var(--p3-muted)] hover:text-[var(--p3-err)] disabled:opacity-40"
                aria-label={`Delete map ${map.name}`}
              >
                <Trash2 size={11} />
              </button>
            </div>

            {/* Coverage CTA — expanded on toggle */}
            {expandedCoverageId === map.id && (
              <CoverageRow mapId={map.id} collection={collection} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
