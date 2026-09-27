/**
 * /app/graph — Phase 4 graph explorer route. Nested under _layout so it
 * inherits the persistent sidebar grid. Validates walk params from the URL
 * search string (zod coercion + enum bounds). Lazy-loads GraphPage so that
 * sigma + graphology stay isolated in the graph-viz Vite chunk (D-08).
 */
import { createFileRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { lazy, Suspense } from "react";
import { z } from "zod";

const GraphPage = lazy(() => import("@/components/graph/GraphPage"));

const graphSearchSchema = z.object({
  node: z.coerce.number().optional(),
  collection: z.string().default("trading"),
  depth: z.coerce.number().min(1).max(5).default(2),
  direction: z.enum(["references", "citers", "both"]).default("both"),
  yearFrom: z.coerce.number().optional(),
  yearTo: z.coerce.number().optional(),
  minCites: z.coerce.number().default(0),
  listOpen: z.coerce.boolean().default(false),
});

export const Route = createFileRoute("/app/_layout/graph")({
  component: GraphRoute,
  validateSearch: graphSearchSchema,
});

function GraphSkeleton(): ReactElement {
  return (
    <div
      className="h-full w-full bg-[var(--p3-surface)] flex items-center justify-center"
      role="status"
      aria-label="Loading citation graph…"
    >
      <div
        className="w-80 h-80 rounded-full border-2 border-dashed animate-pulse motion-reduce:animate-none"
        style={{ borderColor: "var(--p3-border)" }}
      />
    </div>
  );
}

function GraphRoute(): ReactElement {
  return (
    <div className="p3 h-full">
      <Suspense fallback={<GraphSkeleton />}>
        <GraphPage />
      </Suspense>
    </div>
  );
}
