/**
 * /app/queries/$queryId — Phase 3 query-detail forensic route (backlink,
 * KPI strip, report.md, stage accordion + JSON viewer, retrieved chunks).
 */
import { createFileRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";

import { QueryDetail } from "@/components/phase3/QueryDetail";

export const Route = createFileRoute("/app/_layout/queries/$queryId")({
  component: QueryDetailRoute,
});

function QueryDetailRoute(): ReactElement {
  return (
    <div className="p3">
      <QueryDetail />
    </div>
  );
}
