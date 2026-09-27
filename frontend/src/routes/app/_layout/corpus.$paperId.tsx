/**
 * /app/corpus/$paperId — Phase 3 corpus detail route (provenance trail,
 * queries-for-paper, chats-for-paper, cross-link launcher).
 */
import { createFileRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";

import { CorpusDetail } from "@/components/phase3/CorpusDetail";

export const Route = createFileRoute("/app/_layout/corpus/$paperId")({
  component: CorpusDetailRoute,
});

function CorpusDetailRoute(): ReactElement {
  return (
    <div className="p3">
      <CorpusDetail />
    </div>
  );
}
