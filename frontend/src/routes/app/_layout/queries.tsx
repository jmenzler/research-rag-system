/**
 * /app/queries — Phase 3 query-log list route. Owns its zod search schema for
 * the forensic-log facet filters + the `scopePaperId` cross-link (rendered as
 * the amber-edged scope chip inline in the facet row). URL is the source of
 * truth.
 */
import { createFileRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { z } from "zod";

import { QueryList } from "@/components/phase3/QueryList";

const RETRIEVER_VALUES = [
  "milvus",
  "paperqa",
  "lightrag",
  "hipporag",
  "lazygraph",
  "graphrag",
  "fused",
] as const;
const OUTCOME_VALUES = ["ok", "retry", "failed", "skipped"] as const;

const csvRetrievers = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((s): Array<(typeof RETRIEVER_VALUES)[number]> => {
    const raw =
      s === undefined ? [] : Array.isArray(s) ? s : s.split(",").filter(Boolean);
    return raw.filter((c): c is (typeof RETRIEVER_VALUES)[number] =>
      (RETRIEVER_VALUES as readonly string[]).includes(c),
    );
  });

const csvOutcome = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((s): Array<(typeof OUTCOME_VALUES)[number]> => {
    const raw =
      s === undefined ? [] : Array.isArray(s) ? s : s.split(",").filter(Boolean);
    return raw.filter((c): c is (typeof OUTCOME_VALUES)[number] =>
      (OUTCOME_VALUES as readonly string[]).includes(c),
    );
  });

const queriesSearchSchema = z.object({
  retrievers: csvRetrievers,
  outcome: csvOutcome,
  costMin: z.string().optional().default(""),
  latencyMin: z.string().optional().default(""),
  sort: z.enum(["ts", "cost", "latencyMs"]).optional().default("ts"),
  dir: z.enum(["asc", "desc"]).optional().default("desc"),
  scopePaperId: z.string().optional(),
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/app/_layout/queries")({
  component: QueriesRoute,
  validateSearch: queriesSearchSchema,
});

function QueriesRoute(): ReactElement {
  return (
    <div className="p3">
      <QueryList />
    </div>
  );
}
