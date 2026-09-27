/**
 * /app/corpus — Phase 3 corpus list route. Nested under _layout so it
 * inherits the persistent sidebar grid. Owns its own zod search schema for
 * the corpus facet filters + cross-link scope (mirrors the FilterChipRow
 * useSearch/zod pattern; state lives in the URL, not Zustand).
 *
 * The `.p3` wrapper scopes the Phase 3 (Fira/desaturated-slate) token layer to
 * this surface only — chat keeps its shadcn-slate look (scope-then-retrofit).
 */
import { createFileRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { z } from "zod";

import { CorpusList } from "@/components/phase3/CorpusList";

const csvCollections = z
  .union([z.string(), z.array(z.string())])
  .optional()
  .transform((s): Array<"trading" | "ecology" | "notes" | "system"> => {
    const raw =
      s === undefined ? [] : Array.isArray(s) ? s : s.split(",").filter(Boolean);
    const allowed = ["trading", "ecology", "notes", "system"] as const;
    return raw.filter((c): c is (typeof allowed)[number] =>
      (allowed as readonly string[]).includes(c),
    );
  });

const corpusSearchSchema = z.object({
  q: z.string().optional().default(""),
  collections: csvCollections,
  yearFrom: z.string().optional().default(""),
  yearTo: z.string().optional().default(""),
  ingested: z.enum(["any", "7d", "30d", "1y"]).optional().default("any"),
  hasArxiv: z.coerce.boolean().optional().default(false),
  hasDoi: z.coerce.boolean().optional().default(false),
  sort: z.enum(["title", "year", "ingested"]).optional().default("ingested"),
  dir: z.enum(["asc", "desc"]).optional().default("desc"),
  scopeChatId: z.string().optional(),
  scopeChatTitle: z.string().optional(),
  /** Cursor-pagination token (opaque, from backend's encode_cursor). */
  cursor: z.string().optional(),
});

export const Route = createFileRoute("/app/_layout/corpus")({
  component: CorpusRoute,
  validateSearch: corpusSearchSchema,
});

function CorpusRoute(): ReactElement {
  return (
    <div className="p3">
      <CorpusList />
    </div>
  );
}
