/**
 * Pure label-derivation utilities for the graph explorer (D-03).
 *
 * No React, no sigma, no DOM — fully unit-testable in jsdom.
 */

/**
 * Derive a short display label for a graph node.
 *
 * Priority:
 *   1. shortCite verbatim (e.g. "Avellaneda 2008") — Tier-1 corpus papers only.
 *   2. Best-effort from title + year: extract first token as surname proxy.
 *      If first token length <= 2, fall back to title.slice(0, 12).
 *   3. Year only (when title is absent).
 *   4. "?" as last resort.
 */
export function deriveShortLabel(
  shortCite: string | null | undefined,
  title: string | null | undefined,
  year: number | null | undefined,
): string {
  if (shortCite) return shortCite;
  if (!year && !title) return "?";
  if (!title) return String(year ?? "?");
  const firstToken = title.split(/[\s,;]+/)[0] ?? "";
  const surname = firstToken.length > 2 ? firstToken : title.slice(0, 12);
  return year ? `${surname} ${year}` : surname;
}
