/**
 * Copy-as-text and copy-as-markdown helpers (CHAT-09, D-24, D-25).
 *
 * Plan 02-14 splits the menu wiring (`<MessageActions>` opens a shadcn
 * `<DropdownMenu>` and calls the functions here) from the clipboard /
 * formatting logic (this file). Keeping the formatters in `lib/` means
 * the UI layer mounts thin wrappers and the test surface
 * (`CopyMenuItems.test.tsx`) imports the pure functions directly without
 * pulling React or shadcn primitives into the harness.
 *
 * Footnote pattern (D-25, locked in UI-SPEC § Copywriting Contract):
 *     [^N]: short_cite. URL
 *
 * URL preference (asserted by CopyMenuItems.test.tsx case "URL preference:
 * arxiv_id > doi > local PDF path"):
 *     1. arxiv_id (https://arxiv.org/abs/{id})
 *     2. doi (https://doi.org/{doi})
 *     3. local_pdf_path (verbatim — likely a file://-style abs path)
 *
 * Fallback when short_cite is empty (asserted by "fallback to 'paper
 * {paper_id}' on empty short_cite"): `paper {paper_id}`.
 *
 * Copy-as-text strips citation marker syntax `[1]`, `[1, 2]`, etc. AND
 * the remark-citations link form `[1](chunk://child)` (defensive — the
 * pre-render content shouldn't include these but content fed in via
 * `props.content` may have been intercepted before the remark plugin
 * runs; stripping both forms means the user never sees a stray `[1]`
 * in their pasted text).
 *
 * Toasts via Sonner (D-24 — success path emits the EXACT strings asserted
 * by the test:
 *     "Copied as text"  |  "Copied as markdown"
 * and the failure-path toast wording (sanitize.test.tsx + this file's
 * test do not assert the exact failure string, but UI-SPEC § Copywriting
 * Contract pins it):
 *     "Couldn't copy — check clipboard permissions"
 */
import { toast } from "sonner";

export interface CopyCitation {
  marker: number;
  short_cite?: string | null;
  arxiv_id?: string | null;
  doi?: string | null;
  local_pdf_path?: string | null;
  paper_id?: string | null;
}

export interface CopyMessageProps {
  content: string;
  citations: CopyCitation[];
}

// `[1]`, `[12]`, `[1, 2, 3]` — matches the same numeric block shape as
// the server emitter (src/query/generate.py:_NUMERIC_BLOCK_RE) so the
// stripped text reads as if the markers were never present. Negative
// lookbehind `(?<![A-Za-z0-9_])` skips array indexers like `M[0,1]`,
// matching Message.tsx's NUMERIC_BLOCK_RE on the render side.
const NUMERIC_BLOCK_RE = /(?<![A-Za-z0-9_])\[(?:\d+\s*,\s*)*\d+\]/g;

// Defensive — the chunk-link form `[1](chunk://...)` (rare; the remark
// plugin normally consumes these before render). Strip the whole link
// expression in copy-as-text so the user doesn't see `[1](chunk://...)`
// in their clipboard.
const CHUNK_LINK_RE = /\[[^\]]*\]\(chunk:\/\/[^)]+\)/g;

function citationUrl(c: CopyCitation): string {
  if (c.arxiv_id) return `https://arxiv.org/abs/${c.arxiv_id}`;
  if (c.doi) return `https://doi.org/${c.doi}`;
  if (c.local_pdf_path) return c.local_pdf_path;
  return "";
}

function citationLabel(c: CopyCitation): string {
  if (c.short_cite && c.short_cite.trim().length > 0) {
    return c.short_cite;
  }
  if (c.paper_id) return `paper ${c.paper_id}`;
  return "unknown";
}

/** Format the markdown footnote block — exact `[^N]: label. URL` per
 * citation (D-25), one line each, sorted by marker so the output is
 * deterministic and round-trippable. */
export function formatFootnotes(citations: CopyCitation[]): string {
  if (citations.length === 0) return "";
  const sorted = [...citations].sort((a, b) => a.marker - b.marker);
  return sorted
    .map((c) => {
      const url = citationUrl(c);
      const label = citationLabel(c);
      return url ? `[^${c.marker}]: ${label}. ${url}` : `[^${c.marker}]: ${label}.`;
    })
    .join("\n");
}

/** Strip citation markers + chunk-link forms from the content. Pure
 * formatting — does NOT touch the clipboard. Exposed for tests + any
 * future surface that wants a marker-free preview (e.g. a "share this
 * paragraph" affordance). */
export function stripCitationMarkers(content: string): string {
  return content.replace(CHUNK_LINK_RE, "").replace(NUMERIC_BLOCK_RE, "").replace(/\s+/g, " ").trim();
}

async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * Copy assistant content to the clipboard as plain text with citation
 * markers stripped. Fires Sonner success/failure toast.
 */
export async function copyMessageAsText(props: CopyMessageProps): Promise<void> {
  const stripped = stripCitationMarkers(props.content);
  const ok = await writeClipboard(stripped);
  if (ok) {
    toast("Copied as text");
  } else {
    toast("Couldn't copy — check clipboard permissions");
  }
}

/**
 * Copy assistant content + footnote block to the clipboard as markdown.
 * Markers in the body are kept (so the `[^N]` footnote references read
 * naturally); the footnote block appends after a blank line. Fires
 * Sonner success/failure toast.
 */
export async function copyMessageAsMarkdown(
  props: CopyMessageProps,
): Promise<void> {
  const footnotes = formatFootnotes(props.citations);
  const body = props.content;
  const full = footnotes ? `${body}\n\n${footnotes}` : body;
  const ok = await writeClipboard(full);
  if (ok) {
    toast("Copied as markdown");
  } else {
    toast("Couldn't copy — check clipboard permissions");
  }
}
