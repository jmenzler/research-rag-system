/**
 * Markdown sanitize SCHEMA for chat-rendering (CRIT-9 / D-23).
 *
 *   - allows the custom <citation-token> element with marker + chunkId attrs
 *   - restricts anchor hrefs to http://, https://, chunk:// schemes only
 *   - everything else inherits the rehype-sanitize defaultSchema allowlist
 *
 * Lives in lib/ (not co-located with Message.tsx) so it can be imported by
 * non-component consumers (tests, future server-side rendering) without
 * tripping the react-refresh/only-export-components ESLint rule.
 *
 * Plan 02-13 adds a sibling export ``CITATION_SANITIZE_SCHEMA_WITH_KATEX``
 * that ALSO allows MathML + KaTeX classNames + a tight span ``style`` regex
 * for KaTeX vlist spans (RESEARCH §Pitfall 1). The chat Message renderer
 * picks the WITH_KATEX schema; sidebar snippet renderers (no math content)
 * stick with the bare schema for minimum surface area.
 */
import { defaultSchema, type Options as SanitizeOptions } from "rehype-sanitize";

type PropertyDefinition = string | [string, ...(string | RegExp | number)[]];

const baseAttrs = (defaultSchema.attributes ?? {}) as Record<
  string,
  PropertyDefinition[]
>;

export const CITATION_SANITIZE_SCHEMA: SanitizeOptions = {
  ...defaultSchema,
  // Plan 02-12 — <mark> is needed to render FTS5 snippet() highlights in
  // the sidebar search results. Allowlisted with NO attributes so any
  // <mark onclick=...> from a hostile payload is stripped (T-02-12-01).
  tagNames: [...(defaultSchema.tagNames ?? []), "citation-token", "mark"],
  attributes: {
    ...baseAttrs,
    "citation-token": ["marker", "chunkId", "chunkid"],
    mark: [], // no attrs — XSS-safe FTS5 highlight
    a: [
      ["href", /^https?:\/\//i, /^chunk:\/\//i],
      ...(baseAttrs.a ?? []),
    ],
  },
};

// Plan 02-13 — KaTeX-aware schema for the chat Message renderer.
//
// The rehype-sanitize pipeline runs BEFORE rehype-katex (per official
// rehype-sanitize guidance — KaTeX is trusted to emit safe MathML/HTML
// AFTER untrusted markdown has been gated). The allowlist below covers
// every node KaTeX 0.16 emits:
//
//   - MathML primitives (math, mrow, mi, mo, mn, mfrac, msup, msub, …)
//   - KaTeX HTML wrapper classNames (katex, katex-display, katex-html, …)
//   - The TIGHT span style regex /^[\s\w.:;%-]+$/ allows numeric/length
//     values like "height:0.5em" that KaTeX's vlist spans require, while
//     rejecting "javascript:…" / URL schemes / expression() / quotes.
//
// Verified by tests/sanitize-schema-katex.test.tsx (RED → GREEN this plan).
const KATEX_MATHML_TAGS = [
  "math",
  "annotation",
  "semantics",
  "mrow",
  "mi",
  "mo",
  "mn",
  "mfrac",
  "msup",
  "msub",
  "msubsup",
  "munder",
  "mover",
  "munderover",
  "msqrt",
  "mroot",
  "mtable",
  "mtr",
  "mtd",
  "mtext",
  "mspace",
  "mstyle",
  "menclose",
  "mpadded",
] as const;

// Tight regex used both as the <span style="…"> attribute filter and as
// documentation: matches plain CSS length / percentage / numeric tokens
// (digits, dot, colon, semicolon, percent, dash, underscore, whitespace,
// word chars). Critically does NOT permit "(" / ")" so url(...) /
// expression(...) cannot slip through, and does NOT permit "/" / "&" /
// quote chars so "javascript:" / "data:…" payloads fail.
const SPAN_STYLE_REGEX = /^[\s\w.:;%-]+$/;

export const CITATION_SANITIZE_SCHEMA_WITH_KATEX: SanitizeOptions = {
  ...CITATION_SANITIZE_SCHEMA,
  tagNames: [
    ...(CITATION_SANITIZE_SCHEMA.tagNames ?? []),
    ...KATEX_MATHML_TAGS,
  ],
  attributes: {
    ...(CITATION_SANITIZE_SCHEMA.attributes ?? {}),
    // Allow className on every element so KaTeX's mord/mbin/vlist/…
    // class hierarchy survives sanitize. Inherits the existing wildcard
    // entry from CITATION_SANITIZE_SCHEMA (currently empty in defaults).
    "*": [
      ...((CITATION_SANITIZE_SCHEMA.attributes?.["*"] ?? []) as PropertyDefinition[]),
      "className",
    ],
    // <span> requires both className (vlist, katex-html, …) and the
    // tight style regex (height:…em for vertical lists).
    span: [
      ...((CITATION_SANITIZE_SCHEMA.attributes?.span ?? []) as PropertyDefinition[]),
      "className",
      ["style", SPAN_STYLE_REGEX],
    ],
  },
};
