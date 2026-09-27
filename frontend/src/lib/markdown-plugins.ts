/**
 * Centralized markdown plugin arrays for the chat renderer.
 *
 * Phase 2 D-26 LOCKED ORDER (do not reorder):
 *
 *   remark:
 *     1. remarkGfm          — tables, strikethrough, autolinks
 *     2. remarkMath         — parse $…$ / $$…$$ into math AST nodes
 *     3. remarkCitations    — rewrite [[N]](chunk://…) into <citation-token>
 *
 *   rehype (runs in this order):
 *     1. rehypeSanitize     — gate every tag/attr through KaTeX-aware allowlist
 *     2. rehypeKatex        — render math nodes AFTER sanitize per official
 *                             rehype-sanitize guidance (KaTeX is trusted to
 *                             emit safe MathML; CITATION_SANITIZE_SCHEMA_WITH_KATEX
 *                             does not strip its classNames or MathML tags).
 *
 * Why this order:
 *   - remark-math must precede remark-citations so $…$ does not get
 *     mis-tokenized when both appear in the same paragraph.
 *   - remarkCitations rewrites links to a custom <citation-token /> element
 *     BEFORE rehype-sanitize sees it, so the sanitizer only encounters the
 *     allowlisted tag (CRIT-9 / D-23 carried forward from Phase 1).
 *   - rehype-sanitize MUST run before rehype-katex (RESEARCH §Pitfall 1):
 *     KaTeX emits trusted MathML; if it ran first the sanitizer would
 *     strip its classNames and break rendering.
 *
 * Test coverage:
 *   frontend/src/__tests__/markdown-katex.test.tsx — round-trip render check
 *     for inline math, display math, code-fence non-math, citation + math
 *     coexistence, malformed math fallback, and plugin order verification.
 */
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeSanitize from "rehype-sanitize";
import rehypeKatex from "rehype-katex";
import type { PluggableList } from "unified";

import { remarkCitations } from "@/lib/remark-citations";
import { CITATION_SANITIZE_SCHEMA_WITH_KATEX } from "@/lib/sanitize-schema";

export const REMARK_PLUGINS: PluggableList = [
  remarkGfm,
  remarkMath,
  remarkCitations,
];

export const REHYPE_PLUGINS: PluggableList = [
  [rehypeSanitize, CITATION_SANITIZE_SCHEMA_WITH_KATEX],
  rehypeKatex,
];
