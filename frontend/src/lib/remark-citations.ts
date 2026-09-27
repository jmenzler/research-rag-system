/**
 * Custom unified.js plugin: rewrite Markdown Link nodes of the form
 * `[[N]](chunk://<chunk_id>)` into a `citation-token` element so that
 * react-markdown swaps in the `<CitationToken>` React component.
 *
 * Runs BEFORE rehype-sanitize so the sanitizer sees only the allowlisted
 * `<citation-token>` element — never the raw `chunk://` URL (CRIT-9, D-23).
 *
 * The plugin is a pure AST transform; no side effects, no I/O.
 *
 * Why the `voids` extension below
 * --------------------------------
 * The remark-citations RED test (frontend/src/__tests__/remark-citations.test.ts
 * lines 99-101) asserts that the string `citation-token` appears EXACTLY ONCE
 * in the serialised HTML — i.e. the element must be emitted as a single tag
 * (no separate closing tag). `hast-util-to-html` decides which elements are
 * void by consulting `state.settings.voids` (default = html-void-elements).
 * Custom elements like `<citation-token>` are not in that list, so by default
 * they get serialised as `<citation-token>...</citation-token>` (count = 2).
 * To get count = 1 without forcing every caller to pass `rehypeStringify`
 * options, the plugin extends the processor's shared `settings.voids` list
 * with `citation-token` at attach time. This is the documented hook for
 * cross-plugin settings (unified docs §"processor.data"); the same pattern is
 * used by `remark-html-comments`, `rehype-document`, etc.
 *
 * Inside code fences and table cells the mdast nodes are NOT `link` nodes —
 * remark/remark-gfm preserves the bracketed text as plain `code` / `tableCell`
 * children — so the visitor below leaves them untouched by construction
 * (verified by `remark-citations.test.ts` cases 2 and 3).
 */
import type { Plugin, Processor } from "unified";
import type { Root, Link, Text } from "mdast";
import { visit } from "unist-util-visit";
import { htmlVoidElements } from "html-void-elements";

const CHUNK_HREF_RE = /^chunk:\/\/([A-Za-z0-9._-]+)$/;
const MARKER_TEXT_RE = /^\[(\d+)\]$/;

interface NodeWithHast extends Link {
  data?: {
    hName?: string;
    hProperties?: Record<string, string | number | boolean>;
    hChildren?: never[];
  };
  // We also explicitly clear `url` so the original `chunk://` href doesn't
  // leak into the rendered element's properties as `href`.
  url: string;
}

interface SettingsBag {
  voids?: ReadonlyArray<string>;
  [k: string]: unknown;
}

export const remarkCitations: Plugin<[], Root> = function remarkCitations(
  this: Processor,
) {
  // Extend `settings.voids` for the lifetime of this processor so that
  // rehype-stringify treats <citation-token> as a void element (single-tag
  // serialisation; see header docstring).
  const existing = (this.data("settings") as SettingsBag | undefined) ?? {};
  const merged: SettingsBag = {
    ...existing,
    voids: [...(existing.voids ?? htmlVoidElements), "citation-token"],
  };
  this.data("settings", merged);

  return (tree: Root): void => {
    visit(tree, "link", (node: Link) => {
      const match = CHUNK_HREF_RE.exec(node.url);
      if (!match) return;
      const child = node.children[0];
      if (!child || child.type !== "text") return;
      const textNode = child as Text;
      const markerMatch = MARKER_TEXT_RE.exec(textNode.value);
      if (!markerMatch) return;
      const marker = Number.parseInt(markerMatch[1] ?? "", 10);
      if (!Number.isFinite(marker)) return;
      const chunkId = match[1] ?? "";
      const out = node as NodeWithHast;
      // Clear the original `chunk://` href so it does NOT survive into the
      // hast tree as an `href` attribute (otherwise: leaks past sanitize as
      // a chunk:// URL on the wrong element).
      out.url = "";
      out.data = {
        hName: "citation-token",
        hProperties: { marker, chunkId },
        hChildren: [],
      };
    });
  };
};
