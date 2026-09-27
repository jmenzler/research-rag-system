/**
 * Render one chat turn.
 *
 * D-02: Claude.ai-style flush-left rendering — role label above markdown
 *       content, no left/right bubbles.
 * D-23: Markdown pipeline runs IN ORDER (locked in Phase 2 D-26):
 *         remarkGfm → remarkMath → remarkCitations → rehypeSanitize → rehypeKatex
 *       remarkCitations rewrites the AST BEFORE rehypeSanitize sees it, so
 *       the sanitizer only encounters the allowlisted `citation-token`
 *       element and never the raw `chunk://` URL (CRIT-9). rehypeKatex runs
 *       AFTER rehypeSanitize per the official rehype-sanitize guidance —
 *       KaTeX is trusted to emit safe MathML/HTML once the input has been
 *       gated through CITATION_SANITIZE_SCHEMA_WITH_KATEX.
 *
 * Plan 02-13: the plugin arrays moved to ``@/lib/markdown-plugins`` so the
 * CitationHoverCard / CitationSheet renderers (D-27) can share the same
 * locked configuration without duplication. KaTeX CSS is imported here
 * once — vite's manualChunks splits it into a separate chunk so the chat
 * surface doesn't pay the ~250KB on initial paint until the renderer
 * actually mounts (POLISH-09 budget).
 *
 * Two-shape props: callers MAY pass either
 *   { role, content, messageId?, citations? }   (low-friction — sanitize test)
 * OR
 *   { message: ChatMessage, citations? }        (typed — MessageList consumer)
 *
 * The component picks whichever set is provided.
 */
import "katex/dist/katex.min.css";

import { type ReactElement } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { CircleDot } from "lucide-react";

import type { ChatMessage } from "@/lib/api";
import type { CitationPayload } from "@/lib/sse-client";
import {
  CITATION_SANITIZE_SCHEMA,
  CITATION_SANITIZE_SCHEMA_WITH_KATEX,
} from "@/lib/sanitize-schema";
import { REMARK_PLUGINS, REHYPE_PLUGINS } from "@/lib/markdown-plugins";
import { CitationToken } from "@/components/chat/CitationToken";
import { cn } from "@/lib/cn";

// Re-export so tests / other consumers can grab the schemas via Message too.
export { CITATION_SANITIZE_SCHEMA, CITATION_SANITIZE_SCHEMA_WITH_KATEX };

// Mirror of the server's _NUMERIC_BLOCK_RE (src/query/generate.py:93). Matches
// `[1]` or `[1, 2, 3, 8]` blocks that are NOT preceded by an identifier char,
// so array indexers like `M[0,1]` and code-block tokens stay intact. Per-marker
// links are emitted as `[[N]](chunk://<child_id>)` — the exact shape the remark
// plugin (remark-citations.ts) looks for. Inside fenced code / inline code the
// surrounding context is still preserved by remark-gfm, so the visitor in the
// plugin leaves bracketed text alone when it is NOT a `link` node.
const NUMERIC_BLOCK_RE = /(?<![A-Za-z0-9_])\[((?:\d+\s*,\s*)*\d+)\]/g;

// Plan 02-13: Promote single-line ``$$expr$$`` paragraphs to multi-line
// fenced form so remark-math 6 emits a display-math node (.katex-display)
// instead of an inline-math node. remark-math's syntax follows pandoc:
// ``$$...$$`` only becomes display math when the closing ``$$`` is on a
// separate line (verified by reading mdast-util-math source). LLMs (and
// the markdown-katex RED test) commonly emit ``$$x = 1$$`` on a single
// line; without this normaliser those would render inline, breaking
// long equations and the test contract.
//
// Scope:
//   - Only matches a ``$$…$$`` token that occupies the ENTIRE line (after
//     trimming whitespace). Inline math mid-sentence (``a $$x$$ b``) is
//     left alone — that author/LLM clearly meant inline placement.
//   - Skips matches inside fenced code blocks. We do not parse the full
//     CommonMark AST here (cheap pre-pass); the heuristic is that any
//     line which sits between two ```` ``` ```` fences is ignored.
//   - No-op on user-turn content (only assistant turns go through this).
const DISPLAY_MATH_LINE_RE = /^[ \t]*\$\$([^\n$][\s\S]*?)\$\$[ \t]*$/;

// Models commonly emit display math as ``\[ … \]`` and inline as ``\( … \)``
// (LaTeX delimiters), but in markdown ``\[`` is an ESCAPED bracket that
// renders as a literal ``[`` — remark-math only tokenizes ``$…$`` / ``$$…$$``,
// so the math never renders. Rewrite the LaTeX delimiters to dollar form
// BEFORE the content reaches remark-math.
//
// Scope guard: never touch content inside fenced code blocks (``` / ~~~) or
// inline code spans (`…`). We process the document segment-by-segment,
// passing code regions through untouched so e.g. a literal ``\[`` shown in a
// code example stays verbatim.
const DISPLAY_LATEX_RE = /\\\[([\s\S]*?)\\\]/g;
const INLINE_LATEX_RE = /\\\(([\s\S]*?)\\\)/g;

function rewriteLatexDelimitersInText(text: string): string {
  return text
    .replace(DISPLAY_LATEX_RE, (_m, body: string) => `$$${body}$$`)
    .replace(INLINE_LATEX_RE, (_m, body: string) => `$${body}$`);
}

// Split a non-fenced block into alternating text / inline-code segments and
// rewrite LaTeX delimiters only in the text segments. The display-math regex
// uses [\s\S]*? so a ``\[ … \]`` block that spans several lines (the common
// case) is still rewritten.
function rewriteLatexOutsideInlineCode(block: string): string {
  const parts = block.split(/(`[^`]*`)/);
  for (let p = 0; p < parts.length; p += 1) {
    if (p % 2 === 1) continue; // inline code span — leave verbatim
    parts[p] = rewriteLatexDelimitersInText(parts[p] ?? "");
  }
  return parts.join("");
}

function normalizeLatexDelimiters(content: string): string {
  if (!content || (!content.includes("\\[") && !content.includes("\\("))) {
    return content;
  }
  // Split on fenced code blocks (``` … ``` or ~~~ … ~~~) so their contents
  // pass through untouched; rewrite only the non-fenced segments. Odd-indexed
  // pieces are the captured fences.
  const FENCE_BLOCK_RE = /(^[ \t]*(?:`{3,}|~{3,})[\s\S]*?^[ \t]*(?:`{3,}|~{3,}).*$)/m;
  const segments = content.split(FENCE_BLOCK_RE);
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i] ?? "";
    const isFence = /^[ \t]*(?:`{3,}|~{3,})/.test(seg);
    if (isFence) continue;
    segments[i] = rewriteLatexOutsideInlineCode(seg);
  }
  return segments.join("");
}

function normalizeDisplayMath(content: string): string {
  if (!content || !content.includes("$$")) return content;
  const lines = content.split("\n");
  let insideFence = false;
  const FENCE_RE = /^[ \t]*(`{3,}|~{3,})/;
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? "";
    if (FENCE_RE.test(line)) {
      insideFence = !insideFence;
      continue;
    }
    if (insideFence) continue;
    const m = DISPLAY_MATH_LINE_RE.exec(line);
    if (m) {
      // Expand "$$expr$$" into three lines so remark-math sees a flow node.
      // The expression body keeps its original whitespace; we just split
      // the delimiters onto their own lines.
      lines[i] = `$$\n${m[1]}\n$$`;
    }
  }
  return lines.join("\n");
}

function rewriteCitationMarkers(
  content: string,
  citations: CitationPayload[],
): string {
  if (!content || citations.length === 0) return content;
  const byMarker = new Map<number, CitationPayload>();
  for (const c of citations) byMarker.set(c.marker, c);
  return content.replace(NUMERIC_BLOCK_RE, (match, group: string) => {
    const markers = group
      .split(",")
      .map((s) => Number.parseInt(s.trim(), 10))
      .filter((n) => Number.isFinite(n));
    if (markers.length === 0) return match;
    const links = markers.map((m) => {
      const cite = byMarker.get(m);
      // Unknown marker → render as a resolved=false token by linking to a
      // placeholder chunk id. The plugin still emits `<citation-token>`; the
      // CitationToken renderer falls back to the struck-through `[?]` pill
      // because `citations.find(...)` returns undefined and resolved=false.
      const chunkId = cite?.child_id ?? "unresolved";
      return `[[${m}]](chunk://${chunkId})`;
    });
    return links.join(" ");
  });
}

function buildComponents(
  messageId: string,
  citations: CitationPayload[],
): Components {
  // react-markdown 10's `Components` is keyed on HTML tag names. Cast through
  // `unknown` to add our custom `<citation-token>` entry without loosening
  // the type elsewhere.
  const CitationTokenRenderer = (props: {
    marker?: number | string;
    chunkId?: string | null;
    chunkid?: string | null;
  }) => {
    const markerRaw = props.marker;
    const marker =
      typeof markerRaw === "number"
        ? markerRaw
        : Number.parseInt(String(markerRaw ?? "0"), 10);
    const cite = citations.find((c) => c.marker === marker);
    // resolved unknown → trust the server (canonical). Phase 1 the
    // citations payload is always emitted before any pill renders.
    const resolved = cite ? cite.resolved : true;
    const chunkId =
      (props.chunkId ?? props.chunkid ?? cite?.child_id) ?? null;
    return (
      <CitationToken
        marker={marker}
        chunkId={chunkId}
        resolved={resolved}
        messageId={messageId}
      />
    );
  };
  return { "citation-token": CitationTokenRenderer } as unknown as Components;
}

type Role = "user" | "assistant";

export interface MessageProps {
  /** Pass either the structured ChatMessage… */
  message?: ChatMessage;
  /** …or the loose role+content pair (sanitize test contract). */
  role?: Role;
  content?: string;
  /** Optional explicit id (used as the openCitation message handle). */
  messageId?: string;
  /** Citation payloads from the live SSE stream. */
  citations?: CitationPayload[];
}

export function Message(props: MessageProps): ReactElement {
  const role: Role = props.message?.role ?? props.role ?? "assistant";
  const content: string = props.message?.content ?? props.content ?? "";
  const messageId: string =
    props.messageId ?? props.message?.id ?? "msg-unknown";
  const citations = props.citations ?? [];
  const components = buildComponents(messageId, citations);
  const renderedContent =
    role === "assistant"
      ? rewriteCitationMarkers(
          normalizeDisplayMath(normalizeLatexDelimiters(content)),
          citations,
        )
      : content;

  const isUser = role === "user";
  return (
    <div
      data-testid={`message-${role}-${messageId}`}
      data-role={role}
      className="grid grid-cols-[32px_1fr] gap-4 py-5"
    >
      <div className="flex items-start pt-0.5">
        <div
          aria-hidden="true"
          className={cn(
            "flex size-7 items-center justify-center rounded-full font-mono text-[10px] font-semibold tracking-[-0.02em]",
            isUser
              ? "border border-[var(--p3-border)] bg-[var(--p3-surface)] text-[var(--p3-muted)]"
              : "border border-[var(--p3-primary-edge)] bg-[var(--p3-primary-soft)] text-[var(--p3-primary-text)]",
          )}
        >
          {isUser ? "You" : <CircleDot className="size-3.5" />}
        </div>
      </div>
      <div className="min-w-0">
        <div className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--p3-muted)]">
          {isUser ? "You" : "Assistant"}
        </div>
        {role === "assistant" ? (
          <article className="prose dark:prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={REMARK_PLUGINS}
              rehypePlugins={REHYPE_PLUGINS}
              components={components}
            >
              {renderedContent}
            </ReactMarkdown>
          </article>
        ) : (
          <div className="whitespace-pre-wrap rounded-md border border-[var(--p3-border)] bg-[var(--p3-surface)] px-3.5 py-2.5 text-[var(--p3-fg)]">
            {content}
          </div>
        )}
      </div>
    </div>
  );
}
