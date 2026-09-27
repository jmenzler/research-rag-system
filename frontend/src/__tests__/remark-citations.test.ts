/**
 * Wave-0 RED test for the custom remark-citations AST plugin.
 *
 * Implementation lands in Plan 06 (frontend/src/lib/remark-citations.ts).
 * Tests reference the plugin via dynamic import; vitest treats the missing
 * module as a test failure (NOT a collection error) — RED until Plan 06 ships.
 *
 * Behavioural contract (D-06, CHAT-06, CHAT-04):
 *   - Replaces [[N]](chunk://uuid) Link nodes with a citation-token JSX hName.
 *   - Leaves [1] inside code fences untouched.
 *   - Leaves [1] inside table cells untouched.
 *   - Idempotent under React.StrictMode double-render (Risk 3).
 */
import { describe, it, expect } from "vitest";

describe("remark-citations plugin", () => {
  it("replaces [[N]](chunk://uuid) Link nodes with citation-token hName", async () => {
    const { unified } = await import("unified");
    const remarkParse = (await import("remark-parse")).default;
    const remarkRehype = (await import("remark-rehype")).default;
    const rehypeStringify = (await import("rehype-stringify")).default;
    const { remarkCitations } = await import("@/lib/remark-citations");

    const html = String(
      await unified()
        .use(remarkParse)
        .use(remarkCitations)
        .use(remarkRehype, { allowDangerousHtml: false })
        .use(rehypeStringify)
        .process("hello [[1]](chunk://abc-def-123) world"),
    );

    expect(html).toContain("citation-token");
    expect(html).toMatch(/marker=["']?1["']?/);
    expect(html).toMatch(/chunkid=["']abc-def-123["']/i);
  });

  it("leaves [1] inside a code fence untouched", async () => {
    const { unified } = await import("unified");
    const remarkParse = (await import("remark-parse")).default;
    const remarkRehype = (await import("remark-rehype")).default;
    const rehypeStringify = (await import("rehype-stringify")).default;
    const { remarkCitations } = await import("@/lib/remark-citations");

    const html = String(
      await unified()
        .use(remarkParse)
        .use(remarkCitations)
        .use(remarkRehype)
        .use(rehypeStringify)
        .process("```\n[1]\n```"),
    );

    expect(html).not.toContain("citation-token");
  });

  it("leaves [1] inside a table cell untouched", async () => {
    const { unified } = await import("unified");
    const remarkParse = (await import("remark-parse")).default;
    const remarkGfm = (await import("remark-gfm")).default;
    const remarkRehype = (await import("remark-rehype")).default;
    const rehypeStringify = (await import("rehype-stringify")).default;
    const { remarkCitations } = await import("@/lib/remark-citations");

    const md = "| a | b |\n|---|---|\n| [1] | x |";
    const html = String(
      await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkCitations)
        .use(remarkRehype)
        .use(rehypeStringify)
        .process(md),
    );

    expect(html).not.toContain("citation-token");
  });

  it("renders identically under React.StrictMode (Risk 3)", async () => {
    const { unified } = await import("unified");
    const remarkParse = (await import("remark-parse")).default;
    const remarkRehype = (await import("remark-rehype")).default;
    const rehypeStringify = (await import("rehype-stringify")).default;
    const { remarkCitations } = await import("@/lib/remark-citations");

    // Pure AST transform is deterministic. Running the same pipeline twice
    // (analogous to React 19 strict-mode double-render) MUST yield identical
    // output. If the plugin keeps state across calls, this catches it.
    const pipeline = unified()
      .use(remarkParse)
      .use(remarkCitations)
      .use(remarkRehype)
      .use(rehypeStringify);

    const md = "hello [[1]](chunk://abc) world";
    const first = String(await pipeline.process(md));
    const second = String(await pipeline.process(md));
    expect(first).toBe(second);
    // citation count is exactly one (no duplication on second render)
    const count = (first.match(/citation-token/g) ?? []).length;
    expect(count).toBe(1);
  });
});
