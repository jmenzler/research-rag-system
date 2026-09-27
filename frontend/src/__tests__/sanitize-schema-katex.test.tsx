/**
 * Wave-1 RED test for CITATION_SANITIZE_SCHEMA_WITH_KATEX.
 *
 * Implementation lands in Plan 02-13 (frontend/src/lib/sanitize-schema.ts —
 * extends the existing CITATION_SANITIZE_SCHEMA with MathML tags + KaTeX
 * classNames + <mark> + a tight `style` regex on <span>).
 *
 * Behavioural contract (UI-SPEC § Component Inventory Extends; RESEARCH § Pitfall 1):
 *   - Allowlists ~24 MathML tags
 *   - Allowlists KaTeX classNames (katex, katex-display, katex-html, katex-mathml,
 *     mord, mbin, vlist, vlist-r, accent, etc.)
 *   - Allowlists <mark>
 *   - <span style="…"> permitted only via TIGHT regex /^[\s\w.:;%-]+$/
 *   - Blocks <script>, <span style="javascript:…">, <mark onclick="…">
 */
import { describe, expect, it } from "vitest";

async function pipelineThroughSchema(html: string): Promise<string> {
  const { unified } = await import("unified");
  const rehypeParse = (await import("rehype-parse")).default;
  const rehypeSanitize = (await import("rehype-sanitize")).default;
  const rehypeStringify = (await import("rehype-stringify")).default;
  // Typed cast so test collection succeeds before Plan 02-13 adds the new
  // export. The test fails RED at runtime when the export is missing.
  const schemaMod = (await import("@/lib/sanitize-schema")) as unknown as {
    CITATION_SANITIZE_SCHEMA_WITH_KATEX?: unknown;
  };
  const schema = schemaMod.CITATION_SANITIZE_SCHEMA_WITH_KATEX as never;

  return String(
    await unified()
      .use(rehypeParse, { fragment: true })
      .use(rehypeSanitize, schema)
      .use(rehypeStringify)
      .process(html),
  );
}

describe("CITATION_SANITIZE_SCHEMA_WITH_KATEX — RED tests until Plan 02-13", () => {
  it("preserves all MathML tags listed in RESEARCH § Pitfall 1", async () => {
    const tags = [
      "math",
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
      "annotation",
      "semantics",
    ];
    for (const tag of tags) {
      const html = `<${tag}>x</${tag}>`;
      const cleaned = await pipelineThroughSchema(html);
      expect(cleaned.toLowerCase()).toContain(`<${tag.toLowerCase()}>`);
    }
  });

  it("preserves <mark> tag (FTS5 snippet highlight)", async () => {
    const cleaned = await pipelineThroughSchema("<mark>hit</mark>");
    expect(cleaned.toLowerCase()).toContain("<mark>");
  });

  it("preserves KaTeX classNames", async () => {
    const classes = [
      "katex",
      "katex-display",
      "katex-html",
      "katex-mathml",
      "mord",
      "mbin",
      "vlist",
      "vlist-r",
      "accent",
    ];
    for (const c of classes) {
      const cleaned = await pipelineThroughSchema(
        `<span class="${c}">x</span>`,
      );
      expect(cleaned).toContain(c);
    }
  });

  it("allows <span style='height:0.5em'> via tight regex", async () => {
    const cleaned = await pipelineThroughSchema(
      '<span style="height:0.5em">x</span>',
    );
    expect(cleaned).toMatch(/style=/);
    expect(cleaned).toMatch(/height/);
  });

  it("BLOCKS <span style='javascript:alert(1)'>", async () => {
    const cleaned = await pipelineThroughSchema(
      '<span style="javascript:alert(1)">x</span>',
    );
    expect(cleaned).not.toMatch(/javascript:/i);
  });

  it("BLOCKS <script>", async () => {
    const cleaned = await pipelineThroughSchema(
      "<script>alert(1)</script>",
    );
    expect(cleaned.toLowerCase()).not.toContain("<script");
  });

  it("BLOCKS <mark onclick='...'>", async () => {
    const cleaned = await pipelineThroughSchema(
      '<mark onclick="alert(1)">x</mark>',
    );
    expect(cleaned.toLowerCase()).not.toMatch(/onclick/);
  });
});
