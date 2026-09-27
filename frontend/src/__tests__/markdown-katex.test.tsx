/**
 * Wave-1 RED test for the markdown + KaTeX pipeline (D-26).
 *
 * Implementation lands in Plan 02-13 (frontend/src/components/chat/Message.tsx
 * — adds `remark-math` and `rehype-katex` to the existing pipeline, in this
 * order):
 *
 *   remark-gfm → remark-math → remark-citations → rehype-sanitize → rehype-katex
 *
 * The Message component is the surface — render it and assert on output.
 *
 * Note: `remark-math`/`rehype-katex` are added by Plan 02-13; the imports here
 * are deferred (`await import(...)`) so collection succeeds even before they
 * exist on disk (Vitest treats the resolve failure as a test failure, NOT a
 * collection error).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

describe("Markdown + KaTeX pipeline (D-26) — RED tests until Plan 02-13", () => {
  it("inline math $x = 1$ renders as <span class='katex'>", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"inline: $x = 1$ end"} />,
    );
    expect(container.querySelector("span.katex")).not.toBeNull();
  });

  it("display math $$x = 1$$ renders as <span class='katex-display'>", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"display:\n\n$$x = 1$$\n"} />,
    );
    expect(container.querySelector("span.katex-display, .katex-display")).not.toBeNull();
  });

  it("LaTeX display delimiters \\[ x = 1 \\] render as display KaTeX", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"display:\n\n\\[ x = 1 \\]\n"} />,
    );
    expect(
      container.querySelector("span.katex-display, .katex-display"),
    ).not.toBeNull();
    // The literal escaped bracket must NOT survive as plaintext.
    expect(container.textContent ?? "").not.toContain("\\[");
  });

  it("LaTeX inline delimiters \\( x \\) render as inline KaTeX", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"inline: \\( x \\) end"} />,
    );
    expect(container.querySelector("span.katex")).not.toBeNull();
    expect(container.textContent ?? "").not.toContain("\\(");
  });

  it("LaTeX delimiters inside a code fence stay verbatim", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"```\n\\[ x = 1 \\]\n```"} />,
    );
    const code = container.querySelector("code");
    expect(code).not.toBeNull();
    expect(code?.querySelector(".katex")).toBeNull();
    expect(code?.textContent ?? "").toContain("\\[ x = 1 \\]");
  });

  it("code-fence `$x = 1$` stays as plaintext (no katex class)", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"```\n$x = 1$\n```"} />,
    );
    const code = container.querySelector("code");
    expect(code).not.toBeNull();
    expect(code?.querySelector(".katex")).toBeNull();
    expect(code?.textContent ?? "").toContain("$x = 1$");
  });

  it("citation marker [1] coexists with math $x$ in same paragraph", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message
        role="assistant"
        content={"see [[1]](chunk://abc) and $x$ here"}
      />,
    );
    // KaTeX renders.
    expect(container.querySelector(".katex")).not.toBeNull();
    // Citation token also renders (custom tag or data-chunkid).
    const html = container.innerHTML;
    const hasCite =
      container.querySelector("[data-chunkid='abc']") !== null ||
      container.querySelector("citation-token") !== null ||
      html.includes("abc");
    expect(hasCite).toBe(true);
  });

  it("malformed math $\\frac{1$ emits <span class='katex-error'>", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"oops $\\frac{1$ end"} />,
    );
    expect(container.querySelector(".katex-error")).not.toBeNull();
  });

  it("plugin order: remark-gfm → remark-math → remark-citations → rehype-sanitize → rehype-katex", async () => {
    // Indirect proof of order:
    //   - GFM features (tables) survive
    //   - Math renders (remark-math + rehype-katex)
    //   - Citation token survives (remark-citations runs before sanitize)
    //   - <script> blocked (rehype-sanitize present)
    //   - KaTeX output survives sanitize (rehype-katex AFTER sanitize per official guidance)
    const { Message } = await import("@/components/chat/Message");
    const md = [
      "| a | b |",
      "|---|---|",
      "| $x$ | [[1]](chunk://abc) |",
      "",
      "<script>alert(1)</script>",
    ].join("\n");

    const { container } = render(<Message role="assistant" content={md} />);
    expect(container.querySelector("table")).not.toBeNull(); // remark-gfm
    expect(container.querySelector(".katex")).not.toBeNull(); // remark-math + rehype-katex
    const html = container.innerHTML;
    const hasCite =
      container.querySelector("[data-chunkid='abc']") !== null ||
      container.querySelector("citation-token") !== null ||
      html.includes("abc");
    expect(hasCite).toBe(true); // remark-citations before sanitize
    expect(container.querySelector("script")).toBeNull(); // rehype-sanitize
  });
});
