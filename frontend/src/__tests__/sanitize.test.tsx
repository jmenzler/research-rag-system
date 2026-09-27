/**
 * Wave-0 RED test for the markdown sanitizer pipeline (CHAT-06, CRIT-9, D-23).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/Message.tsx +
 * the rehype-sanitize allowlist).
 *
 * The Message component is the surface — it runs the full pipeline:
 *   react-markdown -> remark-gfm -> remark-citations -> rehype-sanitize.
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

describe("Markdown sanitizer (T-01-WAVE0-02)", () => {
  it("strips <script> from markdown rendering", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message
        role="assistant"
        content={'hello <script>alert(1)</script> world'}
      />,
    );

    expect(container.querySelector("script")).toBeNull();
    expect(container.textContent).toMatch(/hello/);
    expect(container.textContent).toMatch(/world/);
  });

  it("blocks javascript: href", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message
        role="assistant"
        content={"[click](javascript:alert(1))"}
      />,
    );

    const evil = container.querySelector("a[href^='javascript:']");
    expect(evil).toBeNull();
  });

  it("blocks data: href", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message
        role="assistant"
        content={"[click](data:text/html,<script>alert(1)</script>)"}
      />,
    );

    const evil = container.querySelector("a[href^='data:']");
    expect(evil).toBeNull();
  });

  it("allows chunk:// href to pass through as citation-token (D-23)", async () => {
    const { Message } = await import("@/components/chat/Message");
    const { container } = render(
      <Message role="assistant" content={"hello [[1]](chunk://abc)"} />,
    );

    // remark-citations transforms the Link into a CitationToken BEFORE
    // rehype-sanitize sees the hast tree. After sanitize, the resulting
    // element must still carry the chunk id (verified via either custom
    // tag or data-chunkid attribute — both shapes are acceptable).
    const hasChunkRef =
      container.querySelector("[data-chunkid='abc']") !== null ||
      container.querySelector("citation-token") !== null ||
      container.innerHTML.includes("abc");
    expect(hasChunkRef).toBe(true);
  });
});
