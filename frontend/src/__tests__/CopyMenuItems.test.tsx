/**
 * Wave-1 RED test for Copy-as-text / Copy-as-markdown (CHAT-09, D-24, D-25).
 *
 * Implementation lands in Plan 02-14 (frontend/src/lib/copy-text.ts +
 * the menu items in <MessageActions>).
 *
 * Behavioural contract:
 *   - Copy as text strips citation markers like [1], [2] from the content.
 *   - Copy as markdown emits footnote pattern `[^N]: short_cite. URL` (D-25).
 *   - Fallback to "paper {paper_id}" on empty short_cite.
 *   - URL preference: arxiv_id > doi > local PDF path.
 *   - Success toasts "Copied as text" / "Copied as markdown" via Sonner.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
    success: toastMock,
  }),
}));

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
  vi.stubGlobal("navigator", {
    ...globalThis.navigator,
    clipboard: {
      writeText: vi.fn(() => Promise.resolve()),
    },
  });
});

describe("Copy menu items — RED tests until Plan 02-14", () => {
  it("Copy as text strips citation markers like [1] from the content", async () => {
    const { copyMessageAsText } = await import("@/lib/copy-text");

    await copyMessageAsText({
      content: "foo [1] bar [2] baz",
      citations: [],
    });

    const writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    const written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).not.toMatch(/\[\d+\]/);
    expect(written).toMatch(/foo/);
    expect(written).toMatch(/bar/);
    expect(written).toMatch(/baz/);
  });

  it("Copy as markdown emits footnote pattern '[^1]: short_cite. URL' (D-25)", async () => {
    const { copyMessageAsMarkdown } = await import("@/lib/copy-text");

    await copyMessageAsMarkdown({
      content: "foo [1] bar",
      citations: [
        {
          marker: 1,
          short_cite: "Author 2024",
          arxiv_id: "2401.12345",
          doi: null,
          local_pdf_path: null,
          paper_id: "p-1",
        },
      ],
    });

    const writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    const written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).toMatch(/\[\^1\]:\s+Author 2024\./);
    // arxiv URL contains the arxiv id.
    expect(written).toContain("2401.12345");
  });

  it("fallback to 'paper {paper_id}' on empty short_cite", async () => {
    const { copyMessageAsMarkdown } = await import("@/lib/copy-text");

    await copyMessageAsMarkdown({
      content: "see [1]",
      citations: [
        {
          marker: 1,
          short_cite: "",
          arxiv_id: null,
          doi: null,
          local_pdf_path: null,
          paper_id: "abc123",
        },
      ],
    });

    const writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    const written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).toMatch(/\[\^1\]:\s+paper abc123/);
  });

  it("URL preference: arxiv_id > doi > local PDF path", async () => {
    const { copyMessageAsMarkdown } = await import("@/lib/copy-text");

    // (a) arxiv_id present → arxiv URL wins
    await copyMessageAsMarkdown({
      content: "see [1]",
      citations: [
        {
          marker: 1,
          short_cite: "X",
          arxiv_id: "2401.99999",
          doi: "10.1000/xyz",
          local_pdf_path: "/data/local.pdf",
          paper_id: "p",
        },
      ],
    });
    let writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    let written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).toContain("2401.99999");
    expect(written).not.toMatch(/10\.1000/);
    expect(written).not.toMatch(/local\.pdf/);

    // (b) no arxiv, doi present → doi.org URL wins
    vi.restoreAllMocks();
    vi.stubGlobal("navigator", {
      ...globalThis.navigator,
      clipboard: { writeText: vi.fn(() => Promise.resolve()) },
    });
    const { copyMessageAsMarkdown: copy2 } = await import("@/lib/copy-text");
    await copy2({
      content: "see [1]",
      citations: [
        {
          marker: 1,
          short_cite: "X",
          arxiv_id: null,
          doi: "10.1000/xyz",
          local_pdf_path: "/data/local.pdf",
          paper_id: "p",
        },
      ],
    });
    writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).toContain("10.1000/xyz");
    expect(written).not.toMatch(/local\.pdf/);

    // (c) only local PDF → local path is the fallback
    vi.restoreAllMocks();
    vi.stubGlobal("navigator", {
      ...globalThis.navigator,
      clipboard: { writeText: vi.fn(() => Promise.resolve()) },
    });
    const { copyMessageAsMarkdown: copy3 } = await import("@/lib/copy-text");
    await copy3({
      content: "see [1]",
      citations: [
        {
          marker: 1,
          short_cite: "X",
          arxiv_id: null,
          doi: null,
          local_pdf_path: "/data/local.pdf",
          paper_id: "p",
        },
      ],
    });
    writeText = navigator.clipboard.writeText as unknown as ReturnType<
      typeof vi.fn
    >;
    written = String(writeText.mock.calls[0]?.[0] ?? "");
    expect(written).toMatch(/local\.pdf/);
  });

  it("Success toasts 'Copied as text' / 'Copied as markdown' fire from Sonner", async () => {
    const { copyMessageAsText, copyMessageAsMarkdown } = await import(
      "@/lib/copy-text"
    );

    await copyMessageAsText({ content: "x", citations: [] });
    await copyMessageAsMarkdown({ content: "x", citations: [] });

    const calls = toastMock.mock.calls.map((c) => String(c[0])).join(" ");
    expect(calls).toMatch(/Copied as text/);
    expect(calls).toMatch(/Copied as markdown/);
  });
});
