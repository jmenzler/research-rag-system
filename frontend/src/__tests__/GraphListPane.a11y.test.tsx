/**
 * RED — GREEN after 04-05.
 *
 * GRAPH-06 accessibility contract for GraphListPane:
 *   - role="listbox" container with aria-label="Citation graph nodes"
 *   - each row has role="option" and aria-selected
 *   - container has tabIndex=0 for keyboard navigation
 *   - no axe violations on render
 *
 * From 04-PATTERNS.md GraphListPane ARIA spec:
 *   role=listbox, aria-activedescendant, aria-selected on each option.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { configureAxe, toHaveNoViolations } from "jest-axe";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

expect.extend(toHaveNoViolations);

const axe = configureAxe({
  rules: {
    region: { enabled: false },
  },
});

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function renderWithClient(ui: ReactNode): ReturnType<typeof render> {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

// ---------------------------------------------------------------------------
// Fixture data
// ---------------------------------------------------------------------------

type GraphNode = {
  corpus_id: number;
  title: string;
  in_corpus: boolean;
  year: number | null;
  citationcount: number | null;
};

const SAMPLE_NODES: GraphNode[] = [
  { corpus_id: 1, title: "Attention Is All You Need", in_corpus: true, year: 2017, citationcount: 175000 },
  { corpus_id: 2, title: "BERT: Pre-training", in_corpus: false, year: 2019, citationcount: 90000 },
  { corpus_id: 3, title: "GPT-3 Language Models", in_corpus: true, year: 2020, citationcount: 45000 },
  { corpus_id: 4, title: "AlphaFold 2", in_corpus: false, year: 2021, citationcount: 20000 },
  { corpus_id: 5, title: "Diffusion Models", in_corpus: true, year: 2022, citationcount: 10000 },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GraphListPane — accessibility (GRAPH-06)", () => {
  it("list pane has no axe violations", async () => {
    // Import will fail (RED state) until 04-05 ships GraphListPane.
    const { GraphListPane } = await import("@/components/graph/GraphListPane").catch(
      (e) => {
        throw new Error(
          `GraphListPane not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    const { container } = renderWithClient(
      <GraphListPane nodes={SAMPLE_NODES} selectedCorpusId={null} onSelect={() => undefined} />,
    );

    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it("list pane has role=listbox and aria-label='Citation graph nodes'", async () => {
    const { GraphListPane } = await import("@/components/graph/GraphListPane").catch(
      (e) => {
        throw new Error(
          `GraphListPane not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    renderWithClient(
      <GraphListPane nodes={SAMPLE_NODES} selectedCorpusId={null} onSelect={() => undefined} />,
    );

    const listbox = screen.getByRole("listbox", { name: "Citation graph nodes" });
    expect(listbox).toBeDefined();
  });

  it("list pane is keyboard navigable — each row has role=option with aria-selected", async () => {
    const { GraphListPane } = await import("@/components/graph/GraphListPane").catch(
      (e) => {
        throw new Error(
          `GraphListPane not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    renderWithClient(
      <GraphListPane
        nodes={SAMPLE_NODES}
        selectedCorpusId={1}
        onSelect={() => undefined}
      />,
    );

    const listbox = screen.getByRole("listbox", { name: "Citation graph nodes" });

    // Container must be keyboard reachable
    expect(listbox.getAttribute("tabindex")).toBe("0");

    // Each item must be an option with aria-selected
    const options = screen.getAllByRole("option");
    expect(options.length).toBeGreaterThanOrEqual(SAMPLE_NODES.length);

    // The selected node (corpus_id=1) must have aria-selected=true
    const selectedOption = options.find(
      (el) => el.getAttribute("aria-selected") === "true",
    );
    expect(selectedOption).toBeDefined();
    expect(selectedOption?.textContent).toContain("Attention Is All You Need");
  });

  it("unselected rows have aria-selected=false", async () => {
    const { GraphListPane } = await import("@/components/graph/GraphListPane").catch(
      (e) => {
        throw new Error(
          `GraphListPane not yet implemented (expected RED). Original: ${String(e)}`,
        );
      },
    );

    renderWithClient(
      <GraphListPane
        nodes={SAMPLE_NODES}
        selectedCorpusId={1}
        onSelect={() => undefined}
      />,
    );

    const options = screen.getAllByRole("option");
    const unselected = options.filter(
      (el) => el.getAttribute("aria-selected") === "false",
    );
    // All nodes except corpus_id=1 must be aria-selected=false
    expect(unselected.length).toBeGreaterThanOrEqual(SAMPLE_NODES.length - 1);
  });
});
