/**
 * Wave-1 RED test for <FilterChipRow> (D-12, Pitfall 9).
 *
 * Implementation lands in Plan 02-12 (frontend/src/components/history/FilterChipRow.tsx
 * + the URL search schema on the chat-list layout route).
 *
 * Behavioural contract:
 *   - URL search params are the source of truth — read via TanStack Router useSearch.
 *   - zod transform: `?collections=trading,ecology` → `["trading","ecology"]`
 *   - `?retrievers=milvus` → `["milvus"]`
 *   - `?archived=1` → `true` via z.coerce.boolean
 *   - empty `collections=` is treated as all four
 *   - ToggleGroup type="multiple" for collections + retrievers
 *   - Toggle for archived (boolean)
 *   - Toggling a chip updates the URL via TanStack Router navigate
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

// Mock TanStack Router so the component can use useSearch / useNavigate without
// requiring a full route tree. The test installs custom state per case.
let currentSearch: Record<string, unknown> = {};
const navigateSpy = vi.fn();
vi.mock("@tanstack/react-router", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "@tanstack/react-router",
  );
  return {
    ...actual,
    useSearch: () => currentSearch,
    useNavigate: () => navigateSpy,
  };
});

function renderWithClient(ui: ReactNode): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  toastMock.mockReset();
  navigateSpy.mockReset();
  currentSearch = {};
});

describe("<FilterChipRow> + URL search schema — RED tests until Plan 02-12", () => {
  it("?collections=trading,ecology reads as ['trading','ecology'] via zod transform", async () => {
    const { chatListSearchSchema } = await import(
      "@/components/history/FilterChipRow"
    );

    const parsed = chatListSearchSchema.parse({ collections: "trading,ecology" });
    expect(parsed.collections).toEqual(["trading", "ecology"]);
  });

  it("?retrievers=milvus reads as ['milvus']", async () => {
    const { chatListSearchSchema } = await import(
      "@/components/history/FilterChipRow"
    );

    const parsed = chatListSearchSchema.parse({ retrievers: "milvus" });
    expect(parsed.retrievers).toEqual(["milvus"]);
  });

  it("?archived=1 reads as true via z.coerce.boolean", async () => {
    const { chatListSearchSchema } = await import(
      "@/components/history/FilterChipRow"
    );

    const parsed = chatListSearchSchema.parse({ archived: "1" });
    expect(parsed.archived).toBe(true);
  });

  it("empty collections= is treated as all four", async () => {
    const { chatListSearchSchema, normalizeFilters } = await import(
      "@/components/history/FilterChipRow"
    );

    const parsed = chatListSearchSchema.parse({ collections: "" });
    expect(parsed.collections).toEqual([]); // raw zod result
    // The component-level normaliser treats empty as all four (D-03 ergonomics).
    const effective = normalizeFilters(parsed);
    expect(effective.collections).toEqual(
      expect.arrayContaining(["trading", "ecology", "notes", "system"]),
    );
    expect(effective.collections).toHaveLength(4);
  });

  it("ToggleGroup type='multiple' for collections + retrievers", async () => {
    const { FilterChipRow } = await import(
      "@/components/history/FilterChipRow"
    );

    const { container } = render(
      <QueryClientProvider
        client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
      >
        <FilterChipRow />
      </QueryClientProvider>,
    );

    // Radix ToggleGroup root applies a data-orientation attribute and either
    // role="group" (multiple) or role="radiogroup" (single). For type=multiple
    // it must NOT be radiogroup.
    const groups = container.querySelectorAll(
      "[data-slot=toggle-group], [role=group]",
    );
    expect(groups.length).toBeGreaterThanOrEqual(2);
    container
      .querySelectorAll("[data-slot=toggle-group]")
      .forEach((el) => expect(el.getAttribute("role")).not.toBe("radiogroup"));
  });

  it("Toggle for archived (boolean)", async () => {
    const { FilterChipRow } = await import(
      "@/components/history/FilterChipRow"
    );

    renderWithClient(<FilterChipRow />);

    const archived = await screen.findByRole("button", { name: /show archived/i });
    expect(archived).toBeTruthy();
    expect(archived.getAttribute("aria-pressed")).toBeDefined();
  });

  it("toggling a chip updates the URL via TanStack Router navigate", async () => {
    const { FilterChipRow } = await import(
      "@/components/history/FilterChipRow"
    );

    renderWithClient(<FilterChipRow />);
    const user = userEvent.setup();

    const tradingChip = await screen.findByRole("button", { name: /trading/i });
    await user.click(tradingChip);

    expect(navigateSpy).toHaveBeenCalled();
    const call = navigateSpy.mock.calls[0]?.[0];
    expect(call).toBeDefined();
    // navigate({ search: (prev) => ... }) — verify the produced search includes
    // trading, regardless of whether the arg is a function or a literal.
    const next =
      typeof call.search === "function" ? call.search({}) : call.search;
    const collections = next?.collections;
    if (Array.isArray(collections)) {
      expect(collections).toContain("trading");
    } else {
      expect(String(collections ?? "")).toContain("trading");
    }
  });
});
