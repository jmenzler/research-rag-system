/**
 * GraphToolbar — URL-driven toolbar for the citation graph explorer.
 *
 * All toolbar state lives in the URL (URL-as-source-of-truth rule from PATTERNS).
 * Reads from Route.useSearch() via a safe wrapper (no-throw in unit tests).
 * Writes via useNavigate({ from: "/app/_layout/graph" }).
 *
 * Controls:
 *   - Collection switcher (trading default; non-trading disabled in v1 per D-03)
 *   - Seed search input → fires seedSearchQuery; hitting a result seeds a walk
 *   - Depth stepper 1–5 (aria-labels per mockup)
 *   - Direction segmented control (references | citers | both)
 *   - Year-from / year-to inputs
 *   - Min-citations input
 *   - List view toggle (writes listOpen to graphStore + URL param)
 */
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { List, Minus, Plus, Search, Settings } from "lucide-react";
import { useState, type FormEvent, type ReactElement } from "react";

import { collectionsQuery, seedSearchQuery } from "@/queries/graph";
import type { SeedSearchHit } from "@/queries/graph";
import { useGraphStore } from "@/state/graphStore";

// ---------------------------------------------------------------------------
// Router-safe hooks (swallow errors in unit tests without a router context)
// ---------------------------------------------------------------------------

type GraphSearch = {
  node?: number;
  collection: string;
  depth: number;
  direction: "references" | "citers" | "both";
  yearFrom?: number;
  yearTo?: number;
  minCites: number;
  listOpen: boolean;
};

function useSearchSafe(): Partial<GraphSearch> {
  try {
    return useSearch({ strict: false }) as Partial<GraphSearch>;
  } catch {
    return {};
  }
}

function useNavigateSafe(): ReturnType<typeof useNavigate> | null {
  try {
    return useNavigate();
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface DepthStepperProps {
  depth: number;
  onChange: (next: number) => void;
}

function DepthStepper({ depth, onChange }: DepthStepperProps): ReactElement {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        aria-label="Decrease depth"
        onClick={() => onChange(Math.max(1, depth - 1))}
        className="flex h-6 w-6 items-center justify-center rounded text-[var(--p3-fg-2)] hover:bg-white/10 hover:text-[var(--p3-fg)] disabled:opacity-40"
        disabled={depth <= 1}
      >
        <Minus className="size-3" />
      </button>
      <span className="min-w-[1.25rem] text-center font-mono text-sm text-[var(--p3-fg)]">
        {depth}
      </span>
      <button
        type="button"
        aria-label="Increase depth"
        onClick={() => onChange(Math.min(5, depth + 1))}
        className="flex h-6 w-6 items-center justify-center rounded text-[var(--p3-fg-2)] hover:bg-white/10 hover:text-[var(--p3-fg)] disabled:opacity-40"
        disabled={depth >= 5}
      >
        <Plus className="size-3" />
      </button>
    </div>
  );
}

interface DirectionSegmentProps {
  direction: string;
  onChange: (next: "references" | "citers" | "both") => void;
}

function DirectionSegment({
  direction,
  onChange,
}: DirectionSegmentProps): ReactElement {
  const options: Array<{ value: "references" | "citers" | "both"; label: string }> = [
    { value: "references", label: "refs" },
    { value: "both", label: "both" },
    { value: "citers", label: "cited" },
  ];
  return (
    <div className="flex rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)]">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          title={opt.value}
          className={
            "px-2 py-0.5 font-mono text-xs transition-colors " +
            (direction === opt.value
              ? "bg-[var(--p3-primary-soft)] text-[var(--p3-primary-text)]"
              : "text-[var(--p3-fg-2)] hover:text-[var(--p3-fg)]")
          }
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

interface SeedSearchDropdownProps {
  hits: SeedSearchHit[];
  onSelect: (hit: SeedSearchHit) => void;
}

function SeedSearchDropdown({
  hits,
  onSelect,
}: SeedSearchDropdownProps): ReactElement {
  return (
    <div className="absolute top-full left-0 z-50 mt-1 w-full max-h-64 overflow-y-auto rounded border border-[var(--p3-border)] bg-[var(--p3-bg)] shadow-lg">
      {hits.map((hit) => (
        <button
          key={hit.corpusId}
          type="button"
          onClick={() => onSelect(hit)}
          className="flex w-full flex-col gap-0.5 px-3 py-2 text-left hover:bg-[var(--p3-bg-2)] border-b border-[var(--p3-border)] last:border-0"
        >
          <span className="text-sm text-[var(--p3-fg)] line-clamp-1">
            {hit.title}
          </span>
          <span className="font-mono text-xs text-[var(--p3-muted)]">
            {hit.arxivId ?? hit.doi ?? `corpusId:${hit.corpusId}`}
          </span>
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// GraphToolbarProps
// ---------------------------------------------------------------------------

export interface GraphToolbarProps {
  /** Called when the user selects a seed search hit to start a walk. */
  onSeedSelect?: (hit: SeedSearchHit) => void;
  /** Called to open the Display & Forces settings modal. */
  onSettingsOpen?: () => void;
}

// ---------------------------------------------------------------------------
// GraphToolbar
// ---------------------------------------------------------------------------

export function GraphToolbar({ onSeedSelect, onSettingsOpen }: GraphToolbarProps): ReactElement {
  const search = useSearchSafe();
  const navigate = useNavigateSafe();

  const collection = search.collection ?? "trading";
  const depth = search.depth ?? 2;
  const direction = (search.direction as "references" | "citers" | "both") ?? "both";
  const yearFrom = search.yearFrom;
  const yearTo = search.yearTo;
  const minCites = search.minCites ?? 0;
  const listOpen = search.listOpen ?? false;

  const [seedQuery, setSeedQuery] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const setListOpen = useGraphStore((s) => s.setListOpen);

  const collectionsResult = useQuery(collectionsQuery());
  const seedResult = useQuery(seedSearchQuery(seedQuery));

  const updateSearch = <K extends keyof GraphSearch>(
    key: K,
    value: GraphSearch[K],
  ): void => {
    if (navigate === null) return;
    navigate({
      search: ((prev: Partial<GraphSearch>) =>
        ({ ...prev, [key]: value })) as unknown as never,
    });
  };

  const handleSeedSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (seedQuery.trim().length > 0) {
      setDropdownOpen(true);
    }
  };

  const handleSeedSelect = (hit: SeedSearchHit): void => {
    setDropdownOpen(false);
    setSeedQuery(hit.title);
    onSeedSelect?.(hit);
  };

  const handleListToggle = (): void => {
    const next = !listOpen;
    setListOpen(next);
    updateSearch("listOpen", next);
  };

  const collections = collectionsResult.data ?? [
    { name: "trading", hasGraphData: true },
  ];

  // seed-search resolves to a single best hit (backend SeedSearchHit); wrap it
  // as a 0-or-1 list for the dropdown.
  const seedHits = dropdownOpen && seedResult.data ? [seedResult.data] : [];

  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-[var(--p3-border)] bg-[var(--p3-bg)] px-3 py-2">
      {/* Context label + Collection switcher */}
      <div className="flex items-center gap-1.5">
        <span
          className="text-[10px] uppercase tracking-[0.08em]"
          style={{ color: "var(--p3-muted-2)" }}
        >
          Graph
        </span>
        <div className="flex rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)]">
          {collections.map((c) => {
            const active = c.name === collection;
            const disabled = !c.hasGraphData && c.name !== "trading";
            return (
              <button
                key={c.name}
                type="button"
                disabled={disabled}
                onClick={() => {
                  if (!disabled) updateSearch("collection", c.name);
                }}
                className={
                  "px-2 py-0.5 font-mono text-xs transition-colors " +
                  (active
                    ? "bg-[var(--p3-primary-soft)] text-[var(--p3-primary-text)]"
                    : disabled
                      ? "cursor-not-allowed opacity-40 text-[var(--p3-fg-2)]"
                      : "text-[var(--p3-fg-2)] hover:text-[var(--p3-fg)]")
                }
                title={disabled ? `${c.name}: no graph data` : c.name}
              >
                {c.name}
              </button>
            );
          })}
        </div>
      </div>

      {/* Divider */}
      <div className="h-5 w-px bg-[var(--p3-border)]" aria-hidden />

      {/* Seed search */}
      <div className="relative min-w-[220px] flex-1" style={{ maxWidth: 420 }}>
        <form onSubmit={handleSeedSubmit}>
          <div className="flex h-8 items-center gap-1.5 rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-2">
            <Search className="size-3.5 shrink-0 text-[var(--p3-muted)]" />
            <input
              type="text"
              value={seedQuery}
              onChange={(e) => {
                setSeedQuery(e.target.value);
                setDropdownOpen(e.target.value.trim().length > 0);
              }}
              onBlur={() => setTimeout(() => setDropdownOpen(false), 150)}
              placeholder="Search a paper by title, DOI, arXiv…"
              className="flex-1 bg-transparent font-mono text-xs text-[var(--p3-fg)] placeholder:text-[var(--p3-muted)] outline-none"
            />
          </div>
        </form>
        {seedHits.length > 0 && (
          <SeedSearchDropdown hits={seedHits} onSelect={handleSeedSelect} />
        )}
      </div>

      {/* Divider */}
      <div className="h-5 w-px bg-[var(--p3-border)]" aria-hidden />

      {/* Depth stepper */}
      <div className="flex items-center gap-1.5">
        <span
          className="text-[10px] uppercase tracking-[0.08em]"
          style={{ color: "var(--p3-muted-2)" }}
        >
          Depth
        </span>
        <DepthStepper
          depth={depth}
          onChange={(next) => updateSearch("depth", next)}
        />
      </div>

      {/* Direction segmented control */}
      <div className="flex items-center gap-1.5">
        <span
          className="text-[10px] uppercase tracking-[0.08em]"
          style={{ color: "var(--p3-muted-2)" }}
        >
          Direction
        </span>
        <DirectionSegment
          direction={direction}
          onChange={(next) => updateSearch("direction", next)}
        />
      </div>

      {/* Year range */}
      <div className="flex items-center gap-1">
        <span
          className="text-[10px] uppercase tracking-[0.08em]"
          style={{ color: "var(--p3-muted-2)" }}
        >
          Year
        </span>
        <input
          type="number"
          aria-label="Year from"
          value={yearFrom ?? ""}
          onChange={(e) =>
            updateSearch(
              "yearFrom",
              e.target.value ? Number(e.target.value) : undefined,
            )
          }
          placeholder="2000"
          className="w-[52px] rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-1.5 py-0.5 font-mono text-xs text-[var(--p3-fg)] outline-none focus:border-[var(--p3-primary)]"
        />
        <span
          className="text-xs"
          style={{ color: "var(--p3-muted)" }}
        >
          –
        </span>
        <input
          type="number"
          aria-label="Year to"
          value={yearTo ?? ""}
          onChange={(e) =>
            updateSearch(
              "yearTo",
              e.target.value ? Number(e.target.value) : undefined,
            )
          }
          placeholder="2026"
          className="w-[52px] rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-1.5 py-0.5 font-mono text-xs text-[var(--p3-fg)] outline-none focus:border-[var(--p3-primary)]"
        />
      </div>

      {/* Min citations */}
      <div className="flex items-center gap-1.5">
        <span
          className="text-[10px] uppercase tracking-[0.08em]"
          style={{ color: "var(--p3-muted-2)" }}
        >
          Min cites
        </span>
        <input
          type="number"
          aria-label="Minimum citations"
          value={minCites}
          min={0}
          onChange={(e) =>
            updateSearch("minCites", Math.max(0, Number(e.target.value) || 0))
          }
          className="w-[56px] rounded border border-[var(--p3-border)] bg-[var(--p3-bg-2)] px-1.5 py-0.5 font-mono text-xs text-[var(--p3-fg)] outline-none focus:border-[var(--p3-primary)]"
        />
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* List view toggle */}
      <button
        type="button"
        onClick={handleListToggle}
        aria-pressed={listOpen}
        title="Toggle list view"
        className={
          "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md text-[11px] transition-colors " +
          (listOpen
            ? "bg-[var(--p3-primary-soft)] text-[var(--p3-fg)]"
            : "text-[var(--p3-muted)] hover:bg-white/5 hover:text-[var(--p3-fg)]")
        }
        style={{
          border: `1px solid ${listOpen ? "var(--p3-primary-edge)" : "var(--p3-border)"}`,
        }}
      >
        <List className="size-3" />
        <span>List view</span>
      </button>

      {/* Divider before settings gear */}
      <div className="h-5 w-px bg-[var(--p3-border)]" aria-hidden />

      {/* Settings gear — opens Display & Forces modal */}
      <button
        type="button"
        onClick={onSettingsOpen}
        aria-label="Display and forces settings"
        title="Display and forces settings"
        className="inline-flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-white/5"
        style={{
          border: "1px solid var(--p3-border)",
          color: "var(--p3-muted)",
        }}
      >
        <Settings size={13} />
      </button>
    </div>
  );
}
