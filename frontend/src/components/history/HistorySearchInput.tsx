/**
 * HistorySearchInput (HIST-04).
 *
 * 200ms-debounced input. URL search params are the source of truth — local
 * input value is mirrored into `?q=` after the debounce expires. The
 * Loader2 spinner inside the right padding signals "fetching" so the user
 * does not type into a dead UI while keepPreviousData shows stale results.
 *
 * Like FilterChipRow this component reads `useSearch` / `useNavigate` via
 * safe wrappers so it stays render-clean in the unit-test render tree
 * (router context absent). Production callers mount it inside the layout
 * route where the router is live.
 */
import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { Loader2, Search, X } from "lucide-react";
import { useNavigate, useSearch } from "@tanstack/react-router";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

function useQuerySafe(): string {
  try {
    const search = useSearch({ strict: false }) as
      | { q?: unknown }
      | undefined;
    return typeof search?.q === "string" ? search.q : "";
  } catch {
    return "";
  }
}

function useNavigateSafe(): ReturnType<typeof useNavigate> | null {
  try {
    return useNavigate();
  } catch {
    return null;
  }
}

export interface HistorySearchInputProps {
  /** Override the URL-derived query value (used by unit tests). */
  value?: string;
  /** Indicator for the spinner — caller supplies fetching state. */
  isFetching?: boolean;
  /** Optional explicit change handler that bypasses URL navigation. */
  onChange?: (q: string) => void;
}

export function HistorySearchInput(
  props: HistorySearchInputProps = {},
): ReactElement {
  const urlQ = useQuerySafe();
  const navigate = useNavigateSafe();
  const externalValue = props.value ?? urlQ;

  // Local input state; debounced into the URL.
  const [local, setLocal] = useState(externalValue);

  // Re-sync local when the canonical (URL) value moves externally.
  useEffect(() => {
    setLocal(externalValue);
  }, [externalValue]);

  // Debounce: 200ms after last keystroke → push to URL.
  useEffect(() => {
    if (local === externalValue) return;
    const t = setTimeout(() => {
      if (props.onChange !== undefined) {
        props.onChange(local);
        return;
      }
      if (navigate !== null) {
        navigate({
          search: ((prev: Record<string, unknown>) => ({
            ...prev,
            q: local,
          })) as unknown as never,
        });
      }
    }, 200);
    return () => clearTimeout(t);
  }, [local, externalValue, props, navigate]);

  const handleClear = (): void => {
    setLocal("");
    if (props.onChange !== undefined) {
      props.onChange("");
      return;
    }
    if (navigate !== null) {
      navigate({
        search: ((prev: Record<string, unknown>) => ({
          ...prev,
          q: "",
        })) as unknown as never,
      });
    }
  };

  return (
    <div className="relative px-2">
      <Search className="text-muted-foreground absolute top-1/2 left-4 size-3.5 -translate-y-1/2" />
      <Input
        type="search"
        placeholder="Search chats…"
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        className="pr-8 pl-7 text-sm"
        aria-label="Search chats"
      />
      {props.isFetching === true && (
        <Loader2 className="text-muted-foreground absolute top-1/2 right-9 size-3 -translate-y-1/2 animate-spin" />
      )}
      {local.length > 0 && (
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-1/2 right-3 size-6 -translate-y-1/2"
          onClick={handleClear}
          aria-label="Clear search"
        >
          <X className="size-3" />
        </Button>
      )}
    </div>
  );
}
