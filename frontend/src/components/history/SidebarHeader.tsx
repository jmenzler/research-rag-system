/**
 * SidebarHeader — logo + collapse toggle + new-chat button (HIST-01, D-10).
 *
 * Collapse state lives in useUiStore.sidebarOpen (persisted via Zustand
 * persist middleware). The collapsed icon-rail variant renders only the
 * toggle + "New chat" plus icon; the expanded variant renders the full
 * "New chat" label + collapse button with the close icon.
 */
import {
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
} from "lucide-react";
import type { ReactElement } from "react";

import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";

export interface SidebarHeaderProps {
  /** Click handler for the "New chat" button. */
  onNewChat?: (() => void) | undefined;
}

export function SidebarHeader(props: SidebarHeaderProps): ReactElement {
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  if (!sidebarOpen) {
    // Icon-rail (3rem) variant.
    return (
      <div className="flex flex-col items-center gap-2 px-2.5 py-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          aria-label="Expand sidebar"
          className="size-7 text-[var(--p3-muted)] hover:bg-white/5 hover:text-[var(--p3-fg)]"
        >
          <PanelLeftOpen className="size-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={props.onNewChat}
          aria-label="New chat"
          className="size-9 rounded-md border border-[var(--p3-border)] bg-white/[0.04] text-[var(--p3-fg)] hover:border-[var(--p3-primary-edge)] hover:bg-white/[0.07]"
        >
          <Plus className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 px-3 pt-4">
      <div className="flex h-7 items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-[13px] font-semibold tracking-[0.01em] text-[var(--p3-fg)]">
          <span
            className="size-2 rounded-full bg-[var(--p3-ok)] shadow-[0_0_0_3px_var(--p3-ok-soft)]"
            aria-hidden="true"
          />
          rag-system
        </span>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          aria-label="Collapse sidebar"
          className="size-7 text-[var(--p3-muted)] hover:bg-white/5 hover:text-[var(--p3-fg)]"
        >
          <PanelLeftClose className="size-4" />
        </Button>
      </div>
      <Button
        variant="ghost"
        onClick={props.onNewChat}
        className="h-9 w-full justify-start gap-2.5 rounded-md border border-[var(--p3-border)] bg-white/[0.04] px-3 text-[13px] font-medium text-[var(--p3-fg)] hover:border-[var(--p3-primary-edge)] hover:bg-white/[0.07]"
      >
        <Plus className="size-[15px]" /> New chat
        <span className="ml-auto rounded-sm border border-[var(--p3-border)] bg-black/30 px-1.5 py-px font-mono text-[11px] text-[var(--p3-muted)]">
          ⌘N
        </span>
      </Button>
    </div>
  );
}
