/**
 * ChatListItem — one row in the sidebar list (HIST-01, HIST-02, HIST-03).
 *
 * Layout: title + per-row ⋯ DropdownMenu with items in EXACT order
 *   1. Rename…
 *   2. Archive / Unarchive
 *   <separator>
 *   3. Delete…
 * (UI-SPEC Copywriting Contract, D-15)
 *
 * Active-row indicator: 4px primary bar on the left edge when `isActive`.
 * The active state is prop-driven so this component stays unit-testable
 * outside a TanStack Router context (HistorySidebar.test.tsx passes
 * `activeChatId="active-id"` directly).
 *
 * Navigation uses a bare `<a href>` rather than TanStack Router's
 * `NavLink` for the same reason — tests render in isolation. Production
 * routing still works because the URL is a top-level anchor target.
 */
import { Archive, ArchiveRestore, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import type { ReactElement } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cn";

export interface ChatRow {
  id: string;
  title: string | null;
  retriever: string;
  collections: string;
  version: number;
  created_at: string;
  updated_at: string;
  archived?: boolean;
  snippet?: string;
}

export interface ChatListItemProps {
  chat: ChatRow;
  isActive?: boolean;
  onRename?: (chat: ChatRow) => void;
  onDelete?: (chat: ChatRow) => void;
  onArchiveToggle?: (chat: ChatRow) => void;
}

export function ChatListItem(props: ChatListItemProps): ReactElement {
  const { chat, isActive } = props;
  const title = chat.title ?? "Untitled chat";

  return (
    <div
      data-active={isActive ? "true" : undefined}
      aria-current={isActive ? "page" : undefined}
      className={cn(
        "group relative mx-2 flex items-center gap-1 rounded-md px-2.5 py-[7px] transition-colors duration-150 hover:bg-white/5 motion-reduce:transition-none",
        isActive &&
          "bg-white/[0.06] shadow-[inset_2px_0_0_var(--p3-primary)]",
      )}
    >
      <a
        href={`/app/chat/${chat.id}`}
        className={cn(
          "flex-1 truncate text-[13px]",
          isActive
            ? "text-[var(--p3-fg)]"
            : "text-[var(--p3-fg-2)]",
        )}
      >
        {title}
      </a>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="size-7 opacity-0 group-hover:opacity-100 data-[state=open]:opacity-100"
            aria-label={`Chat actions for ${title}`}
          >
            <MoreHorizontal className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => props.onRename?.(chat)}>
            <Pencil className="size-4" /> Rename…
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => props.onArchiveToggle?.(chat)}>
            {chat.archived === true ? (
              <>
                <ArchiveRestore className="size-4" /> Unarchive
              </>
            ) : (
              <>
                <Archive className="size-4" /> Archive
              </>
            )}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            variant="destructive"
            onSelect={() => props.onDelete?.(chat)}
          >
            <Trash2 className="size-4" /> Delete…
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
