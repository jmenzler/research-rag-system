/**
 * SidebarNav — the three Phase 3 section nav buttons (Chats / Corpus /
 * Queries) that sit between the SidebarHeader and the chat-history list.
 *
 * Rendered inside HistorySidebar so the chat-history list ALWAYS appears
 * below it and chat routing stays intact. Styled with Tailwind/shadcn-slate
 * (NOT the p3 token layer) so the shared sidebar's existing look isn't
 * regressed — the p3 surfaces adopt the p3 look only inside their own
 * `.p3` content panes.
 *
 * Navigation + active-state read through the SAME router-context try/catch
 * pattern the rest of the history sidebar uses (FilterChipRow.useNavigateSafe)
 * so the sidebar still mounts in router-less unit tests.
 */
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { Activity, GitBranch, Library, MessageSquare } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactElement } from "react";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

const ITEMS: NavItem[] = [
  { to: "/app/chat", label: "Chats", icon: MessageSquare },
  { to: "/app/corpus", label: "Corpus", icon: Library },
  { to: "/app/queries", label: "Queries", icon: Activity },
  { to: "/app/graph", label: "Graph", icon: GitBranch },
];

function useCurrentPath(): string {
  try {
    return useRouterState({ select: (s) => s.location.pathname });
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

export function SidebarNav(): ReactElement {
  const pathname = useCurrentPath();
  const navigate = useNavigateSafe();

  return (
    <nav className="flex flex-col gap-0.5 px-2" aria-label="Sections">
      {ITEMS.map((item) => {
        const ItemIcon = item.icon;
        const active = pathname.startsWith(item.to);
        return (
          <button
            key={item.to}
            type="button"
            aria-current={active ? "page" : undefined}
            onClick={() => {
              if (navigate !== null) void navigate({ to: item.to });
            }}
            className={
              "flex h-9 items-center gap-2.5 rounded-md px-2.5 text-left text-[13px] font-medium transition-colors duration-150 motion-reduce:transition-none " +
              (active
                ? "bg-[var(--p3-primary-soft)] text-[var(--p3-fg)] shadow-[inset_2px_0_0_var(--p3-primary)] [&_svg]:text-[var(--p3-primary-text)]"
                : "text-[var(--p3-fg-2)] hover:bg-white/5 hover:text-[var(--p3-fg)] [&_svg]:text-[var(--p3-muted)] hover:[&_svg]:text-[var(--p3-fg-2)]")
            }
          >
            <ItemIcon className="size-4 shrink-0 transition-colors duration-150 motion-reduce:transition-none" />
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
