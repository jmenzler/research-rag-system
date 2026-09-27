import type { NavigateFn } from "@tanstack/react-router";
import type { ScopeDialogMode } from "@/state/uiStore";

export interface HotkeyContext {
  navigate: NavigateFn;
  setCommandOpen: (open: boolean) => void;
  setHelpOpen: (open: boolean) => void;
  setScopeDialogOpen: (mode: ScopeDialogMode) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export interface KeyBinding {
  key?: string;
  modifiers?: string[];
  sequence?: string[];
  label: string;
  group: "Navigation" | "Chat" | "Graph" | "App";
  action: () => void;
}

export function buildKeymap(ctx: HotkeyContext): KeyBinding[] {
  return [
    {
      key: "k",
      modifiers: ["meta"],
      label: "Open command palette",
      group: "App",
      action: () => ctx.setCommandOpen(true),
    },
    {
      key: "n",
      modifiers: ["meta"],
      label: "New chat",
      group: "Chat",
      action: () => ctx.setScopeDialogOpen("create"),
    },
    {
      key: "b",
      modifiers: ["meta"],
      label: "Toggle sidebar",
      group: "App",
      action: () => ctx.toggleSidebar(),
    },
    {
      key: "/",
      modifiers: ["meta"],
      label: "Keyboard shortcuts",
      group: "App",
      action: () => ctx.setHelpOpen(true),
    },
    {
      key: "[",
      label: "Previous citation",
      group: "Chat",
      action: () =>
        window.dispatchEvent(new CustomEvent("hotkey:citation-prev")),
    },
    {
      key: "]",
      label: "Next citation",
      group: "Chat",
      action: () =>
        window.dispatchEvent(new CustomEvent("hotkey:citation-next")),
    },
    {
      sequence: ["g", "c"],
      label: "Go to Chat",
      group: "Navigation",
      action: () => ctx.navigate({ to: "/app/chat" }),
    },
    {
      sequence: ["g", "h"],
      label: "Go to History",
      group: "Navigation",
      action: () => ctx.setSidebarOpen(true),
    },
    {
      sequence: ["g", "g"],
      label: "Go to Graph",
      group: "Navigation",
      action: () => ctx.navigate({ to: "/app/graph" }),
    },
    {
      sequence: ["Escape", "Escape"],
      label: "Close overlay / blur",
      group: "App",
      action: () => {
        ctx.setCommandOpen(false);
        ctx.setHelpOpen(false);
        (document.activeElement as HTMLElement | null)?.blur?.();
      },
    },
    {
      key: "+",
      label: "Graph zoom in",
      group: "Graph",
      action: () =>
        window.dispatchEvent(new CustomEvent("hotkey:graph-zoom-in")),
    },
    {
      key: "-",
      label: "Graph zoom out",
      group: "Graph",
      action: () =>
        window.dispatchEvent(new CustomEvent("hotkey:graph-zoom-out")),
    },
    {
      key: "f",
      label: "Graph fit",
      group: "Graph",
      action: () => window.dispatchEvent(new CustomEvent("hotkey:graph-fit")),
    },
  ];
}

export function groupKeymap(
  bindings: KeyBinding[],
): Record<string, KeyBinding[]> {
  const result: Record<string, KeyBinding[]> = {};
  for (const b of bindings) {
    const g = b.group;
    if (!result[g]) result[g] = [];
    result[g].push(b);
  }
  return result;
}

export function matchSequence(buffer: string[], binding: KeyBinding): boolean {
  if (!binding.sequence) return false;
  if (buffer.length !== binding.sequence.length) return false;
  return binding.sequence.every((k, i) => k === buffer[i]);
}

export function formatBinding(binding: KeyBinding): string {
  if (binding.sequence) {
    return binding.sequence
      .map((k) => (k === "Escape" ? "Esc" : k))
      .join(" ");
  }
  const mods = binding.modifiers ?? [];
  const parts: string[] = [];
  if (mods.includes("meta")) parts.push("⌘");
  if (mods.includes("ctrl")) parts.push("Ctrl");
  if (mods.includes("alt")) parts.push("Alt");
  if (mods.includes("shift")) parts.push("⇧");
  if (binding.key) parts.push(binding.key.toUpperCase());
  return parts.join("");
}
