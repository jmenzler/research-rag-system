import { useEffect } from "react";
import { matchSequence } from "@/lib/keymap";
import type { KeyBinding } from "@/lib/keymap";

export { matchSequence };

export function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el || el === document.body) return false;
  if (el instanceof HTMLInputElement) return true;
  if (el instanceof HTMLTextAreaElement) return true;
  if (el instanceof HTMLSelectElement) return true;
  if (
    el instanceof HTMLElement &&
    (el.isContentEditable || el.getAttribute("contenteditable") === "true")
  )
    return true;
  return false;
}

let sequenceBuffer: string[] = [];
let lastKeyTime = 0;
const SEQUENCE_WINDOW_MS = 1000;

let activeHandler: ((e: KeyboardEvent) => void) | null = null;

function makeHandler(bindings: KeyBinding[]): (e: KeyboardEvent) => void {
  return function handleKeydown(e: KeyboardEvent): void {
    if (isInputFocused()) {
      sequenceBuffer = [];
      return;
    }

    const now = Date.now();
    if (now - lastKeyTime > SEQUENCE_WINDOW_MS) {
      sequenceBuffer = [];
    }
    lastKeyTime = now;

    // Modifier bindings (meta combos)
    for (const binding of bindings) {
      if (binding.key && !binding.sequence) {
        const mods = binding.modifiers ?? [];
        const wantsMeta = mods.includes("meta");
        const wantsCtrl = mods.includes("ctrl");
        const wantsAlt = mods.includes("alt");
        const wantsShift = mods.includes("shift");

        if (
          binding.key === e.key &&
          wantsMeta === e.metaKey &&
          wantsCtrl === e.ctrlKey &&
          wantsAlt === e.altKey &&
          wantsShift === e.shiftKey &&
          mods.length > 0
        ) {
          e.preventDefault();
          binding.action();
          sequenceBuffer = [];
          return;
        }
      }
    }

    // Sequence bindings — push key into buffer and check for match
    sequenceBuffer.push(e.key);

    // Check if any sequence binding fully matches
    for (const binding of bindings) {
      if (binding.sequence && matchSequence(sequenceBuffer, binding)) {
        binding.action();
        sequenceBuffer = [];
        return;
      }
    }

    // Check if any sequence binding is still a potential prefix
    const isPotentialPrefix = bindings.some(
      (b) =>
        b.sequence &&
        b.sequence.length > sequenceBuffer.length &&
        b.sequence
          .slice(0, sequenceBuffer.length)
          .every((k, i) => k === sequenceBuffer[i]),
    );

    if (!isPotentialPrefix) {
      // No sequence binding can start with this buffer — reset
      // But check single-key (no modifier) bindings first
      const last = e.key;
      for (const binding of bindings) {
        if (
          binding.key === last &&
          !binding.sequence &&
          (!binding.modifiers || binding.modifiers.length === 0)
        ) {
          binding.action();
          sequenceBuffer = [];
          return;
        }
      }
      sequenceBuffer = [];
    }
  };
}

export function installGlobalHotkeys(bindings: KeyBinding[]): void {
  if (activeHandler) {
    document.removeEventListener("keydown", activeHandler);
  }
  sequenceBuffer = [];
  lastKeyTime = 0;
  activeHandler = makeHandler(bindings);
  document.addEventListener("keydown", activeHandler);
}

export function uninstallGlobalHotkeys(): void {
  if (activeHandler) {
    document.removeEventListener("keydown", activeHandler);
    activeHandler = null;
  }
  sequenceBuffer = [];
}

export function useGlobalHotkeys(bindings: KeyBinding[]): void {
  useEffect(() => {
    installGlobalHotkeys(bindings);
    return () => {
      uninstallGlobalHotkeys();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
