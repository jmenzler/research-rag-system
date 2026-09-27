/**
 * GREEN — keymap registry + useGlobalHotkeys behavior tests (06-02 Task 1).
 *
 * Covers the five behavior cases:
 *   1. isInputFocused() returns true for input/textarea/select/contenteditable.
 *   2. A modifier binding (Cmd+K) fires its action with metaKey, only when not input-focused.
 *   3. A sequence binding (g c) fires only when both keys arrive within 1000ms.
 *   4. Esc Esc fires only on two Escapes within the window.
 *   5. When input is focused, no binding fires and the sequence buffer is flushed.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { isInputFocused, matchSequence } from "@/hooks/useGlobalHotkeys";
import type { KeyBinding, HotkeyContext } from "@/lib/keymap";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fireKeydown(key: string, opts: { metaKey?: boolean } = {}): void {
  document.dispatchEvent(
    new KeyboardEvent("keydown", {
      key,
      metaKey: opts.metaKey ?? false,
      bubbles: true,
      cancelable: true,
    }),
  );
}

function makeCtx(overrides: Partial<Record<string, ReturnType<typeof vi.fn>>> = {}): HotkeyContext {
  return {
    navigate: vi.fn(),
    setCommandOpen: vi.fn(),
    setHelpOpen: vi.fn(),
    setScopeDialogOpen: vi.fn(),
    toggleSidebar: vi.fn(),
    setSidebarOpen: vi.fn(),
    ...overrides,
  } as unknown as HotkeyContext;
}

// ---------------------------------------------------------------------------
// isInputFocused — behavior case 1
// ---------------------------------------------------------------------------

describe("isInputFocused", () => {
  it("returns false when no element is focused", () => {
    expect(isInputFocused()).toBe(false);
  });

  it("returns true for a focused <input>", () => {
    const el = document.createElement("input");
    document.body.appendChild(el);
    el.focus();
    expect(isInputFocused()).toBe(true);
    el.blur();
    document.body.removeChild(el);
  });

  it("returns true for a focused <textarea>", () => {
    const el = document.createElement("textarea");
    document.body.appendChild(el);
    el.focus();
    expect(isInputFocused()).toBe(true);
    el.blur();
    document.body.removeChild(el);
  });

  it("returns true for a focused <select>", () => {
    const el = document.createElement("select");
    document.body.appendChild(el);
    el.focus();
    expect(isInputFocused()).toBe(true);
    el.blur();
    document.body.removeChild(el);
  });

  it("returns true for a contentEditable element", () => {
    const el = document.createElement("div");
    el.setAttribute("contenteditable", "true");
    el.setAttribute("tabindex", "0");
    document.body.appendChild(el);
    el.focus();
    expect(isInputFocused()).toBe(true);
    el.blur();
    document.body.removeChild(el);
  });
});

// ---------------------------------------------------------------------------
// matchSequence helper
// ---------------------------------------------------------------------------

describe("matchSequence", () => {
  it("matches a two-key sequence exactly", () => {
    const binding: KeyBinding = {
      sequence: ["g", "c"],
      label: "Go to Chat",
      group: "Navigation",
      action: vi.fn(),
    };
    expect(matchSequence(["g", "c"], binding)).toBe(true);
  });

  it("does not match a partial sequence", () => {
    const binding: KeyBinding = {
      sequence: ["g", "c"],
      label: "Go to Chat",
      group: "Navigation",
      action: vi.fn(),
    };
    expect(matchSequence(["g"], binding)).toBe(false);
  });

  it("does not match a wrong sequence", () => {
    const binding: KeyBinding = {
      sequence: ["g", "c"],
      label: "Go to Chat",
      group: "Navigation",
      action: vi.fn(),
    };
    expect(matchSequence(["g", "g"], binding)).toBe(false);
  });

  it("matches Escape Escape sequence", () => {
    const binding: KeyBinding = {
      sequence: ["Escape", "Escape"],
      label: "Close overlay",
      group: "App",
      action: vi.fn(),
    };
    expect(matchSequence(["Escape", "Escape"], binding)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Behavior case 2: Modifier binding (Cmd+K)
// ---------------------------------------------------------------------------

describe("Modifier binding — Cmd+K", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(async () => {
    const { uninstallGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    uninstallGlobalHotkeys();
    vi.useRealTimers();
    vi.resetModules();
  });

  it("fires Cmd+K action when not input-focused", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const spy = vi.fn();
    const ctx = makeCtx({ setCommandOpen: spy });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("k", { metaKey: true });
    expect(spy).toHaveBeenCalledWith(true);
  });

  it("does NOT fire Cmd+K when an input is focused", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const spy = vi.fn();
    const ctx = makeCtx({ setCommandOpen: spy });
    installGlobalHotkeys(buildKeymap(ctx));

    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();

    fireKeydown("k", { metaKey: true });
    expect(spy).not.toHaveBeenCalled();

    input.blur();
    document.body.removeChild(input);
  });
});

// ---------------------------------------------------------------------------
// Behavior case 3: Sequence binding (g c) timing
// ---------------------------------------------------------------------------

describe("Sequence binding — g c within 1000ms", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(async () => {
    const { uninstallGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    uninstallGlobalHotkeys();
    vi.useRealTimers();
    vi.resetModules();
  });

  it("fires g c sequence when keys arrive within 1000ms", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const spy = vi.fn();
    const ctx = makeCtx({ navigate: spy });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("g");
    vi.advanceTimersByTime(200);
    fireKeydown("c");

    expect(spy).toHaveBeenCalledWith({ to: "/app/chat" });
  });

  it("does NOT fire g c when gap > 1000ms", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const spy = vi.fn();
    const ctx = makeCtx({ navigate: spy });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("g");
    vi.advanceTimersByTime(1100);
    fireKeydown("c");

    expect(spy).not.toHaveBeenCalledWith({ to: "/app/chat" });
  });
});

// ---------------------------------------------------------------------------
// Behavior case 4: Esc Esc sequence
// ---------------------------------------------------------------------------

describe("Esc Esc sequence", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(async () => {
    const { uninstallGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    uninstallGlobalHotkeys();
    vi.useRealTimers();
    vi.resetModules();
  });

  it("fires Esc Esc action when two Escapes arrive within 1000ms", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const setCommandOpen = vi.fn();
    const setHelpOpen = vi.fn();
    const ctx = makeCtx({ setCommandOpen, setHelpOpen });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("Escape");
    vi.advanceTimersByTime(200);
    fireKeydown("Escape");

    // With commandOpen/helpOpen not in ctx (they start closed), blur is called.
    // The important thing is that it does not throw and Esc Esc is handled in-shell.
    // setCommandOpen(false) is called as part of in-shell close.
    expect(setCommandOpen).toHaveBeenCalledWith(false);
  });

  it("does NOT fire Esc Esc when gap > 1000ms", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const setCommandOpen = vi.fn();
    const ctx = makeCtx({ setCommandOpen });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("Escape");
    vi.advanceTimersByTime(1100);
    fireKeydown("Escape");

    // After timeout reset, second Escape starts a NEW sequence — no Esc Esc fires
    expect(setCommandOpen).not.toHaveBeenCalledWith(false);
  });
});

// ---------------------------------------------------------------------------
// Behavior case 5: Input focused → no binding fires, buffer flushed
// ---------------------------------------------------------------------------

describe("Input focus guard — buffer flushed when input gains focus", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(async () => {
    const { uninstallGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    uninstallGlobalHotkeys();
    vi.useRealTimers();
    vi.resetModules();
  });

  it("does not fire g c when input gains focus between the two keys", async () => {
    const { buildKeymap } = await import("@/lib/keymap");
    const { installGlobalHotkeys } = await import("@/hooks/useGlobalHotkeys");
    const spy = vi.fn();
    const ctx = makeCtx({ navigate: spy });
    installGlobalHotkeys(buildKeymap(ctx));

    fireKeydown("g");

    const input = document.createElement("input");
    document.body.appendChild(input);
    input.focus();

    vi.advanceTimersByTime(200);
    fireKeydown("c"); // input is focused — buffer flushes, action does not fire

    expect(spy).not.toHaveBeenCalledWith({ to: "/app/chat" });

    input.blur();
    document.body.removeChild(input);
  });
});
