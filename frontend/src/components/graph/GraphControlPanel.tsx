/**
 * GraphControlPanel — gear-triggered Display & Forces modal (D-07, D-08, D-09).
 *
 * Placement: fixed inset-0 z-50, panel right-aligned, marginTop 56px.
 * Closes on: Esc key, backdrop click, × button.
 * Display sliders: live onChange → store setters (feed Sigma settings/reducers).
 * Forces sliders: onChange → store setters; the canvas FA2 loop reads them live
 *   each frame, so dragging re-tunes the running layout in real time.
 * Animate button: toggles FA2 start/stop via graphDisplayStore.animate flag.
 * Persistence: Zustand persist middleware in graphDisplayStore (key "rag-graph-display").
 *
 * Security: slider values clamped to [0,100] at store setter boundary (T-04.1-07).
 */
import {
  Play,
  RotateCcw,
  SlidersHorizontal,
  X as XIcon,
} from "lucide-react";
import {
  useEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactElement,
} from "react";
import { useGraphDisplayStore } from "@/state/graphDisplayStore";

const FOCUSABLE_SELECTOR =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface GraphControlPanelProps {
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Slider sub-component
// ---------------------------------------------------------------------------

interface SliderRowProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  onPointerUp?: () => void;
}

function SliderRow({
  label,
  value,
  onChange,
  onPointerUp,
}: SliderRowProps): ReactElement {
  const pct = value;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span style={{ color: "var(--p3-muted)" }}>{label}</span>
        <span
          className="tabular-nums"
          style={{ fontFamily: "var(--p3-font-mono)", color: "var(--p3-fg)" }}
        >
          {value}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={onPointerUp}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          accentColor: "var(--p3-primary)",
          background: `linear-gradient(to right, var(--p3-primary) 0%, var(--p3-primary) ${pct}%, var(--p3-border) ${pct}%, var(--p3-border) 100%)`,
        }}
        aria-label={label}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toggle sub-component (Arrows)
// ---------------------------------------------------------------------------

interface ToggleRowProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}

function ToggleRow({ label, value, onChange }: ToggleRowProps): ReactElement {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="flex items-center justify-between w-full text-[11px] py-1"
    >
      <span style={{ color: "var(--p3-muted)" }}>{label}</span>
      <span
        className="relative inline-block w-8 h-4 rounded-full transition-colors"
        style={{ backgroundColor: value ? "var(--p3-primary)" : "var(--p3-border)" }}
      >
        <span
          className="absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform"
          style={{ transform: value ? "translateX(16px)" : "translateX(0)" }}
        />
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

function SectionLabel({ label }: { label: string }): ReactElement {
  return (
    <div
      className="text-[10px] uppercase tracking-wider font-semibold"
      style={{ color: "var(--p3-muted-2)" }}
    >
      {label}
    </div>
  );
}

// ---------------------------------------------------------------------------
// GraphControlPanel
// ---------------------------------------------------------------------------

export function GraphControlPanel({
  onClose,
}: GraphControlPanelProps): ReactElement {
  const {
    nodeSize,    setNodeSize,
    linkThick,   setLinkThick,
    textFade,    setTextFade,
    arrowsOn,    setArrowsOn,
    centerForce, setCenterForce,
    repelForce,  setRepelForce,
    linkForce,   setLinkForce,
    linkDistance, setLinkDistance,
    animate,     setAnimate,
    reset,
  } = useGraphDisplayStore();

  const panelRef = useRef<HTMLDivElement>(null);

  // Esc closes the modal.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Move focus into the dialog on open and restore it to the trigger on close,
  // so keyboard/SR focus does not stay on the obscured canvas behind the scrim.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    return () => previouslyFocused?.focus();
  }, []);

  // Trap Tab within the dialog so focus cannot escape to the canvas behind it.
  const onKeyDown = (e: ReactKeyboardEvent): void => {
    if (e.key !== "Tab") return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusable = panel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
    if (focusable.length === 0) return;
    const first = focusable[0]!;
    const last = focusable[focusable.length - 1]!;
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === panel)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-end p-4"
      style={{ backgroundColor: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
      role="presentation"
    >
      {/* Modal panel — stop clicks propagating to scrim */}
      <div
        ref={panelRef}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        tabIndex={-1}
        className="w-[320px] rounded-lg shadow-2xl flex flex-col max-h-[calc(100vh-32px)] outline-none"
        style={{
          backgroundColor: "var(--p3-surface)",
          border: "1px solid var(--p3-border)",
          marginTop: 56,
        }}
        role="dialog"
        aria-modal="true"
        aria-label="Display and forces"
      >
        {/* Header */}
        <div
          className="flex items-center justify-between h-10 px-3"
          style={{ borderBottom: "1px solid var(--p3-border)" }}
        >
          <div
            className="flex items-center gap-2 text-[11px] uppercase tracking-wider"
            style={{ color: "var(--p3-muted)" }}
          >
            <SlidersHorizontal size={12} />
            Display &amp; Forces
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={reset}
              className="flex items-center gap-1 rounded px-1.5 py-1 text-[10px] uppercase tracking-wider transition-colors hover:text-white/80"
              style={{ color: "var(--p3-muted)" }}
              aria-label="Reset to defaults"
            >
              <RotateCcw size={12} />
              Reset
            </button>
            <button
              type="button"
              onClick={onClose}
              className="hover:text-white/80 p-1 rounded transition-colors"
              style={{ color: "var(--p3-muted)" }}
              aria-label="Close"
            >
              <XIcon size={14} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto px-4 py-4 flex flex-col gap-5">
          {/* Display section */}
          <div className="flex flex-col gap-2.5">
            <SectionLabel label="Display" />
            <SliderRow
              label="Node size"
              value={nodeSize}
              onChange={setNodeSize}
            />
            <SliderRow
              label="Link thickness"
              value={linkThick}
              onChange={setLinkThick}
            />
            <SliderRow
              label="Text-fade threshold"
              value={textFade}
              onChange={setTextFade}
            />
            <ToggleRow label="Arrows" value={arrowsOn} onChange={setArrowsOn} />
          </div>

          <div className="h-px" style={{ backgroundColor: "var(--p3-border)" }} />

          {/* Forces section */}
          <div className="flex flex-col gap-2.5">
            <SectionLabel label="Forces" />
            <SliderRow
              label="Center force"
              value={centerForce}
              onChange={setCenterForce}
            />
            <SliderRow
              label="Repel force"
              value={repelForce}
              onChange={setRepelForce}
            />
            <SliderRow
              label="Link force"
              value={linkForce}
              onChange={setLinkForce}
            />
            <SliderRow
              label="Link distance"
              value={linkDistance}
              onChange={setLinkDistance}
            />

            {/* Animate button */}
            <button
              type="button"
              onClick={() => setAnimate(!animate)}
              className="mt-1 inline-flex items-center justify-center gap-1.5 w-full h-8 rounded-md text-[12px] font-medium transition-colors"
              style={
                animate
                  ? {
                      backgroundColor: "var(--p3-surface)",
                      border: "1px solid var(--p3-border)",
                      color: "var(--p3-fg)",
                    }
                  : {
                      backgroundColor: "var(--p3-primary)",
                      color: "white",
                    }
              }
            >
              <Play size={12} />
              {animate ? "Stop" : "Animate"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
