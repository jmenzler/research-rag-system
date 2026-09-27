/**
 * GraphLegend — glassmorphic color-swatch legend for the citation graph.
 *
 * Positioned absolute bottom-3 right-3 over the canvas. Five rows matching the
 * UI-SPEC §6 table. Background: rgba(24,29,41,0.85) + backdrop-blur(8px).
 */
import type { ReactElement } from "react";

interface LegendEntry {
  status: string;
  label: string;
}

const LEGEND_ENTRIES: LegendEntry[] = [
  { status: "in",        label: "in corpus" },
  { status: "in-other",  label: "other coll." },
  { status: "missing",   label: "missing" },
  { status: "ingesting", label: "ingesting" },
  { status: "seed",      label: "seed" },
];

interface SwatchProps {
  status: string;
}

function Swatch({ status }: SwatchProps): ReactElement {
  if (status === "in") {
    return (
      <svg width="12" height="12" aria-hidden>
        <circle cx="6" cy="6" r="4.5" fill="#3B82F6" stroke="#93C5FD" strokeWidth="1.4" />
      </svg>
    );
  }
  if (status === "in-other") {
    return (
      <svg width="12" height="12" aria-hidden>
        <circle cx="6" cy="6" r="4.5" fill="#8B5CF6" stroke="#C4B5FD" strokeWidth="1.4" />
      </svg>
    );
  }
  if (status === "missing") {
    return (
      <svg width="12" height="12" aria-hidden>
        <circle
          cx="6"
          cy="6"
          r="4.5"
          fill="transparent"
          stroke="#6B7890"
          strokeWidth="1.3"
          strokeDasharray="2.5 2"
        />
      </svg>
    );
  }
  if (status === "ingesting") {
    // Partial amber arc (approx 55% circumference) representing in-progress state
    const r = 4.5;
    const cx = 6;
    const cy = 6;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + Math.PI * 2 * 0.55;
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    return (
      <svg width="12" height="12" aria-hidden>
        <circle cx="6" cy="6" r="4.5" fill="transparent" stroke="#333" strokeWidth="1" opacity="0.3" />
        <path
          d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`}
          fill="none"
          stroke="#F59E0B"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    );
  }
  if (status === "seed") {
    return (
      <svg width="12" height="12" aria-hidden>
        <circle cx="6" cy="6" r="4.5" fill="#22C55E" stroke="#86EFAC" strokeWidth="1.4" />
      </svg>
    );
  }
  return <svg width="12" height="12" aria-hidden />;
}

export function GraphLegend(): ReactElement {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 12,
        right: 12,
        padding: "10px 12px",
        borderRadius: 6,
        background: "rgba(24,29,41,0.85)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        border: "1px solid var(--p3-border)",
        display: "grid",
        gridTemplateColumns: "auto 1fr",
        gap: "6px 8px",
        alignItems: "center",
      }}
      role="list"
      aria-label="Node status legend"
    >
      {LEGEND_ENTRIES.map((entry) => (
        <div key={entry.status} style={{ display: "contents" }} role="listitem">
          <Swatch status={entry.status} />
          <span
            style={{
              fontSize: 10.5,
              color: "var(--p3-muted)",
              fontFamily: "var(--p3-font-mono)",
              whiteSpace: "nowrap",
            }}
          >
            {entry.label}
          </span>
        </div>
      ))}
    </div>
  );
}
