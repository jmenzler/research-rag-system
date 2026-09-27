import { Network } from "lucide-react";
import type { ReactElement } from "react";

export interface GraphMetadataHeaderProps {
  collection: string;
  inCorpus: number;
  total: number;
  depth: number;
  direction: string;
  selectedLabel: string | null;
}

const DOT = (
  <span
    style={{ color: "var(--p3-muted-2)", opacity: 0.5 }}
    aria-hidden="true"
  >
    {" · "}
  </span>
);

export function GraphMetadataHeader({
  collection,
  inCorpus,
  total,
  depth,
  direction,
  selectedLabel,
}: GraphMetadataHeaderProps): ReactElement {
  return (
    <div
      className="flex h-9 shrink-0 items-center justify-between gap-2 border-b px-4 text-[11px]"
      style={{
        background: "var(--p3-bg)",
        borderColor: "var(--p3-border-soft)",
      }}
    >
      {/* Left: breadcrumb */}
      <div className="flex min-w-0 items-center gap-1.5 overflow-hidden">
        <Network
          size={11}
          style={{ color: "var(--p3-muted)", flexShrink: 0 }}
          aria-hidden="true"
        />
        <span style={{ color: "var(--p3-fg)" }}>Graph</span>
        {DOT}
        <span
          className="font-mono shrink-0"
          style={{ color: "var(--p3-muted)" }}
        >
          {collection}
        </span>
        {selectedLabel !== null && selectedLabel !== "" && (
          <>
            {DOT}
            <span
              className="font-mono truncate"
              style={{ color: "var(--p3-fg)" }}
            >
              {selectedLabel}
            </span>
          </>
        )}
      </div>

      {/* Right: stats */}
      <div
        className="flex shrink-0 items-center font-mono"
        style={{ color: "var(--p3-muted)" }}
      >
        <span
          className="tabular-nums"
          style={{ color: "var(--p3-fg)" }}
        >
          {inCorpus}
        </span>
        <span>{" / "}</span>
        <span className="tabular-nums">{total}</span>
        <span>{" in corpus"}</span>
        {DOT}
        <span>{`depth ${depth}`}</span>
        {DOT}
        <span>{direction}</span>
      </div>
    </div>
  );
}
