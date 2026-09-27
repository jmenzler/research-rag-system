/**
 * FooterQueuePill — global queue status pill for the sidebar footer.
 *
 * Consumes useQueueStream().counts and displays:
 *   "◐ {running} ingesting · {queued} queued"  when active
 *   "Queue idle"                                 when empty (faint)
 *
 * Placed in HistorySidebar.tsx footer next to <VersionPill />.
 * Styling follows VersionPill pill shape with var(--p3-*) tokens.
 * jsdom-safe: no WebGL or canvas dependency.
 */
import type { ReactElement } from "react";
import { useQueueStream } from "@/hooks/useQueueStream";

export function FooterQueuePill(): ReactElement {
  const { counts } = useQueueStream();
  const hasActivity = counts.running > 0 || counts.queued > 0;

  if (hasActivity) {
    return (
      <span
        data-testid="footer-queue-pill"
        className="rounded-md border px-2 py-1 font-mono text-xs"
        style={{
          borderColor: "var(--p3-amber, #F59E0B)",
          color: "var(--p3-amber, #F59E0B)",
          backgroundColor: "rgba(245,158,11,0.08)",
        }}
        aria-live="polite"
        aria-label={`${counts.running} ingesting, ${counts.queued} queued`}
      >
        &#9680; {counts.running} ingesting &middot; {counts.queued} queued
      </span>
    );
  }

  return (
    <span
      data-testid="footer-queue-pill"
      className="rounded-md border px-2 py-1 font-mono text-xs"
      style={{
        borderColor: "var(--p3-border)",
        color: "var(--p3-muted-2, #6B7890)",
      }}
      aria-label="Queue idle"
    >
      Queue idle
    </span>
  );
}
