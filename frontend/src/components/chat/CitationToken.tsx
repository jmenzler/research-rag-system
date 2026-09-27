/**
 * Inline citation pill (D-06 / D-07 / D-09).
 *
 *  - resolved=true (default): superscript marker number, click dispatches
 *    `useUiStore.openCitation(messageId, marker)`. Hover-card opens after a
 *    150 ms delay with the parent-chunk preview (CitationHoverCard).
 *  - resolved=false (hallucinated marker, CHAT-05 / CRIT-7): superscript
 *    struck-through `[?]` glyph. Click still dispatches openCitation — Plan 07's
 *    CitationSheet renders an explanatory empty state. Hover-card explains the
 *    source could not be resolved.
 */
import { type ReactElement } from "react";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { CitationHoverCard } from "@/components/chat/CitationHoverCard";
import { useUiStore } from "@/state/uiStore";
import { cn } from "@/lib/cn";

export interface CitationTokenProps {
  marker: number;
  chunkId: string | null;
  resolved?: boolean;
  messageId: string;
}

export function CitationToken(props: CitationTokenProps): ReactElement {
  const resolved = props.resolved ?? true;
  // Read the action via getState() so tests that inject a spy via
  // useUiStore.setState({ openCitation: spy }) wire up correctly.
  const handleClick = (): void => {
    useUiStore.getState().openCitation(props.messageId, props.marker);
  };

  // p3 citation pill — bordered inline chip. Resolved: blue tint. Unresolved:
  // neutral border with a struck "?" (the `line-through` class is asserted by
  // CitationToken.test.tsx and signals the hallucinated-marker state, D-09).
  const pillBase =
    "mx-px inline-flex h-4 min-w-4 items-center justify-center rounded-sm border px-1 align-[2px] font-mono text-[10px] font-medium transition-colors duration-150 motion-reduce:transition-none";

  if (!resolved) {
    return (
      <HoverCard openDelay={150} closeDelay={150}>
        <HoverCardTrigger asChild>
          <button
            type="button"
            data-testid={`citation-token-${props.marker}-unresolved`}
            data-resolved="false"
            data-marker={props.marker}
            aria-label={`citation ${props.marker}, unresolved source`}
            onClick={handleClick}
            className={cn(
              pillBase,
              "line-through cursor-help border-[var(--p3-border)] bg-transparent text-[var(--p3-muted)] hover:bg-[var(--p3-surface)] hover:text-[var(--p3-fg-2)]",
            )}
          >
            [?]
          </button>
        </HoverCardTrigger>
        <HoverCardContent side="top">
          <CitationHoverCard chunkId={null} />
        </HoverCardContent>
      </HoverCard>
    );
  }

  return (
    <HoverCard openDelay={150} closeDelay={150}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          data-testid={`citation-token-${props.marker}`}
          data-resolved="true"
          data-marker={props.marker}
          data-chunkid={props.chunkId ?? undefined}
          aria-label={`citation ${props.marker}, open source`}
          onClick={handleClick}
          className={cn(
            pillBase,
            "cursor-pointer border-[var(--p3-primary-edge)] bg-[var(--p3-primary-soft)] text-[var(--p3-primary-text)] hover:border-[var(--p3-primary)] hover:bg-[rgba(59,130,246,0.22)]",
          )}
        >
          {props.marker}
        </button>
      </HoverCardTrigger>
      <HoverCardContent side="top">
        <CitationHoverCard chunkId={props.chunkId} />
      </HoverCardContent>
    </HoverCard>
  );
}
