/**
 * <RetrieverPicker> (Plan 02-13).
 *
 * Renders 5 click-anchored rows from RETRIEVER_ORDER with the canonical
 * perf-cost label (exact strings from UI-SPEC § Copywriting Contract).
 *
 *   - Fused row carries a trailing Info <Popover> with FUSED_POPOVER_BODY
 *     (UI-SPEC §"shadcn components Phase 2 MUST add" — popover for
 *     click-anchored content, NOT HoverCard which is hover-only).
 *   - Rows whose availability flag is false render as visually disabled
 *     (opacity-60 cursor-not-allowed) AND with a <HoverCard> on hover
 *     carrying STUB_UNAVAILABLE_HOVERCARD. They cannot be selected.
 *
 * Controlled component: parent NewChatDialog owns the selected id.
 * Backed by Radix RadioGroup so a11y / keyboard nav (arrow keys) work
 * for free; the disabled rows opt out via `disabled` on the RadioGroupItem.
 */
import { type ReactElement } from "react";
import { Info } from "lucide-react";

import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  FUSED_POPOVER_BODY,
  RETRIEVERS,
  RETRIEVER_ORDER,
  STUB_UNAVAILABLE_HOVERCARD,
} from "@/lib/retrievers";
import type { RetrieverAvailability, RetrieverId } from "@/lib/api";
import { cn } from "@/lib/cn";

export interface RetrieverPickerProps {
  value: RetrieverId;
  onChange: (next: RetrieverId) => void;
  availability: RetrieverAvailability;
  disabled?: boolean;
}

export function RetrieverPicker(props: RetrieverPickerProps): ReactElement {
  const { value, onChange, availability, disabled = false } = props;

  return (
    <RadioGroup
      value={value}
      onValueChange={(v) => onChange(v as RetrieverId)}
      disabled={disabled}
      data-slot="retriever-picker"
      className="flex flex-col gap-2"
    >
      {RETRIEVER_ORDER.map((id) => {
        const meta = RETRIEVERS[id];
        const available = availability[id];
        const isFused = id === "fused";
        const itemId = `retriever-${id}`;

        const row = (
          <div
            key={id}
            data-retriever={id}
            className={cn(
              "flex items-center gap-2 rounded-md border border-transparent p-2 text-sm",
              value === id && available && "border-primary",
              !available && "opacity-60 cursor-not-allowed",
            )}
          >
            <RadioGroupItem
              id={itemId}
              value={id}
              aria-label={meta.label}
              disabled={!available || disabled}
            />
            <label
              htmlFor={itemId}
              aria-disabled={!available}
              className={cn(
                "cursor-pointer",
                !available && "opacity-60 cursor-not-allowed",
              )}
            >
              {meta.perfCostLabel}
            </label>
            {isFused && (
              <Popover>
                <PopoverTrigger
                  aria-label="fused info"
                  className="text-muted-foreground hover:text-foreground ml-1 inline-flex"
                >
                  <Info className="size-4" />
                </PopoverTrigger>
                <PopoverContent className="text-sm" align="start">
                  {FUSED_POPOVER_BODY}
                </PopoverContent>
              </Popover>
            )}
          </div>
        );

        if (!available) {
          return (
            <HoverCard key={id} openDelay={150}>
              <HoverCardTrigger asChild>{row}</HoverCardTrigger>
              <HoverCardContent className="text-sm" align="start">
                {STUB_UNAVAILABLE_HOVERCARD}
              </HoverCardContent>
            </HoverCard>
          );
        }

        return row;
      })}
    </RadioGroup>
  );
}
