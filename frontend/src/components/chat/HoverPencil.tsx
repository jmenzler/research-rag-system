/**
 * <HoverPencil> — fade-in pencil affordance on the LAST user turn that
 * opens `<MessageEditor>` (D-17).
 *
 * Behavioural contract (MessageEditor.test.tsx):
 *   1. Returns null entirely when `isLastUserTurn === false`. Older user
 *      turns do NOT render a pencil.
 *   2. While the chat is streaming (`streaming === true`) the button is
 *      either unmounted OR rendered with `pointer-events-none opacity-0`
 *      so a hovered cursor cannot trigger it. We pick "rendered but
 *      neutralised" so the layout stays stable when the stream lands —
 *      no shift when the pencil reappears.
 *   3. Click sets `useUiStore.editingMessageId = messageId`. The user-
 *      turn `<Message>` swaps content for `<MessageEditor>` on the
 *      next render (D-18).
 *
 * Visual: 32×32 ghost button (UI-SPEC § Spacing — `size-8`) holding a
 * 16-px `<Pencil>` glyph. Opacity transitions 0 → 100 on group hover
 * via the parent row's `group` class. The opacity stays 100 when this
 * messageId IS the editing target, so the affordance does not blink
 * out while the editor is open.
 *
 * The component does NOT manage focus / keyboard wiring — Phase 6
 * POLISH-05 owns keyboard map. Mouse-hover + click is the Phase 2
 * surface.
 */
import { type ReactElement } from "react";
import { Pencil } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";
import { cn } from "@/lib/cn";

export interface HoverPencilProps {
  messageId: string;
  isLastUserTurn: boolean;
  streaming: boolean;
}

export function HoverPencil(props: HoverPencilProps): ReactElement | null {
  const { messageId, isLastUserTurn, streaming } = props;
  const editingMessageId = useUiStore((s) => s.editingMessageId);
  const setEditingMessageId = useUiStore((s) => s.setEditingMessageId);

  // Older user turns never get a pencil (test "older user turns get no pencil").
  if (!isLastUserTurn) return null;

  const isEditingThis = editingMessageId === messageId;

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label="Edit message"
      onClick={() => setEditingMessageId(messageId)}
      className={cn(
        "size-8 opacity-0 transition-opacity group-hover:opacity-100",
        isEditingThis && "opacity-100",
        streaming && "pointer-events-none opacity-0",
      )}
    >
      <Pencil className="size-4" />
    </Button>
  );
}
