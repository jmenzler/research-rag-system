/**
 * <MessageActions> — per-assistant-turn ⋯ overflow menu (D-19).
 *
 * Renders a shadcn `<DropdownMenu>` trigger at the end of the assistant
 * turn footer. The five items render in EXACT order asserted by
 * MessageActions.test.tsx:
 *
 *   1. Regenerate           — disabled while `streaming === true`
 *   2. Copy as text         — from <CopyMenuItems>
 *   3. Copy as markdown     — from <CopyMenuItems>
 *   <DropdownMenuSeparator>
 *   4. Open audit detail    — disabled when queryId === null
 *
 * State props:
 *   - streaming         — true when the parent chat has an in-flight
 *                         stream (any turn). Disables Regenerate.
 *   - streamingThisTurn — true when THIS specific turn is the one
 *                         currently being streamed. Hides the trigger
 *                         entirely (UI-SPEC states "trigger button is
 *                         hidden on the streaming turn").
 *
 * Open audit detail: calls `useUiStore.openAuditDetail(queryId)` —
 * Plan 1's audit-detail sheet handles the rest. The menu item is
 * `disabled` when queryId is null (e.g. an in-flight assistant turn
 * whose audit dir hasn't been minted yet).
 *
 * Regenerate handler: optional `onRegenerate` prop. When omitted
 * (e.g. the test harness which only asserts label + order) the click
 * is a no-op. Production callers wire it up to call
 * `useChatStream.start()` against the `/regenerate-last` endpoint
 * via the parent surface — see ChatSurface.tsx (Plan 02-14 wires this).
 */
import { type ReactElement } from "react";
import { MoreHorizontal, RotateCcw, FileText } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";
import { CopyMenuItems } from "@/components/chat/CopyMenuItems";
import type { CopyCitation } from "@/lib/copy-text";

export interface MessageActionsProps {
  chatId: string;
  messageId: string;
  /** Audit query_id; null on in-flight turns (Open audit disables). */
  queryId: string | null;
  /** Stream active anywhere in this chat — disables Regenerate. */
  streaming?: boolean;
  /** This specific turn is currently streaming — hides the trigger. */
  streamingThisTurn?: boolean;
  /** Plan-02-14 wires the regenerate POST flow through this. */
  onRegenerate?: (() => void) | undefined;
  /** Content + citations forwarded to the copy menu items. Both default
   *  to empty (test harness path) — production callers supply the real
   *  assistant content + persisted citations. */
  content?: string;
  citations?: CopyCitation[];
}

export function MessageActions(props: MessageActionsProps): ReactElement | null {
  const {
    queryId,
    streaming = false,
    streamingThisTurn = false,
    onRegenerate,
    content = "",
    citations = [],
  } = props;

  const openAuditDetail = useUiStore((s) => s.openAuditDetail);

  // Hide the entire menu on the in-flight assistant turn (UI-SPEC State 2
  // for `<MessageActions>` per-turn menu).
  if (streamingThisTurn) return null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Message actions"
          data-testid="message-actions-trigger"
        >
          <MoreHorizontal className="size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          onClick={() => onRegenerate?.()}
          disabled={streaming}
        >
          <RotateCcw className="mr-2 size-4" />
          Regenerate
        </DropdownMenuItem>
        <CopyMenuItems content={content} citations={citations} />
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={() => {
            if (queryId) openAuditDetail(queryId);
          }}
          disabled={!queryId}
        >
          <FileText className="mr-2 size-4" />
          Open audit detail
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
