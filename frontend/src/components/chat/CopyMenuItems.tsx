/**
 * <CopyMenuItems> — DropdownMenuItem pair that triggers copy-as-text +
 * copy-as-markdown for an assistant turn (D-24, D-25 / CHAT-09).
 *
 * Visual: two `<DropdownMenuItem>` rows with `<Copy>` icons, labelled
 * "Copy as text" and "Copy as markdown" (exact UI-SPEC strings).
 *
 * Click handler: invokes `copyMessageAsText` / `copyMessageAsMarkdown`
 * from `@/lib/copy-text`. Those helpers own the toast wiring and
 * clipboard fallback — the menu items are pure UI.
 *
 * Mounted inline inside `<MessageActions>` so the menu order asserted
 * by MessageActions.test.tsx (Regenerate → text → markdown → separator
 * → audit) stays a single render path.
 */
import { type ReactElement } from "react";
import { Copy } from "lucide-react";

import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import {
  copyMessageAsText,
  copyMessageAsMarkdown,
  type CopyCitation,
} from "@/lib/copy-text";

export interface CopyMenuItemsProps {
  content: string;
  citations: CopyCitation[];
}

export function CopyMenuItems(props: CopyMenuItemsProps): ReactElement {
  return (
    <>
      <DropdownMenuItem
        onClick={() =>
          void copyMessageAsText({
            content: props.content,
            citations: props.citations,
          })
        }
      >
        <Copy className="mr-2 size-4" />
        Copy as text
      </DropdownMenuItem>
      <DropdownMenuItem
        onClick={() =>
          void copyMessageAsMarkdown({
            content: props.content,
            citations: props.citations,
          })
        }
      >
        <Copy className="mr-2 size-4" />
        Copy as markdown
      </DropdownMenuItem>
    </>
  );
}
