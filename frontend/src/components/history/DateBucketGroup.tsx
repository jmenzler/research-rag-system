/**
 * DateBucketGroup — one labeled bucket of chat rows in the sidebar (D-11).
 *
 * The label string is rendered verbatim from BUCKET_ORDER (Today /
 * Yesterday / Last 7 days / Last 30 days / Older). Callers must skip
 * empty buckets BEFORE invoking this component — `bucketByDate` already
 * filters those out for them.
 */
import type { ReactElement } from "react";

import { ChatListItem } from "@/components/history/ChatListItem";
import type { ChatRow } from "@/components/history/ChatListItem";

export interface DateBucketGroupProps {
  label: string;
  items: ChatRow[];
  activeChatId?: string | null | undefined;
  onRename?: ((chat: ChatRow) => void) | undefined;
  onDelete?: ((chat: ChatRow) => void) | undefined;
  onArchiveToggle?: ((chat: ChatRow) => void) | undefined;
}

export function DateBucketGroup(props: DateBucketGroupProps): ReactElement {
  return (
    <div className="flex flex-col gap-1">
      <p className="text-muted-foreground px-3 pt-2 text-xs font-medium">
        {props.label}
      </p>
      <ul className="flex flex-col">
        {props.items.map((chat) => {
          const itemProps = {
            chat,
            isActive: props.activeChatId === chat.id,
            ...(props.onRename !== undefined
              ? { onRename: props.onRename }
              : {}),
            ...(props.onDelete !== undefined
              ? { onDelete: props.onDelete }
              : {}),
            ...(props.onArchiveToggle !== undefined
              ? { onArchiveToggle: props.onArchiveToggle }
              : {}),
          };
          return (
            <li key={chat.id}>
              <ChatListItem {...itemProps} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}
