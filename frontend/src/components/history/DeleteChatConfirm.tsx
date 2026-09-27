/**
 * DeleteChatConfirm (HIST-05).
 *
 * shadcn AlertDialog with exact copy from UI-SPEC:
 *   Title: "Delete this chat?"
 *   Body: '"{title}" will be permanently removed along with its messages
 *         and citations. This cannot be undone.'
 *   Destructive: "Delete chat"
 *   Cancel: "Keep chat"
 *
 * Title is truncated to 60 chars with `…` if longer (UI-SPEC).
 *
 * Mounted ONCE per HistorySidebar. If the deleted chat is the active
 * route, the caller navigates back to /app/chat after success.
 */
import type { ReactElement } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useDeleteChatMutation } from "@/queries/chats";

import type { ChatRow } from "@/components/history/ChatListItem";

function truncateTitle(title: string): string {
  return title.length > 60 ? title.slice(0, 60) + "…" : title;
}

export interface DeleteChatConfirmProps {
  /** When non-null the dialog is open targeting that chat. */
  target: ChatRow | null;
  /** Called when the dialog wants to close (success, cancel, escape). */
  onClose: () => void;
  /** Called after a successful delete, after the mutation invalidates list. */
  onDeleted?: (chat: ChatRow) => void;
}

export function DeleteChatConfirm(
  props: DeleteChatConfirmProps,
): ReactElement {
  const { target, onClose, onDeleted } = props;
  const mutation = useDeleteChatMutation();

  const handleDelete = (): void => {
    if (target === null) return;
    mutation.mutate(target.id, {
      onSuccess: () => {
        onDeleted?.(target);
        onClose();
      },
    });
  };

  const title = target?.title ?? "Untitled chat";
  const truncated = truncateTitle(title);

  return (
    <AlertDialog
      open={target !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
          <AlertDialogDescription>
            &quot;{truncated}&quot; will be permanently removed along with its
            messages and citations. This cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={mutation.isPending}>
            Keep chat
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleDelete}
            disabled={mutation.isPending}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            Delete chat
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
