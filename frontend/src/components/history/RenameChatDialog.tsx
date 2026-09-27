/**
 * RenameChatDialog (HIST-03).
 *
 * shadcn Dialog with a prefilled Input. Save (primary) / Cancel (ghost).
 * 409 surfaces via the mutation's onError (Phase 1 toast wording). The
 * dialog stays open on error so the user can retry.
 *
 * Mounted ONCE per HistorySidebar. The sidebar's local `renameTarget`
 * state controls open/close — when null, the dialog is closed; when a
 * chat row triggers Rename… , the row is set as the target and the
 * dialog opens with title prefilled.
 */
import { useEffect, useState } from "react";
import type { FormEvent, ReactElement } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useUpdateChatMutation } from "@/queries/chats";

import type { ChatRow } from "@/components/history/ChatListItem";

export interface RenameChatDialogProps {
  /** When non-null the dialog opens with the chat's title prefilled. */
  target: ChatRow | null;
  /** Called when the dialog requests a close (success, cancel, escape). */
  onClose: () => void;
}

export function RenameChatDialog(props: RenameChatDialogProps): ReactElement {
  const { target, onClose } = props;
  const [value, setValue] = useState(target?.title ?? "");

  useEffect(() => {
    setValue(target?.title ?? "");
  }, [target]);

  const mutation = useUpdateChatMutation(target?.id ?? "");

  const handleSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (target === null) return;
    const trimmed = value.trim();
    if (trimmed.length === 0 || trimmed === target.title) {
      onClose();
      return;
    }
    mutation.mutate(
      { title: trimmed, expected_version: target.version },
      { onSuccess: () => onClose() },
    );
  };

  return (
    <Dialog
      open={target !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Rename chat</DialogTitle>
            <DialogDescription>
              Pick a new title for this chat.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Input
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              disabled={mutation.isPending}
              aria-label="Chat title"
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={onClose}
              disabled={mutation.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              Save
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
