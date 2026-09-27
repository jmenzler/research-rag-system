/**
 * CreateChatDialog — the always-mounted "New chat" create flow.
 *
 * Driven by useUiStore.scopeDialogOpen === "create". Wraps NewChatDialog in
 * create mode and, on a successful POST /api/chats, navigates to the new
 * chat at /app/chat/$chatId. Mounted at the layout level so the affordance
 * works from the empty chat index and from any chat surface.
 *
 * NewChatDialog's own create mutation closes the dialog on success; here we
 * additionally drive the post-create navigation off useCreateChatMutation so
 * we get the returned ChatDetail (with its id) without double-handling the
 * dialog's internal close.
 */
import type { ReactElement } from "react";
import { useNavigate } from "@tanstack/react-router";

import { NewChatDialog } from "@/components/chat/NewChatDialog";
import { useUiStore } from "@/state/uiStore";

export function CreateChatDialog(): ReactElement | null {
  const scopeDialogOpen = useUiStore((s) => s.scopeDialogOpen);
  const setScopeDialogOpen = useUiStore((s) => s.setScopeDialogOpen);
  const navigate = useNavigate();

  if (scopeDialogOpen !== "create") return null;

  return (
    <NewChatDialog
      open
      mode="create"
      onOpenChange={(o) => setScopeDialogOpen(o ? "create" : null)}
      onCreated={(chatId) => {
        void navigate({ to: "/app/chat/$chatId", params: { chatId } });
      }}
    />
  );
}
