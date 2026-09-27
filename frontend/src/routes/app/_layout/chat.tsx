import {
  createFileRoute,
  Outlet,
  useChildMatches,
} from "@tanstack/react-router";
import { MessageSquarePlus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";

export const Route = createFileRoute("/app/_layout/chat")({
  component: ChatLayout,
});

function ChatLayout() {
  // A child match means /app/chat/$chatId is active — render the surface via
  // <Outlet/>. With no child we are on the bare /app/chat index, so show the
  // empty-state prompt instead of a blank page.
  const hasActiveChat = useChildMatches().length > 0;
  const setScopeDialogOpen = useUiStore((s) => s.setScopeDialogOpen);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 overflow-hidden">
      {hasActiveChat ? (
        <Outlet />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-4 px-6 text-center">
          <MessageSquarePlus className="size-10 text-[var(--p3-muted)]" />
          <div className="flex flex-col gap-1">
            <p className="text-base font-medium text-[var(--p3-fg)]">
              Start a new chat
            </p>
            <p className="max-w-sm text-sm text-[var(--p3-muted)]">
              Pick a retriever and the collections to ground in, then ask a
              question against your corpus.
            </p>
          </div>
          <Button onClick={() => setScopeDialogOpen("create")}>
            New chat
          </Button>
        </div>
      )}
    </div>
  );
}
