/**
 * /app/chat/$chatId — route shell. The actual surface lives in
 * frontend/src/components/chat/ChatSurface.tsx so it can be unit-tested
 * outside the TanStack Router context (optimistic-lock.test.tsx imports
 * ChatSurface directly with a chatId prop).
 */
import { createFileRoute } from "@tanstack/react-router";

import { ChatSurface } from "@/components/chat/ChatSurface";

export const Route = createFileRoute("/app/_layout/chat/$chatId")({
  component: ChatRoute,
});

function ChatRoute() {
  const { chatId } = Route.useParams();
  return <ChatSurface chatId={chatId} />;
}
