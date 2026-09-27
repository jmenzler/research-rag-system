/**
 * Ordered list of chat turns separated by `<Separator>` (D-02 per-turn divider).
 */
import { type ReactElement } from "react";
import { Separator } from "@/components/ui/separator";
import type { ChatMessage } from "@/lib/api";
import type { CitationPayload } from "@/lib/sse-client";
import { Message } from "@/components/chat/Message";

export interface MessageListProps {
  messages: ChatMessage[];
  citationsByMessageId?: Record<string, CitationPayload[]>;
}

export function MessageList(props: MessageListProps): ReactElement {
  const cites = props.citationsByMessageId ?? {};
  return (
    <div data-testid="message-list" className="flex flex-col">
      {props.messages.map((m, i) => {
        const c = cites[m.id];
        return (
          <div key={m.id}>
            {c !== undefined ? (
              <Message message={m} citations={c} />
            ) : (
              <Message message={m} />
            )}
            {i < props.messages.length - 1 && <Separator />}
          </div>
        );
      })}
    </div>
  );
}
