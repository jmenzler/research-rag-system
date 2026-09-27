/**
 * <MessageEditor> — inline edit-last `<Textarea>` swap for the last
 * user turn (D-17, D-18, Pitfall 4).
 *
 * Behavioural contract (MessageEditor.test.tsx):
 *   - Renders a `<Textarea>` pre-populated with `initialContent`.
 *   - "Save & regenerate" button POSTs to /api/chats/{chatId}/edit-last
 *     with `{content, message_uuid, expected_version}` where
 *     `message_uuid` MUST be a FRESH `crypto.randomUUID()` — NOT the
 *     original (Pitfall 4: reusing the prior user uuid triggers the
 *     server's idempotent-replay path which returns the stale answer).
 *     The backend (src/server/api_chats.py EditLastRequest) also
 *     auto-mints a fresh uuid when the client omits it; we send our
 *     own so the audit trail tracks the same uuid client→server.
 *   - "Cancel" button clears useUiStore.editingMessageId silently — no
 *     alertdialog, no confirm, even if the textarea is dirty (D-18:
 *     truncation + cancel are both silent flows).
 *
 * After a successful Save, the assistant turn is regenerated server-
 * side. The component itself does NOT attach to the new SSE stream —
 * the parent (Plan 02-14's `<Message>` wiring) clears
 * `editingMessageId` once the API call settles AND triggers a
 * `queryClient.invalidateQueries(["chats", chatId])` so the canonical
 * message list reloads with the edited user turn + the new assistant
 * turn.
 *
 * The JSON edit-last path (Plan 02-11 deviation) returns synchronously
 * — the new chat-detail fetch carries the regenerated assistant turn.
 * Plan 02-11 SUMMARY documents the dispatch contract.
 */
import { useEffect, useRef, useState, type ReactElement } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useUiStore } from "@/state/uiStore";
import { randomUuid } from "@/lib/uuid";

export interface MessageEditorProps {
  chatId: string;
  messageId: string;
  /** The user-turn's existing message_uuid — used by the cancel path
   *  for telemetry and asserted NOT-equal to the submitted uuid by the
   *  Pitfall-4 test. */
  originalMessageUuid: string;
  initialContent: string;
  expectedVersion: number;
}

export function MessageEditor(props: MessageEditorProps): ReactElement {
  const {
    chatId,
    originalMessageUuid,
    initialContent,
    expectedVersion,
  } = props;

  const [content, setContent] = useState<string>(initialContent);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const setEditingMessageId = useUiStore((s) => s.setEditingMessageId);
  // Query client is optional — tests mount via QueryClientProvider but a
  // future caller might not. Reading via useQueryClient inside the
  // submit handler keeps the dependency optional.
  const queryClient = useQueryClient();

  // Autofocus + cursor to end on mount (UI-SPEC State 1 for MessageEditor).
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.focus();
      const len = el.value.length;
      el.setSelectionRange(len, len);
    }
  }, []);

  const handleSave = async (): Promise<void> => {
    if (submitting) return;
    if (content.trim().length === 0) return; // Save is no-op on empty
    setSubmitting(true);
    // PITFALL 4 — fresh uuid per submit; reusing originalMessageUuid
    // would hit the server's idempotent-replay path and return the
    // stale answer. The test asserts `body.message_uuid !== originalMessageUuid`.
    const freshUuid = randomUuid();
    try {
      const res = await fetch(`/api/chats/${chatId}/edit-last`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          content,
          message_uuid: freshUuid,
          expected_version: expectedVersion,
        }),
      });
      if (!res.ok) {
        // 409 surfaces through the same OPTIMISTIC_409_TOAST elsewhere;
        // local toast covers the rest. Keep editor open so user can retry.
        toast.error(
          res.status === 409
            ? "This chat was updated in another tab — reloading…"
            : `Edit failed (HTTP ${res.status})`,
        );
        if (res.status === 409) {
          void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
        }
        return;
      }
      // Success — clear edit state and refresh canonical chat detail.
      setEditingMessageId(null);
      void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Edit failed — network error",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = (): void => {
    // D-18 silence: no confirm dialog even if dirty.
    setEditingMessageId(null);
  };

  // Original uuid surfaces purely as a hidden marker so debugger / dev
  // tools can correlate the editor instance with the source row; not
  // user-visible. Without this reference the lint warns about an unused
  // prop.
  return (
    <div
      data-testid="message-editor"
      data-original-uuid={originalMessageUuid}
      className="flex flex-col gap-2 py-2"
    >
      <Textarea
        ref={textareaRef}
        value={content}
        onChange={(e) => setContent(e.target.value)}
        disabled={submitting}
        aria-label="Edit message"
        className="min-h-24"
      />
      <div className="flex items-center justify-end gap-2">
        <Button
          type="button"
          variant="ghost"
          onClick={handleCancel}
          disabled={submitting}
        >
          Cancel
        </Button>
        <Button
          type="button"
          onClick={() => void handleSave()}
          disabled={
            submitting ||
            content.trim().length === 0 ||
            content === initialContent
          }
        >
          Save & regenerate
        </Button>
      </div>
    </div>
  );
}
