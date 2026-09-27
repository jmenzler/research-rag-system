/**
 * Compose box for the chat surface.
 *
 *   D-03: Enter (no shift) submits via `onSubmit`; Shift+Enter inserts a newline.
 *   D-04: Send button morphs to a red `destructive` Stop variant while streaming —
 *         same `<Button>` instance + focus target; only the variant + icon + click
 *         handler swap.
 *
 * `onSubmit` receives `{ content: string }` (object — extensible) rather than a
 * bare string. The chat surface route lifts the content into the `useChatStream`
 * call shape `{ content, message_uuid?, expected_version }`.
 */
import {
  type KeyboardEvent,
  type ReactElement,
  useCallback,
  useRef,
  useState,
} from "react";
import { Square, SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/cn";

export interface ComposeBoxProps {
  onSubmit: (payload: { content: string }) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
}

export function ComposeBox(props: ComposeBoxProps): ReactElement {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const submit = useCallback((): void => {
    const trimmed = value.trim();
    if (trimmed.length === 0) return;
    props.onSubmit({ content: trimmed });
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [value, props]);

  const onKeyDown = (ev: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (
      ev.key === "Enter" &&
      !ev.shiftKey &&
      !ev.nativeEvent.isComposing
    ) {
      ev.preventDefault();
      if (props.streaming) {
        // While streaming the button is the cancel control; Enter is a no-op.
        return;
      }
      submit();
    }
    // Shift+Enter falls through — textarea inserts a newline natively.
  };

  const onInput = (ev: React.FormEvent<HTMLTextAreaElement>): void => {
    const t = ev.currentTarget;
    t.style.height = "auto";
    const maxPx = 240; // ~10 lines @ 24px line-height
    t.style.height = `${Math.min(t.scrollHeight, maxPx)}px`;
  };

  const onButtonClick = (): void => {
    if (props.streaming) {
      props.onStop();
      return;
    }
    submit();
  };

  return (
    <div className="px-6 pb-5 pt-4">
      <form
        data-testid="compose-box"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="flex items-end gap-2 rounded-[14px] border border-[var(--p3-border)] bg-[var(--p3-surface)] p-2.5 shadow-[0_4px_24px_-8px_rgba(0,0,0,0.4)] transition-colors duration-150 focus-within:border-[var(--p3-primary-edge)] motion-reduce:transition-none"
      >
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          onInput={onInput}
          placeholder="Ask something… (⏎ to send, ⇧⏎ for newline)"
          rows={1}
          disabled={props.disabled}
          className={cn(
            "max-h-[240px] min-h-[40px] flex-1 resize-none border-none bg-transparent shadow-none focus-visible:ring-0",
          )}
          data-testid="compose-textarea"
        />
        <Button
          type="button"
          variant={props.streaming ? "destructive" : "default"}
          size="icon"
          onClick={onButtonClick}
          disabled={
            props.disabled || (!props.streaming && value.trim().length === 0)
          }
          aria-label={props.streaming ? "stop generation" : "send message"}
          data-testid="send-button"
          data-variant={props.streaming ? "destructive" : "default"}
          className="size-8 shrink-0 rounded-lg"
        >
          {props.streaming ? (
            <Square className="size-3.5 fill-current" />
          ) : (
            <SendHorizontal className="size-4" />
          )}
        </Button>
      </form>
    </div>
  );
}
