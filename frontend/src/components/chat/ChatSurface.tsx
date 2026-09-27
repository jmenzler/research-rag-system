/**
 * ChatSurface (CHAT-11, D-17): standalone chat surface component.
 *
 * Plan 06 originally placed this logic directly inside the route file
 * (frontend/src/routes/app/chat.$chatId.tsx). Plan 07 extracts the surface
 * into a Conversation-wrapped, Sheet-mounting component so it can be unit
 * tested without TanStack Router (frontend/src/__tests__/optimistic-lock.test.tsx
 * imports `<ChatSurface chatId="c1" />` directly).
 *
 * Owns:
 *   - useChatQuery(chatId) — canonical message list (CRIT-3 staleTime: 0)
 *   - useChatStream(chatId) — live SSE state machine
 *   - 409 handling: fires Sonner toast with the EXACT D-17 wording
 *     ("This chat was updated in another tab — reloading…") + invalidates
 *     ["chats", chatId] so the next read pulls the canonical version.
 *     The ref-based dedup key ensures React StrictMode double-effect /
 *     repeated 409s don't fire the toast more than once per error event.
 *     NO auto-retry — the user must re-submit (CRIT-8 mitigation).
 *   - Citation registry hydration: every time the live stream emits a
 *     citations event we call `registerCitations(pendingId, stream.citations)`
 *     so CitationSheet can resolve parent_id without falling back to the
 *     synthetic key.
 *   - Body-portal Sheets: CitationSheet + AuditDetailSheet mounted ONCE at
 *     the route root (radix portals attach to document.body — open/close
 *     is driven by uiStore selectors inside each Sheet).
 */
import {
  useEffect,
  useMemo,
  useRef,
  type ReactElement,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { useChatQuery } from "@/queries/chats";
import { useChatStream } from "@/hooks/useChatStream";
import { ComposeBox } from "@/components/chat/ComposeBox";
import { StageProgressStrip } from "@/components/chat/StageProgressStrip";
import { Conversation } from "@/components/chat/Conversation";
import { CitationSheet } from "@/components/chat/CitationSheet";
import { AuditDetailSheet } from "@/components/chat/AuditDetailSheet";
import { ChatScopeBar } from "@/components/chat/ChatScopeBar";
import { NewChatDialog } from "@/components/chat/NewChatDialog";
import { Separator } from "@/components/ui/separator";
import { useUiStore } from "@/state/uiStore";
import type { ChatMessage, RetrieverId } from "@/lib/api";
import { isLock409, OPTIMISTIC_409_TOAST } from "@/lib/api";
import type { CitationPayload } from "@/lib/sse-client";
import type { CollectionId } from "@/lib/retrievers";

// Plan 02-12 moved isLock409 + OPTIMISTIC_409_TOAST into frontend/src/lib/api.ts
// so the new sidebar mutations (rename/archive/delete) can surface the same
// optimistic-lock toast without forcing a circular import. Behaviour for the
// chat surface is unchanged — same regex, same toast wording (D-17).

export interface ChatSurfaceProps {
  chatId: string;
}

export function ChatSurface(props: ChatSurfaceProps): ReactElement {
  const { chatId } = props;
  const chatQuery = useChatQuery(chatId);
  const stream = useChatStream(chatId);
  const queryClient = useQueryClient();
  const registerCitations = useUiStore((s) => s.registerCitations);
  // Plan 02-13 — scope dialog open mode lives in the shared uiStore so the
  // sidebar's "New chat" CTA + the in-chat ChatScopeBar both flip it.
  const scopeDialogOpen = useUiStore((s) => s.scopeDialogOpen);
  const setScopeDialogOpen = useUiStore((s) => s.setScopeDialogOpen);

  const lastToastedKey = useRef<string | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);

  const expectedVersion = chatQuery.data?.version ?? 0;
  const streaming = stream.state === "queued" || stream.state === "streaming";

  /**
   * 409 detection: fires Sonner toast + invalidate ONCE per error event.
   *
   * Two paths trigger optimistic-lock 409:
   *   (a) the initial chatQuery fetch hits a 409 — error.message contains
   *       "HTTP 409" (set by src/lib/api.ts postJson/getJson wrappers)
   *   (b) the SSE POST hits a 409 — fetchEventSource throws and the
   *       stream.error string carries the upstream message
   *
   * The fixed-literal toast wording (NOT interpolating stream.error / API
   * error message) is the T-01-W4-04 mitigation — eliminates LLM/server-text
   * injection into the toast surface for the 409 path.
   */
  useEffect(() => {
    const candidates: Array<{ source: string; err: unknown }> = [];
    if (chatQuery.error) {
      candidates.push({ source: "chatQuery", err: chatQuery.error });
    }
    if (stream.state === "error" && stream.error !== null) {
      // The SSE branch surfaces errors as plain strings via stream.error
      // (we don't carry the structured ApiError through onError), so the
      // regex test in isLock409's fallback path is what catches these.
      candidates.push({ source: "stream", err: stream.error });
    }
    const has409 = candidates.some(({ err }) => isLock409(err));
    if (!has409) {
      // Both error sources are clear of 409s — reset the dedup so a future
      // 409 (different retry sequence) can fire the toast again.
      lastToastedKey.current = null;
      return;
    }
    // Coarse dedup key (WR-08): a single "OPTIMISTIC_409" key collapses
    // simultaneous 409s from BOTH chatQuery and stream into one toast.
    // The user is being told the same thing twice otherwise — same
    // upstream cause (chat version moved on us). The key resets above
    // when neither error source is a 409 anymore.
    if (lastToastedKey.current === "OPTIMISTIC_409") return;
    lastToastedKey.current = "OPTIMISTIC_409";
    toast.error(OPTIMISTIC_409_TOAST);
    void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
  }, [
    chatQuery.error,
    stream.state,
    stream.error,
    queryClient,
    chatId,
  ]);

  // Pending assistant turn (Phase 1 ships a single delta after synthesize).
  const pendingAssistant: ChatMessage | null = useMemo(() => {
    if (!streaming && stream.state !== "done") return null;
    const uuid = stream.messageUuid ?? "0";
    return {
      id: `pending-${uuid}`,
      role: "assistant",
      content: stream.delta,
      query_id: stream.done?.query_id ?? null,
      message_uuid: `${uuid}.assistant`,
      created_at: new Date().toISOString(),
      version: expectedVersion + 2,
    };
  }, [
    streaming,
    stream.state,
    stream.delta,
    stream.done,
    stream.messageUuid,
    expectedVersion,
  ]);

  // Hydrate the citation registry so CitationSheet can resolve parent_ids
  // from the (messageId, marker) pair without the synthetic fallback.
  useEffect(() => {
    if (pendingAssistant && stream.citations.length > 0) {
      registerCitations(pendingAssistant.id, stream.citations);
    }
  }, [pendingAssistant, stream.citations, registerCitations]);

  // Also hydrate from the rehydrated chat payload: assistants whose citations
  // were persisted server-side (chats_store.get_chat join) need the registry
  // populated on reload so CitationSheet resolves parent_id without 404-ing
  // through the synthetic fallback path.
  const persistedMessages = chatQuery.data?.messages;
  useEffect(() => {
    if (!persistedMessages) return;
    for (const m of persistedMessages) {
      if (m.citations && m.citations.length > 0) {
        registerCitations(m.id, m.citations as unknown as CitationPayload[]);
      }
    }
  }, [persistedMessages, registerCitations]);

  const messages: ChatMessage[] = chatQuery.data?.messages ?? [];
  // Keep the pending shim mounted until the canonical assistant row actually
  // lands in `messages`. `onDone` fires invalidateQueries (staleTime: 0), but
  // the refetch is a full async round-trip — dropping the shim the instant
  // state flips to "done" leaves a window where `messages` has neither the
  // shim nor the server row, so the just-streamed answer blanks then reappears.
  // The canonical row carries the deterministic `${messageUuid}.assistant`
  // uuid (api_chats.py), which is exactly the shim's message_uuid — so once it
  // arrives we drop the shim to avoid double-rendering the same turn.
  const canonicalArrived =
    pendingAssistant !== null &&
    messages.some((m) => m.message_uuid === pendingAssistant.message_uuid);
  const merged =
    pendingAssistant && !canonicalArrived
      ? [...messages, pendingAssistant]
      : messages;

  // Plan 02-14 — compute the index of the LAST user turn in the merged
  // list. The hover-pencil eligibility is exactly that one row, AND
  // only when no stream is active (UI-SPEC § Hover-pencil edit /
  // Eligibility). When a stream IS in flight the pencil hides itself
  // via the `streaming` prop, but we still gate the mount so the
  // computation stays cheap and deterministic.
  let lastUserIdx = -1;
  for (let i = merged.length - 1; i >= 0; i--) {
    if (merged[i]?.role === "user") {
      lastUserIdx = i;
      break;
    }
  }

  // Plan 02-14 — regenerate-last handler: POSTs to the JSON endpoint
  // (Plan 02-11 deviation — synchronous response, not SSE). On success
  // we invalidate the chat detail so the new assistant turn lands via
  // the canonical refetch path.
  const handleRegenerate = (): void => {
    if (streaming) return;
    void (async () => {
      try {
        const res = await fetch(`/api/chats/${chatId}/regenerate-last`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ expected_version: expectedVersion }),
        });
        if (!res.ok) {
          if (res.status === 409) {
            toast.error(OPTIMISTIC_409_TOAST);
          } else {
            toast.error(`Regenerate failed (HTTP ${res.status})`);
          }
          void queryClient.invalidateQueries({
            queryKey: ["chats", chatId],
          });
          return;
        }
        void queryClient.invalidateQueries({
          queryKey: ["chats", chatId],
        });
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Regenerate failed",
        );
      }
    })();
  };

  // Auto-scroll the thread to the bottom on mount, whenever the message list
  // grows, and as streamed deltas arrive — so the newest turn is always in
  // view without the user reaching for the scrollbar.
  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [merged.length, stream.delta]);

  // Contextual hotkeys: [ / ] cycle through citation pills in DOM order.
  // Dispatched by Plan 02's useGlobalHotkeys registry (hotkey:citation-next/prev).
  // No-ops when no citation pills are rendered (guard so focus() never throws).
  useEffect(() => {
    const getPills = (): HTMLElement[] =>
      Array.from(
        document.querySelectorAll<HTMLElement>(
          '[data-testid^="citation-token-"]',
        ),
      );

    const handleNext = (): void => {
      const pills = getPills();
      if (pills.length === 0) return;
      const focused = document.activeElement as HTMLElement | null;
      const idx = focused ? pills.indexOf(focused) : -1;
      const next = pills[(idx + 1) % pills.length];
      next?.focus();
    };

    const handlePrev = (): void => {
      const pills = getPills();
      if (pills.length === 0) return;
      const focused = document.activeElement as HTMLElement | null;
      const idx = focused ? pills.indexOf(focused) : -1;
      const prev = pills[(idx - 1 + pills.length) % pills.length];
      prev?.focus();
    };

    window.addEventListener("hotkey:citation-next", handleNext);
    window.addEventListener("hotkey:citation-prev", handlePrev);
    return () => {
      window.removeEventListener("hotkey:citation-next", handleNext);
      window.removeEventListener("hotkey:citation-prev", handlePrev);
    };
  }, []);

  // Persisted citations from GET /api/chats/<id> (chats_store.get_chat joins
  // the citations table). Live SSE citations from the in-flight stream overlay
  // on the pending assistant — so reload and live render through the same
  // CitationToken pipeline downstream.
  const citationsByMessageId: Record<string, CitationPayload[]> = {};
  for (const m of messages) {
    if (m.citations && m.citations.length > 0) {
      citationsByMessageId[m.id] = m.citations as unknown as CitationPayload[];
    }
  }
  if (pendingAssistant) {
    citationsByMessageId[pendingAssistant.id] = stream.citations;
  }

  return (
    <div
      data-testid="chat-surface"
      className="flex h-full w-full flex-1 flex-col bg-[var(--p3-bg)]"
    >
      {/* Always-mounted aria-live region — must be in DOM before content changes (Pitfall 4).
          Render stream.delta regardless of streaming state: the backend ships the full answer
          as one delta right before `done`, so emptying on the done transition (when `streaming`
          flips false) races the announcement and screen readers never hear the completed
          response. delta survives `done` (the hook only clears it on the next start), so the
          final text persists in the region for AT to announce. */}
      <div className="sr-only" aria-live="polite" aria-atomic="false">
        {stream.delta}
      </div>

      <div
        ref={threadRef}
        className="mx-auto flex w-full max-w-[720px] flex-1 flex-col overflow-y-auto px-6 py-6"
      >
        {chatQuery.isLoading && (
          <div className="flex flex-col gap-2 py-4" aria-hidden="true">
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className="bg-muted/40 h-8 animate-pulse rounded motion-reduce:animate-none"
                style={{ width: i % 2 === 0 ? "80%" : "60%" }}
              />
            ))}
          </div>
        )}
        {chatQuery.error && (
          <p className="text-destructive text-sm">
            Failed to load chat: {String(chatQuery.error)}
          </p>
        )}
        {chatQuery.data && (
          <>
            <h2 className="mb-4 text-base font-medium text-[var(--p3-fg-2)]">
              {chatQuery.data.title ?? "Untitled chat"}
            </h2>
            {/* Plan 02-13: scope bar sits above the message list so the
                user can see at a glance which retriever + collections are
                active, and click to edit them via NewChatDialog. */}
            <ChatScopeBar
              chatId={chatId}
              retriever={chatQuery.data.retriever as RetrieverId}
              collections={parseCollections(chatQuery.data.collections)}
              streaming={streaming}
            />
            <div className="flex flex-col">
              {merged.map((m, i) => {
                const isLive = pendingAssistant?.id === m.id;
                // Plan 02-14 — hover-pencil eligibility: ONLY the
                // last user-turn row, AND only when no stream is
                // in flight. The component itself short-circuits
                // when streaming is true, but the gate at i ===
                // lastUserIdx keeps the DOM clean for older turns.
                const isLastUser = i === lastUserIdx && !streaming;
                return (
                  <div key={m.id}>
                    <Conversation
                      message={m}
                      citations={citationsByMessageId[m.id] ?? []}
                      liveDone={isLive ? stream.done : null}
                      isLive={isLive}
                      isLastUserTurn={isLastUser}
                      chatId={chatId}
                      expectedVersion={expectedVersion}
                      streaming={streaming}
                      streamingThisTurn={isLive}
                      onRegenerate={handleRegenerate}
                      retriever={chatQuery.data?.retriever ?? null}
                    />
                    {i < merged.length - 1 && <Separator />}
                  </div>
                );
              })}
            </div>
            {streaming && (
              <StageProgressStrip
                stagesObserved={stream.stagesObserved}
                state={stream.state}
                subStages={stream.subStages}
              />
            )}
            {stream.state === "done" && stream.done && (
              <StageProgressStrip
                stagesObserved={stream.stagesObserved}
                state="done"
                totalLatencyMs={stream.done.latency_ms}
              />
            )}
          </>
        )}
      </div>
      <div className="mx-auto w-full max-w-[720px]">
        <ComposeBox
          onSubmit={({ content }) =>
            stream.start({ content, expected_version: expectedVersion })
          }
          onStop={stream.stop}
          streaming={streaming}
          disabled={chatQuery.isLoading}
        />
      </div>
      {/* Body-portal Sheets — mounted once per surface. */}
      <CitationSheet />
      <AuditDetailSheet />
      {/* Plan 02-13 — scope edit dialog (Radix Dialog handles its own
          portal). Render only when chatQuery has resolved so we have a
          retriever + collections + version to prefill from. */}
      {chatQuery.data && scopeDialogOpen !== null && (
        <NewChatDialog
          open={scopeDialogOpen !== null}
          mode={scopeDialogOpen}
          onOpenChange={(o) => setScopeDialogOpen(o ? scopeDialogOpen : null)}
          chatId={chatId}
          initialRetriever={chatQuery.data.retriever as RetrieverId}
          initialCollections={parseCollections(chatQuery.data.collections)}
          expectedVersion={chatQuery.data.version}
        />
      )}
    </div>
  );
}

// Plan 02-13 — chat.collections arrives as a JSON-encoded string from the
// server (chats_store.get_chat); parse it lazily here so the scope bar and
// the edit dialog share one canonical typed list. Falls back to all-four
// when the field is malformed (defensive — D-03's "no filter = all four"
// ergonomics carry through to display).
function parseCollections(raw: string | null | undefined): CollectionId[] {
  if (!raw) return ["trading", "ecology", "notes", "system"];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.filter((c): c is CollectionId =>
        typeof c === "string" &&
        (c === "trading" || c === "ecology" || c === "notes" || c === "system"),
      );
    }
  } catch {
    /* fall through to default */
  }
  return ["trading", "ecology", "notes", "system"];
}
