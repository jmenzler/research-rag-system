/**
 * <NewChatDialog> (Plan 02-13).
 *
 * Two modes via useUiStore.scopeDialogOpen (D-19):
 *   - "create" — empty form (retriever=milvus, all 4 collections checked).
 *                Submit → POST /api/chats; close on success.
 *   - "edit"   — fields prefilled from current chat (props.initialRetriever
 *                + props.initialCollections). Title field hidden — rename
 *                lives in the sidebar's RenameChatDialog. Submit →
 *                PATCH /api/chats/{chat_id}; close on success.
 *
 * Availability:
 *   - Caller MAY pass `availability` explicitly (test path).
 *   - Otherwise the dialog issues a useQuery against api.chats.availability
 *     to populate disabled retriever states. Reqs the Plan 02-13 backend
 *     endpoint to be reachable.
 *   - Default while loading / on error: milvus + fused True, others False.
 *     This is the SAFE default — the user can't accidentally pick a stub'd
 *     retriever and see degraded results.
 *
 * Test contract (frontend/src/__tests__/NewChatDialog.test.tsx asserts):
 *   - All five perf-cost labels render verbatim
 *   - Fused popover body is verbatim
 *   - Collections default checked (all 4)
 *   - Retriever default = milvus
 *   - Disabled retriever surfaces the exact HoverCard text
 *   - Submit POSTs to /api/chats with {retriever, collections}
 */
import { useEffect, useMemo, useState, type ReactElement } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { CollectionPicker } from "@/components/chat/CollectionPicker";
import { RetrieverPicker } from "@/components/chat/RetrieverPicker";
import { api, isLock409, OPTIMISTIC_409_TOAST } from "@/lib/api";
import type {
  ChatPatchBody,
  CreateChatBody,
  RetrieverAvailability,
  RetrieverId,
} from "@/lib/api";
import { COLLECTION_ORDER, type CollectionId } from "@/lib/retrievers";

type Mode = "create" | "edit";

export interface NewChatDialogProps {
  open: boolean;
  mode: Mode;
  onOpenChange: (open: boolean) => void;
  /** Optional explicit availability map — when present, the dialog skips
   * the network query and renders directly from this. Tests use this. */
  availability?: RetrieverAvailability;
  /** Edit-mode prefill values. */
  chatId?: string;
  initialRetriever?: RetrieverId;
  initialCollections?: CollectionId[];
  /** Edit-mode CAS version (passed straight through to PATCH). */
  expectedVersion?: number;
  /** Create-mode success callback — receives the new chat id so the caller
   * can navigate to it. Fires AFTER the dialog closes itself. */
  onCreated?: (chatId: string) => void;
}

// Safe defaults when availability hasn't loaded yet OR the endpoint fails.
// Optimistic-disable for the optional retrievers keeps the user away from
// a stub'd result silently — milvus + fused are always available.
const SAFE_AVAILABILITY: RetrieverAvailability = {
  milvus: true,
  paperqa: false,
  hipporag: false,
  lazygraph: false,
  fused: true,
};

export function NewChatDialog(props: NewChatDialogProps): ReactElement {
  const {
    open,
    mode,
    onOpenChange,
    availability: availabilityProp,
    chatId,
    initialRetriever,
    initialCollections,
    expectedVersion,
    onCreated,
  } = props;
  const queryClient = useQueryClient();

  // Network availability query; skipped when caller already supplied one.
  // staleTime: 60s — the install state of a retriever lib doesn't flip
  // mid-session. Refetches when the dialog actually opens (not on every
  // chat surface mount).
  const availabilityQuery = useQuery({
    queryKey: ["chats", "availability"],
    queryFn: api.chats.availability,
    enabled: availabilityProp === undefined && open,
    staleTime: 60_000,
  });

  const availability: RetrieverAvailability =
    availabilityProp ?? availabilityQuery.data ?? SAFE_AVAILABILITY;

  // Form state — reset on mode-or-open transitions so reopening doesn't
  // leak stale values from the previous edit.
  const defaultRetriever: RetrieverId =
    mode === "edit" && initialRetriever ? initialRetriever : "milvus";
  const defaultCollections: CollectionId[] =
    mode === "edit" && initialCollections
      ? [...initialCollections]
      : [...COLLECTION_ORDER];

  const [retriever, setRetriever] = useState<RetrieverId>(defaultRetriever);
  const [collections, setCollections] = useState<CollectionId[]>(
    defaultCollections,
  );

  // Reset on dialog open so the form reflects the current props every time
  // it is shown (edit dialogs need to pick up the latest chat retriever).
  useEffect(() => {
    if (open) {
      setRetriever(defaultRetriever);
      setCollections(defaultCollections);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, mode, initialRetriever]);

  const createMutation = useMutation({
    mutationFn: (body: CreateChatBody) => api.chats.create(body),
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ["chats"] });
      onOpenChange(false);
      onCreated?.(created.id);
    },
    onError: (err) => {
      toast.error(`Couldn't create chat: ${(err as Error).message}`);
    },
  });

  const patchMutation = useMutation({
    mutationFn: (body: ChatPatchBody) =>
      api.chats.patch(chatId ?? "", body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["chats"] });
      if (chatId) {
        void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
      }
      onOpenChange(false);
    },
    onError: (err) => {
      if (isLock409(err)) {
        toast.error(OPTIMISTIC_409_TOAST);
        if (chatId) {
          void queryClient.invalidateQueries({
            queryKey: ["chats", chatId],
          });
        }
        return;
      }
      toast.error(`Couldn't update chat: ${(err as Error).message}`);
    },
  });

  const submitting = createMutation.isPending || patchMutation.isPending;

  function handleSubmit(): void {
    if (collections.length === 0) {
      toast.error("Select at least one collection — a chat with none can't retrieve.");
      return;
    }
    if (mode === "create") {
      createMutation.mutate({ retriever, collections });
    } else {
      if (!chatId || expectedVersion === undefined) {
        toast.error("Cannot save: missing chat id or version");
        return;
      }
      patchMutation.mutate({
        retriever,
        collections,
        expected_version: expectedVersion,
      });
    }
  }

  const primaryLabel = mode === "create" ? "Create chat" : "Save";
  const dialogTitle = useMemo(
    () => (mode === "create" ? "New chat" : "Edit chat scope"),
    [mode],
  );
  const dialogDescription = useMemo(
    () =>
      mode === "create"
        ? "Pick the retriever and collections this chat should ground in."
        : "Update which retriever and collections this chat uses for new turns.",
    [mode],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent data-slot="new-chat-dialog" className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
          <DialogDescription>{dialogDescription}</DialogDescription>
        </DialogHeader>

        <section className="flex flex-col gap-2">
          <h3 className="text-muted-foreground text-xs font-medium">
            Collections
          </h3>
          <CollectionPicker
            value={collections}
            onChange={setCollections}
            disabled={submitting}
          />
          {collections.length === 0 && (
            <p className="text-xs text-[var(--destructive)]">
              Select at least one collection — a chat with none can't retrieve.
            </p>
          )}
        </section>

        <section className="flex flex-col gap-2">
          <h3 className="text-muted-foreground text-xs font-medium">
            Retrievers
          </h3>
          <RetrieverPicker
            value={retriever}
            onChange={setRetriever}
            availability={availability}
            disabled={submitting}
          />
        </section>

        <DialogFooter>
          <Button
            variant="ghost"
            onClick={() => onOpenChange(false)}
            disabled={submitting}
          >
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting || collections.length === 0}>
            {primaryLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
