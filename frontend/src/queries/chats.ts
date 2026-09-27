/**
 * TanStack Query modules for chats + chunks (Phase 1 Plan 05 + Phase 2 Plan 02-12).
 *
 * staleTime discipline (CRIT-3 / RESEARCH §Pitfall 4):
 *   - `useChatsListQuery`, `useChatQuery`: staleTime 0 — server is source of truth.
 *   - `useChunkQuery`: staleTime Infinity — parent chunks are immutable by id.
 *
 * Plan 02-12 mutations: rename / archive / delete chats. All surface 409 via
 * the shared OPTIMISTIC_409_TOAST (Phase 1 D-17 wording, locked).
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  UseMutationResult,
  UseQueryResult,
} from "@tanstack/react-query";
import { toast } from "sonner";

import { api, isLock409, OPTIMISTIC_409_TOAST } from "@/lib/api";
import type {
  ChatDetail,
  ChatPatchBody,
  ChatSummary,
  ChunkPayload,
  CreateChatBody,
} from "@/lib/api";

export function useChatsListQuery(): UseQueryResult<{ chats: ChatSummary[] }> {
  return useQuery({
    queryKey: ["chats"],
    queryFn: api.chats.list,
    staleTime: 0, // server is source of truth (CRIT-3 / RESEARCH §Pitfall 4)
  });
}

export function useChatQuery(chatId: string): UseQueryResult<ChatDetail> {
  return useQuery({
    queryKey: ["chats", chatId],
    queryFn: () => api.chats.get(chatId),
    staleTime: 0, // CRIT-3
  });
}

export function useChunkQuery(
  parentId: string | null,
): UseQueryResult<ChunkPayload> {
  return useQuery({
    queryKey: ["chunks", parentId],
    queryFn: () => api.chunks.get(parentId as string),
    enabled: parentId !== null,
    // Parent chunks are immutable by id (RESEARCH §Pitfall 4 row 3).
    staleTime: Number.POSITIVE_INFINITY,
  });
}

export function useCreateChatMutation(): UseMutationResult<
  ChatDetail,
  Error,
  CreateChatBody
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateChatBody) => api.chats.create(body),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["chats"] });
    },
  });
}

/**
 * Plan 02-12 — rename / archive / retriever-or-collections edit.
 *
 * On 409: fires the shared OPTIMISTIC_409_TOAST (D-17 wording, locked) and
 * invalidates `["chats", chatId]` so the next read pulls the canonical row.
 * Other errors: generic toast describing the failure. The mutation does NOT
 * retry automatically — the user re-submits via the dialog (CRIT-8 mitigation).
 */
export function useUpdateChatMutation(
  chatId: string,
): UseMutationResult<ChatSummary, Error, ChatPatchBody> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: ChatPatchBody) => api.chats.patch(chatId, body),
    retry: false,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["chats"] });
      void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
    },
    onError: (err) => {
      if (isLock409(err)) {
        toast.error(OPTIMISTIC_409_TOAST);
        void queryClient.invalidateQueries({ queryKey: ["chats", chatId] });
        return;
      }
      toast.error(`Couldn't update chat: ${err.message}`);
    },
  });
}

/**
 * Plan 02-12 — delete chat. On success invalidates the chat list so the
 * sidebar refetches. The caller (DeleteChatConfirm) is responsible for
 * navigating away from the deleted chat's route if it was the active one.
 */
export function useDeleteChatMutation(): UseMutationResult<
  void,
  Error,
  string
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (chatId: string) => api.chats.delete(chatId),
    retry: false,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["chats"] });
    },
    onError: (err) => {
      if (isLock409(err)) {
        toast.error(OPTIMISTIC_409_TOAST);
        return;
      }
      toast.error(`Couldn't delete chat: ${err.message}`);
    },
  });
}
