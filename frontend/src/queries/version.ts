import { useQuery } from "@tanstack/react-query";
import { api, type VersionPayload } from "@/lib/api";
import { useUiStore } from "@/state/uiStore";
import { useEffect } from "react";

export function useVersionQuery() {
  const setFirstSeenSha = useUiStore((s) => s.setFirstSeenSha);
  const noteHealthFailure = useUiStore((s) => s.noteHealthFailure);
  const noteHealthSuccess = useUiStore((s) => s.noteHealthSuccess);

  const query = useQuery<VersionPayload>({
    queryKey: ["version"],
    queryFn: api.version,
    refetchInterval: 60_000, // D-15
    refetchIntervalInBackground: true,
    // retry is controlled by QueryClient defaults (1 in production, overridden to false in tests)
  });

  useEffect(() => {
    if (query.data?.sha) {
      setFirstSeenSha(query.data.sha);
      noteHealthSuccess();
    }
  }, [query.data?.sha, setFirstSeenSha, noteHealthSuccess]);

  useEffect(() => {
    if (query.isError) noteHealthFailure();
  }, [query.isError, noteHealthFailure]);

  return query;
}
