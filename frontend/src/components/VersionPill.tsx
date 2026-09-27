import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useVersionQuery } from "@/queries/version";
import { useUiStore } from "@/state/uiStore";
import { cn } from "@/lib/cn";

export function VersionPill() {
  const { data, isPending, isError } = useVersionQuery();
  const firstSeenSha = useUiStore((s) => s.firstSeenSha);
  const toastFiredRef = useRef(false);

  // D-15: when the fetched SHA differs from the first seen, fire a "new version" toast.
  useEffect(() => {
    if (
      data?.sha &&
      firstSeenSha &&
      data.sha !== firstSeenSha &&
      !toastFiredRef.current
    ) {
      toastFiredRef.current = true;
      toast("New version available — reload to update.", {
        action: { label: "Reload", onClick: () => window.location.reload() },
        duration: Infinity,
      });
    }
  }, [data?.sha, firstSeenSha]);

  let display: string;
  let stateClass: string;
  if (isPending) {
    display = "?";
    stateClass = "text-muted-foreground";
  } else if (isError || !data?.sha) {
    display = "offline";
    stateClass = "text-destructive";
  } else {
    display = data.sha.slice(0, 7);
    stateClass = "text-muted-foreground";
  }

  return (
    <span
      data-testid="version-pill"
      className={cn(
        "rounded-md border border-border px-2 py-1 font-mono text-xs",
        stateClass,
      )}
      title={data?.sha ?? undefined}
    >
      {display}
    </span>
  );
}
