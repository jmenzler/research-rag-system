import { useUiStore } from "@/state/uiStore";

export function OfflineBanner() {
  const streak = useUiStore((s) => s.healthFailureStreak);

  if (streak < 3) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      data-testid="offline-banner"
      className="flex shrink-0 items-center justify-center gap-2 border-b border-destructive/40 bg-destructive/10 px-4 py-2 text-sm text-destructive"
    >
      Backend unreachable — check that the API server is running
    </div>
  );
}
