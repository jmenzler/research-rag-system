/**
 * Thin AbortController accessor for the morphing Stop button (D-04).
 *
 * The AbortController instance lives in the Zustand UI store so unrelated
 * surfaces (e.g. global keyboard shortcut, Phase 6 polish) can stop a running
 * generation. Plan 06's `useChatStream` populates the slot on start() and
 * clears it on done/error.
 */
import { useUiStore } from "@/state/uiStore";

export interface UseStopGenerationReturn {
  setController: (c: AbortController | null) => void;
  stop: () => void;
  controller: AbortController | null;
}

export function useStopGeneration(): UseStopGenerationReturn {
  const controller = useUiStore((s) => s.streamAbortController);
  const setController = useUiStore((s) => s.setStreamAbortController);

  const stop = (): void => {
    controller?.abort();
    setController(null);
  };

  return { setController, stop, controller };
}
