/**
 * Tests for useQueueStream hook (Plan 05-04 Task 2).
 *
 * EventSource is not available in jsdom — we define a fake class that
 * dispatches events manually. api.graph.inCorpusCheck and toast are spied on.
 */
import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// --- Mock EventSource --------------------------------------------------

interface FakeListener {
  (ev: MessageEvent): void;
}

class FakeEventSource {
  static instance: FakeEventSource | null = null;
  static constructCount = 0;

  readonly url: string;
  private listeners: Map<string, FakeListener[]> = new Map();
  closeCalled = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instance = this;
    FakeEventSource.constructCount += 1;
  }

  addEventListener(type: string, cb: FakeListener): void {
    const existing = this.listeners.get(type) ?? [];
    existing.push(cb);
    this.listeners.set(type, existing);
  }

  removeEventListener(type: string, cb: FakeListener): void {
    const existing = this.listeners.get(type) ?? [];
    this.listeners.set(
      type,
      existing.filter((fn) => fn !== cb),
    );
  }

  dispatch(type: string, data: unknown): void {
    const cbs = this.listeners.get(type) ?? [];
    const ev = { data: JSON.stringify(data) } as MessageEvent;
    cbs.forEach((cb) => cb(ev));
  }

  close(): void {
    this.closeCalled = true;
  }
}

vi.stubGlobal("EventSource", FakeEventSource);

// --- Mock api and toast -----------------------------------------------

vi.mock("@/lib/api", () => ({
  api: {
    graph: {
      inCorpusCheck: vi.fn().mockResolvedValue({}),
    },
  },
}));

vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), {
    error: vi.fn(),
  }),
}));

// eslint-disable-next-line import/order
import { api } from "@/lib/api";
// eslint-disable-next-line import/order
import { toast } from "sonner";
// eslint-disable-next-line import/order
import { useQueueStream } from "@/hooks/useQueueStream";
// eslint-disable-next-line import/order
import { useGraphStore } from "@/state/graphStore";

beforeEach(() => {
  FakeEventSource.instance = null;
  FakeEventSource.constructCount = 0;
  vi.clearAllMocks();
  // Reset graphStore in-flight set
  useGraphStore.setState({ inFlightCorpusIds: new Set() });
});

afterEach(() => {
  vi.clearAllMocks();
});

function getEs(): FakeEventSource {
  expect(FakeEventSource.instance).not.toBeNull();
  return FakeEventSource.instance!;
}

describe("useQueueStream — event dispatch", () => {
  it("opens EventSource at /api/queue/stream", () => {
    renderHook(() => useQueueStream());
    expect(getEs().url).toBe("/api/queue/stream");
  });

  it("mounts EventSource ONCE across re-renders with a changing onRetry (white-flash regression)", () => {
    let renderCount = 0;
    const { rerender } = renderHook(() => {
      renderCount += 1;
      // New onRetry identity every render — must NOT recreate the stream.
      return useQueueStream({ onRetry: () => {} });
    });
    rerender();
    rerender();
    rerender();
    expect(renderCount).toBeGreaterThan(1);
    expect(FakeEventSource.constructCount).toBe(1);
  });

  it("position event → jobs[run_id].state=queued with position + queue_depth", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("position", { run_id: "r1", position: 2, queue_depth: 3 });
    });
    const job = result.current.jobs["r1"]!;
    expect(job).toBeDefined();
    expect(job.state).toBe("queued");
    expect(job.position).toBe(2);
    expect(job.queue_depth).toBe(3);
  });

  it("running event → jobs[run_id].state=running with stage set", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("running", { run_id: "r1", elapsed_s: 5, stage: "embedding" });
    });
    const job = result.current.jobs["r1"]!;
    expect(job.state).toBe("running");
    expect(job.stage).toBe("embedding");
    expect(job.elapsed_s).toBe(5);
  });

  it("done event → calls api.graph.inCorpusCheck with the event corpus_ids", async () => {
    renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("done", { run_id: "r1", corpus_ids: [10, 20], n_imported: 2 });
    });
    expect(api.graph.inCorpusCheck).toHaveBeenCalledWith({ corpus_ids: [10, 20] });
  });

  it("done event → job marked done", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("done", { run_id: "r1", corpus_ids: [10], n_imported: 1 });
    });
    expect(result.current.jobs["r1"]!.state).toBe("done");
  });

  it("done event → removeInFlight clears corpus_ids from graphStore", () => {
    useGraphStore.setState({ inFlightCorpusIds: new Set([10, 20]) });
    renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("done", { run_id: "r1", corpus_ids: [10, 20], n_imported: 2 });
    });
    const { inFlightCorpusIds } = useGraphStore.getState();
    expect(inFlightCorpusIds.has(10)).toBe(false);
    expect(inFlightCorpusIds.has(20)).toBe(false);
  });

  it("crashed event → toast.error called with note text + Retry action", () => {
    renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("crashed", { run_id: "r1", note: "disk full", corpus_ids: [5] });
    });
    expect(toast.error).toHaveBeenCalledWith(
      expect.stringContaining("disk full"),
      expect.objectContaining({
        action: expect.objectContaining({ label: "Retry" }),
      }),
    );
  });

  it("crashed event → job marked crashed (not silently dropped)", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("crashed", { run_id: "r1", note: "disk full", corpus_ids: [5] });
    });
    expect(result.current.jobs["r1"]).toBeDefined();
    expect(result.current.jobs["r1"]!.state).toBe("crashed");
  });

  it("repeated crashed frames for one run → toast fires ONCE (stuck-toast regression)", () => {
    renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("crashed", { run_id: "r1", note: "boom", corpus_ids: [5] });
      getEs().dispatch("crashed", { run_id: "r1", note: "boom", corpus_ids: [5] });
      getEs().dispatch("crashed", { run_id: "r1", note: "boom", corpus_ids: [5] });
    });
    expect(toast.error).toHaveBeenCalledTimes(1);
  });

  it("repeated done frames for one run → inCorpusCheck fires ONCE (replay-flash regression)", () => {
    renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("done", { run_id: "r1", corpus_ids: [10], n_imported: 1 });
      getEs().dispatch("done", { run_id: "r1", corpus_ids: [10], n_imported: 1 });
    });
    expect(api.graph.inCorpusCheck).toHaveBeenCalledTimes(1);
  });

  it("cancelled event → job marked cancelled", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("cancelled", { run_id: "r1", corpus_ids: [5] });
    });
    expect(result.current.jobs["r1"]!.state).toBe("cancelled");
  });

  it("counts derive correctly from jobs map", () => {
    const { result } = renderHook(() => useQueueStream());
    act(() => {
      getEs().dispatch("position", { run_id: "r1", position: 1, queue_depth: 2 });
      getEs().dispatch("position", { run_id: "r2", position: 2, queue_depth: 2 });
      getEs().dispatch("running", { run_id: "r3", elapsed_s: 1, stage: "embedding" });
    });
    expect(result.current.counts.queued).toBe(2);
    expect(result.current.counts.running).toBe(1);
  });

  it("es.close() called on unmount", () => {
    const { unmount } = renderHook(() => useQueueStream());
    const es = getEs();
    unmount();
    expect(es.closeCalled).toBe(true);
  });
});
