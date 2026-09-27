/**
 * Wave-0 RED test for useChatStream() hook (D-04, D-20, Risk 5).
 *
 * Implementation lands in Plan 06 (frontend/src/hooks/useChatStream.ts).
 *
 * Behavioural contract (01-RESEARCH.md §Code Examples / Frontend SSE consumer):
 *   - State machine: idle -> queued -> streaming -> done | cancelled | error.
 *   - Accumulates citations on `event: citations`.
 *   - Captures delta text on `event: delta`.
 *   - stop() invokes AbortController.abort() and sets state to "cancelled".
 *   - Regenerates message_uuid on stop+retry (Risk 5).
 *
 * The hook is exercised via React Testing Library's renderHook + mocked
 * @microsoft/fetch-event-source so we can drive the SSE events imperatively.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

interface FetchEventSourceMockArgs {
  onmessage?: (ev: { event: string; data: string }) => void;
  signal?: AbortSignal;
  body?: string;
}

const capturedCalls: FetchEventSourceMockArgs[] = [];

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: vi.fn(
    async (
      _url: string,
      opts: FetchEventSourceMockArgs,
    ): Promise<void> => {
      capturedCalls.push(opts);
      // Keep the promise open until aborted or explicit drain; tests close it.
      return new Promise<void>((resolve) => {
        if (opts.signal) {
          opts.signal.addEventListener("abort", () => resolve());
        }
      });
    },
  ),
}));

beforeEach(() => {
  capturedCalls.length = 0;
  vi.clearAllMocks();
});

describe("useChatStream", () => {
  it("transitions idle -> queued -> streaming -> done", async () => {
    const { useChatStream } = await import("@/hooks/useChatStream");
    const { result } = renderHook(() => useChatStream("chat-1"));

    expect(result.current.state).toBe("idle");

    act(() => {
      result.current.start({
        content: "hello",
        message_uuid: "u-1",
        expected_version: 0,
      });
    });

    await waitFor(() => expect(capturedCalls.length).toBe(1));
    const handler = capturedCalls[0]!.onmessage!;

    act(() => handler({ event: "queued", data: JSON.stringify({ chat_id: "chat-1" }) }));
    await waitFor(() => expect(result.current.state).toBe("queued"));

    act(() =>
      handler({
        event: "stage",
        data: JSON.stringify({ stage: "decompose", latency_ms: 10 }),
      }),
    );
    await waitFor(() => expect(result.current.state).toBe("streaming"));

    act(() =>
      handler({
        event: "done",
        data: JSON.stringify({
          query_id: "q-1",
          cost_usd: 0.001,
          latency_ms: 100,
        }),
      }),
    );
    await waitFor(() => expect(result.current.state).toBe("done"));
  });

  it("accumulates citations on event: citations", async () => {
    const { useChatStream } = await import("@/hooks/useChatStream");
    const { result } = renderHook(() => useChatStream("chat-2"));

    act(() => {
      result.current.start({
        content: "hi",
        message_uuid: "u-2",
        expected_version: 0,
      });
    });
    await waitFor(() => expect(capturedCalls.length).toBe(1));
    const handler = capturedCalls[0]!.onmessage!;

    act(() =>
      handler({
        event: "citations",
        data: JSON.stringify([
          {
            marker: 1,
            child_id: "c1",
            parent_id: "p1",
            paper_id: "x",
            score_dense: 0.9,
            score_sparse: 0.8,
            score_rerank: 0.7,
            resolved: true,
          },
        ]),
      }),
    );
    await waitFor(() => expect(result.current.citations.length).toBe(1));
    expect(result.current.citations[0]!.marker).toBe(1);
  });

  it("captures delta text on event: delta", async () => {
    const { useChatStream } = await import("@/hooks/useChatStream");
    const { result } = renderHook(() => useChatStream("chat-3"));

    act(() => {
      result.current.start({
        content: "hi",
        message_uuid: "u-3",
        expected_version: 0,
      });
    });
    await waitFor(() => expect(capturedCalls.length).toBe(1));
    const handler = capturedCalls[0]!.onmessage!;

    act(() =>
      handler({ event: "delta", data: JSON.stringify({ text: "hello world" }) }),
    );
    await waitFor(() => expect(result.current.delta).toBe("hello world"));
  });

  it("stop() calls AbortController.abort and sets state to cancelled", async () => {
    const { useChatStream } = await import("@/hooks/useChatStream");
    const { result } = renderHook(() => useChatStream("chat-4"));

    act(() => {
      result.current.start({
        content: "hi",
        message_uuid: "u-4",
        expected_version: 0,
      });
    });
    await waitFor(() => expect(capturedCalls.length).toBe(1));
    const signal = capturedCalls[0]!.signal!;
    expect(signal.aborted).toBe(false);

    act(() => result.current.stop());
    await waitFor(() => expect(signal.aborted).toBe(true));
    expect(result.current.state).toBe("cancelled");
  });

  it("regenerates message_uuid on stop+retry (Risk 5)", async () => {
    const { useChatStream } = await import("@/hooks/useChatStream");
    const { result } = renderHook(() => useChatStream("chat-5"));

    act(() => {
      // First attempt without explicit uuid → hook generates one.
      result.current.start({ content: "x" });
    });
    await waitFor(() => expect(capturedCalls.length).toBe(1));
    const first = JSON.parse(capturedCalls[0]!.body ?? "{}");

    act(() => result.current.stop());
    await waitFor(() => expect(result.current.state).toBe("cancelled"));

    act(() => {
      result.current.start({ content: "x" });
    });
    await waitFor(() => expect(capturedCalls.length).toBe(2));
    const second = JSON.parse(capturedCalls[1]!.body ?? "{}");

    expect(first.message_uuid).toBeTruthy();
    expect(second.message_uuid).toBeTruthy();
    expect(second.message_uuid).not.toBe(first.message_uuid);
  });
});
