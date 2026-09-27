/**
 * Tests for graphStore inFlightCorpusIds slice (Plan 05-04 Task 2).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { useGraphStore } from "@/state/graphStore";

beforeEach(() => {
  useGraphStore.setState({ inFlightCorpusIds: new Set() });
});

describe("graphStore inFlightCorpusIds", () => {
  it("starts empty", () => {
    const { inFlightCorpusIds } = useGraphStore.getState();
    expect(inFlightCorpusIds.size).toBe(0);
  });

  it("addInFlight populates inFlightCorpusIds", () => {
    const { addInFlight } = useGraphStore.getState();
    addInFlight([1, 2, 3]);
    const { inFlightCorpusIds } = useGraphStore.getState();
    expect(inFlightCorpusIds.has(1)).toBe(true);
    expect(inFlightCorpusIds.has(2)).toBe(true);
    expect(inFlightCorpusIds.has(3)).toBe(true);
  });

  it("removeInFlight clears specific ids", () => {
    const { addInFlight, removeInFlight } = useGraphStore.getState();
    addInFlight([1, 2, 3]);
    removeInFlight([2]);
    const { inFlightCorpusIds } = useGraphStore.getState();
    expect(inFlightCorpusIds.has(1)).toBe(true);
    expect(inFlightCorpusIds.has(2)).toBe(false);
    expect(inFlightCorpusIds.has(3)).toBe(true);
  });

  it("addInFlight is idempotent", () => {
    const { addInFlight } = useGraphStore.getState();
    addInFlight([1]);
    addInFlight([1]);
    const { inFlightCorpusIds } = useGraphStore.getState();
    expect(inFlightCorpusIds.size).toBe(1);
  });
});

describe("graphStore overlay slice (tab-switch persistence regression)", () => {
  beforeEach(() => {
    useGraphStore.setState({ overlayNodes: null, overlayEdges: null, overflowCount: 0 });
  });

  it("starts null and lives in the store (not component-local, survives unmount)", () => {
    expect(useGraphStore.getState().overlayNodes).toBeNull();
  });

  it("setOverlayNodes accepts a direct value and a functional updater", () => {
    const { setOverlayNodes } = useGraphStore.getState();
    setOverlayNodes([{ corpusId: 1 }] as never);
    expect(useGraphStore.getState().overlayNodes).toHaveLength(1);
    // functional updater (the expand-merge path) appends without replacing
    setOverlayNodes(((prev: unknown[] | null) => [...(prev ?? []), { corpusId: 2 }]) as never);
    expect(useGraphStore.getState().overlayNodes).toHaveLength(2);
  });

  it("setOverflowCount supports updater form", () => {
    const { setOverflowCount } = useGraphStore.getState();
    setOverflowCount(5);
    setOverflowCount((n) => n + 1);
    expect(useGraphStore.getState().overflowCount).toBe(6);
  });
});
