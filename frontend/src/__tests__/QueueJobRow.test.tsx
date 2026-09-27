/**
 * Tests for QueueJobRow component (Plan 05-05 Task 1).
 *
 * Covers:
 * - Renders stage strip when running with known stage
 * - Highlights current stage (aria-current="step")
 * - Renders Retry button for crashed state
 * - Renders Retry button for cancelled state
 * - No Retry button for running or done states
 * - Renders failure note for crashed jobs
 *
 * jsdom-safe: pure DOM, no WebGL.
 */
import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { QueueJobRow } from "@/components/graph/QueueJobRow";
import type { QueueJob } from "@/hooks/useQueueStream";

function makeJob(overrides: Partial<QueueJob> = {}): QueueJob {
  return {
    run_id: "run-abc123",
    state: "queued",
    ...overrides,
  };
}

describe("QueueJobRow", () => {
  it("renders a row for a queued job", () => {
    render(<QueueJobRow job={makeJob()} onRetry={() => {}} />);
    expect(screen.getByTestId("queue-job-row")).toBeInTheDocument();
    expect(screen.getByText("Queued")).toBeInTheDocument();
  });

  it("renders the 4-stage strip when running with known stage", () => {
    const job = makeJob({ state: "running", stage: "parsing", corpus_ids: [1, 2] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.getByText("Fetch")).toBeInTheDocument();
    expect(screen.getByText("Parse")).toBeInTheDocument();
    expect(screen.getByText("Chunk")).toBeInTheDocument();
    expect(screen.getByText("Embed")).toBeInTheDocument();
  });

  it("marks the current stage with aria-current='step'", () => {
    const job = makeJob({ state: "running", stage: "chunking", corpus_ids: [3] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    const chunkEl = screen.getByText("Chunk");
    expect(chunkEl).toHaveAttribute("aria-current", "step");

    const parseEl = screen.getByText("Parse");
    expect(parseEl).not.toHaveAttribute("aria-current", "step");
  });

  it("renders Retry button for crashed state", () => {
    const onRetry = vi.fn();
    const job = makeJob({
      state: "crashed",
      note: "Timeout exceeded",
      corpus_ids: [10],
    });
    render(<QueueJobRow job={job} onRetry={onRetry} />);

    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    expect(screen.getByText(/timeout exceeded/i)).toBeInTheDocument();
  });

  it("calls onRetry with the job when Retry is clicked", () => {
    const onRetry = vi.fn();
    const job = makeJob({ state: "crashed", corpus_ids: [10] });
    render(<QueueJobRow job={job} onRetry={onRetry} />);

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledWith(job);
  });

  it("renders Retry button for cancelled state", () => {
    const job = makeJob({ state: "cancelled", corpus_ids: [7] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("does NOT render Retry button for running state", () => {
    const job = makeJob({ state: "running", stage: "embedding", corpus_ids: [5] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  });

  it("does NOT render Retry button for done state", () => {
    const job = makeJob({ state: "done", corpus_ids: [4] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  });

  it("shows 'Running' state label when running", () => {
    const job = makeJob({ state: "running", stage: "fetching", corpus_ids: [2] });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("shows 'Failed' state label for crashed", () => {
    const job = makeJob({ state: "crashed" });
    render(<QueueJobRow job={job} onRetry={() => {}} />);

    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
