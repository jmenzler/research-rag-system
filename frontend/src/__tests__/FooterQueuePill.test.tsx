/**
 * Tests for FooterQueuePill component (Plan 05-05 Task 2).
 *
 * Covers:
 * - Renders "Queue idle" when counts are {0,0}
 * - Renders "◐ N ingesting · M queued" when counts are active
 *
 * jsdom-safe: mocks useQueueStream so no EventSource.
 */
import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/hooks/useQueueStream", () => ({
  useQueueStream: vi.fn(),
}));

import { useQueueStream } from "@/hooks/useQueueStream";
import { FooterQueuePill } from "@/components/graph/FooterQueuePill";

const mockUseQueueStream = vi.mocked(useQueueStream);

describe("FooterQueuePill", () => {
  it("renders 'Queue idle' when counts are {0,0}", () => {
    mockUseQueueStream.mockReturnValue({
      jobs: {},
      counts: { running: 0, queued: 0 },
      isConnected: false,
    });

    render(<FooterQueuePill />);
    expect(screen.getByTestId("footer-queue-pill")).toHaveTextContent("Queue idle");
  });

  it("renders active text when running=3 and queued=1", () => {
    mockUseQueueStream.mockReturnValue({
      jobs: {},
      counts: { running: 3, queued: 1 },
      isConnected: true,
    });

    render(<FooterQueuePill />);
    const pill = screen.getByTestId("footer-queue-pill");
    expect(pill).toHaveTextContent("3 ingesting");
    expect(pill).toHaveTextContent("1 queued");
  });

  it("renders active text when running=1 and queued=0", () => {
    mockUseQueueStream.mockReturnValue({
      jobs: {},
      counts: { running: 1, queued: 0 },
      isConnected: true,
    });

    render(<FooterQueuePill />);
    const pill = screen.getByTestId("footer-queue-pill");
    expect(pill).toHaveTextContent("1 ingesting");
    expect(pill).toHaveTextContent("0 queued");
  });

  it("renders 'Queue idle' when running=0 and queued=0 even if connected", () => {
    mockUseQueueStream.mockReturnValue({
      jobs: {},
      counts: { running: 0, queued: 0 },
      isConnected: true,
    });

    render(<FooterQueuePill />);
    expect(screen.getByTestId("footer-queue-pill")).toHaveTextContent("Queue idle");
  });
});
