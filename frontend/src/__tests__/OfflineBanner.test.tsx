/**
 * RED — OfflineBanner unit tests (06-02 Task 3).
 *
 * Behavior cases:
 *   1. Renders null when healthFailureStreak < 3.
 *   2. Renders role="alert" aria-live="assertive" when healthFailureStreak >= 3.
 *   3. Has data-testid="offline-banner" when visible.
 */
import { describe, expect, it, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useUiStore } from "@/state/uiStore";

describe("OfflineBanner", () => {
  beforeEach(() => {
    useUiStore.setState({ healthFailureStreak: 0 });
  });

  it("renders nothing when healthFailureStreak is 0", async () => {
    const { OfflineBanner } = await import("@/components/OfflineBanner");
    useUiStore.setState({ healthFailureStreak: 0 });

    const { container } = render(<OfflineBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when healthFailureStreak is 2 (below threshold)", async () => {
    const { OfflineBanner } = await import("@/components/OfflineBanner");
    useUiStore.setState({ healthFailureStreak: 2 });

    const { container } = render(<OfflineBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("renders role=alert with aria-live=assertive when healthFailureStreak is 3", async () => {
    const { OfflineBanner } = await import("@/components/OfflineBanner");
    useUiStore.setState({ healthFailureStreak: 3 });

    render(<OfflineBanner />);

    const banner = screen.getByRole("alert");
    expect(banner).toBeDefined();
    expect(banner.getAttribute("aria-live")).toBe("assertive");
  });

  it("renders data-testid=offline-banner when healthFailureStreak >= 3", async () => {
    const { OfflineBanner } = await import("@/components/OfflineBanner");
    useUiStore.setState({ healthFailureStreak: 5 });

    render(<OfflineBanner />);

    const banner = screen.getByTestId("offline-banner");
    expect(banner).toBeDefined();
  });

  it("banner text mentions backend/connection", async () => {
    const { OfflineBanner } = await import("@/components/OfflineBanner");
    useUiStore.setState({ healthFailureStreak: 3 });

    render(<OfflineBanner />);

    const banner = screen.getByRole("alert");
    expect(banner.textContent?.toLowerCase()).toMatch(/backend|connection|unreachable/);
  });
});
