import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { VersionPill } from "@/components/VersionPill";
import { useUiStore } from "@/state/uiStore";

// Mock sonner at module-level so we can assert the toast call.
const toastMock = vi.fn();
vi.mock("sonner", () => ({ toast: (...args: unknown[]) => toastMock(...args) }));

function renderWithClient(ui: React.ReactNode): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  toastMock.mockClear();
  useUiStore.setState({ firstSeenSha: null, healthFailureStreak: 0 });
});

describe("VersionPill", () => {
  it("renders '?' while pending", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {}))); // never resolves
    renderWithClient(<VersionPill />);
    expect(screen.getByTestId("version-pill")).toHaveTextContent("?");
  });

  it("renders first 7 chars of SHA on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ sha: "abc1234deadbeef", built_at: null }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      ),
    );
    renderWithClient(<VersionPill />);
    await waitFor(() =>
      expect(screen.getByTestId("version-pill")).toHaveTextContent("abc1234"),
    );
  });

  it("renders 'offline' on fetch error and increments healthFailureStreak", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response("err", { status: 500 }))),
    );
    renderWithClient(<VersionPill />);
    await waitFor(() =>
      expect(screen.getByTestId("version-pill")).toHaveTextContent("offline"),
    );
    expect(useUiStore.getState().healthFailureStreak).toBeGreaterThan(0);
  });

  it("fires a Sonner toast when SHA changes between requests (D-15)", async () => {
    useUiStore.setState({ firstSeenSha: "oldsha12345" });
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ sha: "newsha67890", built_at: null }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        ),
      ),
    );
    renderWithClient(<VersionPill />);
    await waitFor(() => {
      expect(toastMock).toHaveBeenCalled();
      const [msg] = toastMock.mock.calls[0]!;
      expect(String(msg)).toMatch(/new version/i);
    });
  });
});
