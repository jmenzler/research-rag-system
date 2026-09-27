/**
 * Wave-0 RED test for <ComposeBox> (D-03, D-04).
 *
 * Implementation lands in Plan 06 (frontend/src/components/chat/ComposeBox.tsx).
 *
 * Behavioural contract:
 *   - Enter submits; Shift+Enter inserts newline.
 *   - Send button morphs to red Stop button while streaming (same focus target).
 *   - Click Stop calls onStop.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const toastMock = vi.fn();
vi.mock("sonner", () => ({
  toast: Object.assign((...args: unknown[]) => toastMock(...args), {
    error: toastMock,
  }),
}));

beforeEach(() => {
  toastMock.mockReset();
  vi.restoreAllMocks();
});

describe("ComposeBox", () => {
  it("Enter submits the form (D-03)", async () => {
    const { ComposeBox } = await import("@/components/chat/ComposeBox");
    const onSubmit = vi.fn();
    const onStop = vi.fn();
    render(
      <ComposeBox onSubmit={onSubmit} onStop={onStop} streaming={false} />,
    );

    const textarea = screen.getByRole("textbox");
    const user = userEvent.setup();
    await user.click(textarea);
    await user.keyboard("hello{Enter}");

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ content: "hello" }),
    );
  });

  it("Shift+Enter inserts newline (no submit)", async () => {
    const { ComposeBox } = await import("@/components/chat/ComposeBox");
    const onSubmit = vi.fn();
    const onStop = vi.fn();
    render(
      <ComposeBox onSubmit={onSubmit} onStop={onStop} streaming={false} />,
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    const user = userEvent.setup();
    await user.click(textarea);
    await user.keyboard("hello{Shift>}{Enter}{/Shift}world");

    expect(textarea.value).toContain("\n");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Send button morphs to Stop variant while streaming (D-04)", async () => {
    const { ComposeBox } = await import("@/components/chat/ComposeBox");
    render(
      <ComposeBox
        onSubmit={vi.fn()}
        onStop={vi.fn()}
        streaming={true}
      />,
    );

    const button = screen.getByTestId("send-button");
    expect(button.getAttribute("data-variant")).toBe("destructive");
  });

  it("Stop click calls onStop while streaming", async () => {
    const { ComposeBox } = await import("@/components/chat/ComposeBox");
    const onStop = vi.fn();
    render(
      <ComposeBox onSubmit={vi.fn()} onStop={onStop} streaming={true} />,
    );

    const button = screen.getByTestId("send-button");
    const user = userEvent.setup();
    await user.click(button);
    expect(onStop).toHaveBeenCalled();
  });
});
