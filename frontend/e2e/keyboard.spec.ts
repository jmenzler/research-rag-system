import { test, expect } from "@playwright/test";

test.describe("POLISH-05: global keyboard shortcuts", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/app/chat");
    // networkidle never settles — the app polls /api/version every 60s.
    // Wait for the shell's <main> landmark (hotkeys are registered there).
    await page.locator("main").first().waitFor({ state: "visible" });
  });

  test("Meta+k opens command palette", async ({ page }) => {
    await page.keyboard.press("Meta+k");
    await expect(page.getByRole("dialog")).toBeVisible();
  });

  test("Meta+/ opens keyboard help dialog", async ({ page }) => {
    await page.keyboard.press("Meta+/");
    await expect(page.getByRole("dialog")).toBeVisible();
  });

  test("g g sequence navigates to graph route", async ({ page }) => {
    await page.keyboard.press("g");
    await page.keyboard.press("g");
    await expect(page).toHaveURL(/\/app\/graph/);
  });

  test("Meta+b toggles sidebar grid column", async ({ page }) => {
    // The sidebar collapses/expands via the layout grid column (17rem <-> 3rem),
    // not by hiding an element — assert the rendered <aside> width changes.
    const sidebar = page.locator("aside").first();
    const before = (await sidebar.boundingBox())?.width ?? 0;
    await page.keyboard.press("Meta+b");
    await expect
      .poll(async () => (await sidebar.boundingBox())?.width ?? 0)
      .not.toBe(before);
  });
});
