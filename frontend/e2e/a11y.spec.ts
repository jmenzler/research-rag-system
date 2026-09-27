import { test, expect } from "./fixtures/axe";

const ROUTES = [
  "/app/chat",
  "/app/corpus",
  "/app/queries",
  "/app/graph",
];

for (const route of ROUTES) {
  test(`axe clean: ${route}`, async ({ page, makeAxeBuilder }) => {
    await page.goto(route);
    // networkidle never settles — the app polls /api/version every 60s.
    // Wait for the shell's <main> landmark instead.
    await page.locator("main").first().waitFor({ state: "visible" });
    const results = await makeAxeBuilder().analyze();
    expect(results.violations).toEqual([]);
  });
}
