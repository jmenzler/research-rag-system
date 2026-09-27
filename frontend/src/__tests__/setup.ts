import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement window.scrollTo. @tanstack/router's scroll-restoration
// calls it during the React commit phase when a router-aware component (e.g.
// GraphCanvas/GraphPage) renders; the "Not implemented" throw can abort the commit
// before effects like onGraphReady fire (GRAPH-09 pin test). Define a no-op stub so
// renders are deterministic regardless of test ordering or vi.unstubAllGlobals().
function stubScrollTo(): void {
  if (typeof window === "undefined") return;
  Object.defineProperty(window, "scrollTo", {
    value: () => {},
    writable: true,
    configurable: true,
  });
}

// Apply once at module load and again before every test — a test that calls
// vi.unstubAllGlobals() (e.g. GraphCanvasPin) can otherwise restore jsdom's
// throwing implementation between tests.
stubScrollTo();
beforeEach(() => stubScrollTo());

afterEach(() => cleanup());
