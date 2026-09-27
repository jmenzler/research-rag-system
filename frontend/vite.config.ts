import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import path from "node:path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const proxyTarget = env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8766";

  // D-05 proxy list: /api/*, /query, /notebooks, /discover/*, /monitor/*, /health
  const proxy = Object.fromEntries(
    ["/api", "/query", "/notebooks", "/discover", "/monitor", "/health"].map((p) => [
      p,
      { target: proxyTarget, changeOrigin: true, ws: false },
    ]),
  );

  return {
    plugins: [
      TanStackRouterVite({ routesDirectory: "./src/routes" }),
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: { "@": path.resolve(__dirname, "./src") },
      // Force ONE physical sigma/graphology module shared between the app and
      // @react-sigma/core. Without this, vite's dep pre-bundling inlines sigma
      // into @react-sigma's optimized chunk while the app's `sigma/rendering`
      // import becomes a separate chunk — two distinct NodeCircleProgram class
      // identities. Registering the app's class in nodeProgramClasses then never
      // matches what the @react-sigma-bundled Sigma expects, so node type
      // "circle" has no usable program at render → addNodeToProgram throws.
      dedupe: ["sigma", "graphology"],
    },
    // Pre-bundle the sigma stack together so all consumers resolve to the same
    // optimized module (single class identity for the program classes).
    optimizeDeps: {
      include: [
        "@react-sigma/core",
        "@react-sigma/layout-forceatlas2",
        "sigma",
        "sigma/rendering",
        "graphology",
      ],
    },
    build: {
      target: "es2022",
      sourcemap: true,
      // Plan 02-13 — split the markdown-and-math toolchain out of the main
      // bundle so the chat surface (which doesn't render math on every turn)
      // doesn't pay for KaTeX's ~250KB of CSS+font on initial paint.
      // RESEARCH §Pitfall 5: katex MUST be in its own chunk separately from
      // the rest of the markdown stack so it can be lazy-imported on demand.
      rollupOptions: {
        output: {
          manualChunks: (id: string): string | undefined => {
            if (id.includes("node_modules/katex")) return "katex";
            if (
              id.includes("node_modules/react-markdown") ||
              id.includes("node_modules/remark") ||
              id.includes("node_modules/rehype") ||
              id.includes("node_modules/mdast") ||
              id.includes("node_modules/hast")
            ) {
              return "markdown";
            }
            if (
              id.includes("node_modules/sigma") ||
              id.includes("node_modules/graphology") ||
              id.includes("node_modules/@react-sigma")
            ) {
              return "graph-viz";
            }
            return undefined;
          },
        },
      },
    },
    server: { proxy },
    preview: { proxy },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./src/__tests__/setup.ts"],
      include: ["src/**/*.{test,spec}.{ts,tsx}"],
    },
  };
});
