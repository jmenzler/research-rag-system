import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/app")({
  // Bare /app has no index route (pages live under the pathless _layout), so it
  // dead-ended on a blank shell. Redirect it to the default surface. Guarded to
  // the exact path so child routes (/app/chat, /app/graph, …) pass through.
  beforeLoad: ({ location }) => {
    if (location.pathname === "/app" || location.pathname === "/app/") {
      throw redirect({ to: "/app/chat" });
    }
  },
  component: AppShell,
});

function AppShell() {
  // The brand + version SHA used to live in a global header here. Both moved
  // into the sidebar (SidebarHeader brand, footer VersionPill) so the shell is
  // just the routed outlet — no chrome duplicated against the p3 sidebar.
  return <Outlet />;
}
