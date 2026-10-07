import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

/**
 * Production: a static export (`out/`) that the FastAPI backend serves at "/", so the
 * browser talks to one origin and never needs to know where any model runs.
 *
 * `next dev` (port 3000): static export forbids rewrites even in dev, so the dev server
 * runs without it and proxies the API to the backend on :8000. WebSockets are opened to
 * :8000 directly in dev (see lib/api.ts).
 */
const BACKEND = process.env.STUDIO_BACKEND ?? "http://127.0.0.1:8000";

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      async rewrites() {
        return [
          { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
          { source: "/sessions/:path*", destination: `${BACKEND}/sessions/:path*` },
          { source: "/sessions", destination: `${BACKEND}/sessions` },
        ];
      },
    };
  }
  return {
    output: "export",
    trailingSlash: true,
    images: { unoptimized: true },
  };
}
