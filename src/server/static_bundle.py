"""SPA bundle mount + catch-all SPA fallback.

Wires:
- GET /            -> 302 redirect to /app
- GET /app         -> index.html
- GET /app/{path:path} -> index.html (TanStack Router handles client-side routing)
- GET /assets/*    -> StaticFiles with long-cache immutable headers

Mount order MUST be: include_router(meta_router) first, then register(app).
The /app/{path:path} catch-all is registered LAST so /api/* routes are matched first.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "src" / "server" / "static"

_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_NO_STORE = "no-store"


def _index_response() -> FileResponse:
    """Return index.html with Cache-Control: no-store (MOD-13)."""
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"cache-control": _NO_STORE},
    )


def _bundle_missing_response() -> JSONResponse:
    """Dev-mode fallback when the bundle has not been built yet."""
    return JSONResponse(
        status_code=503,
        content={
            "error": (
                "SPA bundle not built — run "
                "`npm run build -- --outDir ../src/server/static` in frontend/"
            )
        },
    )


class _ImmutableStaticFiles(StaticFiles):
    """StaticFiles subclass that emits long-cache immutable headers on /assets/*."""

    async def get_response(self, path: str, scope: dict[str, object]) -> object:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["cache-control"] = _IMMUTABLE_CACHE
        return response


def register(app: FastAPI) -> None:
    """Register SPA routes on the composer's FastAPI app.

    MUST be called AFTER app.include_router(meta_router) — see mount-order rule.
    Tolerant of missing src/server/static/ directory (dev mode without a built bundle).
    """
    bundle_present = (STATIC_DIR / "index.html").is_file()
    assets_dir = STATIC_DIR / "assets"

    if bundle_present and assets_dir.is_dir():
        app.mount(
            "/assets",
            _ImmutableStaticFiles(directory=assets_dir, html=False),
            name="assets",
        )
    else:
        logger.warning(
            "SPA bundle not built (missing %s). /app will return 503; "
            "build with `npm run build -- --outDir ../src/server/static` in frontend/.",
            STATIC_DIR / "index.html",
        )

    @app.get("/", include_in_schema=False)
    def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/app", status_code=302)

    # MUST be the LAST route registered. Catch-all for /app/* deep links.
    @app.get("/app/{full_path:path}", include_in_schema=False, response_model=None)
    @app.get("/app", include_in_schema=False, response_model=None)
    def _spa_fallback(full_path: str = "") -> FileResponse | JSONResponse:
        if not bundle_present:
            return _bundle_missing_response()
        return _index_response()
