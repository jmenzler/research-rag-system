"""Compose API routes, static frontend serving, and SQLite startup migrations."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from src.server import api_chats, api_graph, api_ingest, api_papers, api_queries, static_bundle
from src.server.api import app as app  # re-export existing FastAPI instance
from src.server.api_meta import router as meta_router
from src.server.migrations import migrate

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHATS_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run sqlite migrations on startup. Failure refuses to bind."""
    logger.info("Running sqlite migrations on %s", CHATS_DB_PATH)
    migrate.run(CHATS_DB_PATH)
    logger.info("Migrations complete; user_version up to date.")
    yield


app.add_middleware(GZipMiddleware, minimum_size=1024)

# Register API routes before the frontend fallback.
app.include_router(meta_router)
app.include_router(api_chats.router)
app.include_router(api_papers.router)
app.include_router(api_queries.router)
app.include_router(api_graph.router)
app.include_router(api_ingest.router)
static_bundle.register(app)

# Attach lifespan to the existing app (no FastAPI re-instantiation — brownfield).
# FastAPI 0.122+ supports replacing lifespan_context on an already-built app.
app.router.lifespan_context = lifespan
