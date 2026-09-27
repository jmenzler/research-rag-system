"""GUI-facing /api/* meta endpoints. Thin, no external probes."""
from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# SHA + BUILT_AT loader: prod-fast-path import; dev fallback to git rev-parse.
# _version.py is stamped by .github/workflows/deploy-pc.yml at deploy time (D-13).
try:
    from src.server._version import BUILT_AT, SHA  # type: ignore[import-not-found]
except ImportError:  # dev mode — no stamped file
    _result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    SHA = _result.stdout.strip() or "dev"
    BUILT_AT = None

# Captured ONCE at import — /api/health is sub-10ms.
_STARTED_AT = datetime.now(UTC).isoformat()

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str | bool]:
    """Liveness probe. No external deps. Sub-10ms."""
    return {"ok": True, "sha": SHA, "started_at": _STARTED_AT}


@router.get("/version")
def version() -> dict[str, str | None]:
    """Deployed git SHA + build timestamp. Drives reload banner (D-15)."""
    return {"sha": SHA, "built_at": BUILT_AT}
