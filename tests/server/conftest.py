"""Shared pytest fixtures for Phase 1 backend server tests.

Extracts the monkeypatch pattern from tests/server/test_audit_smoke.py
(Phase 0 — the ONE sanctioned way to stub the LLM call site) into reusable
fixtures so that test_audit_smoke.py + test_e2e_chat.py share one source of
truth.

All fixtures here are pytest-discovery automatic — no explicit import in tests.

Brownfield invariant: every fixture sandboxes state via pytest's tmp_path +
monkeypatch primitives — at teardown all monkeypatches are reverted and tmp_path
is deleted. Production state (parents/chats.db, logs/queries/) is never touched.
"""

from __future__ import annotations

import sqlite3
import types
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from src.server.discover_queue import DiscoverStartRequest


@pytest.fixture
def fake_genai_client(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Monkeypatch ``genai.Client`` to return a deterministic synthetic client.

    Verbatim extraction of tests/server/test_audit_smoke.py:115-149 — the Phase 0
    sanctioned pattern for stubbing the only authorised LLM call boundary
    (``usage_track.call_text`` uses ``genai.Client(api_key=...)`` then
    ``client.models.generate_content(...)``). The fake response carries
    ``usage_metadata`` so the full ``call_text`` code path executes — including
    usage extraction and cost computation — without outbound HTTP.

    Yields the fake client so a test can override ``generate_content`` for a
    specific case if needed.
    """
    fake_usage = types.SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
        cached_content_token_count=0,
    )
    fake_response = types.SimpleNamespace(
        text="stub answer",
        usage_metadata=fake_usage,
    )
    fake_models = types.SimpleNamespace(
        generate_content=lambda **kwargs: fake_response,
    )
    fake_client = types.SimpleNamespace(models=fake_models)

    from google import genai as _genai_mod

    from src import config as _config

    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)
    monkeypatch.setattr(_config, "DECOMPOSE_MODEL", "gemini-2.5-flash")
    return fake_client


@pytest.fixture
def fake_milvus_search(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Monkeypatch ``milvus_search_only`` to return an empty hit list.

    The real function lives in ``src.query.retrieve.milvus_search_only`` (also
    re-exported via ``src.query.__init__``). The query pipeline imports it via
    ``from src.query.retrieve import ... milvus_search_only`` at call time, so
    monkeypatching the ``src.query.retrieve`` module attribute is the correct
    point of interception.

    Empty hit list keeps the pipeline writing decompose + milvus + rerank
    stages (each calls ``audit.write_stage`` BEFORE the empty-retrieved
    short-circuit). Synthesis is skipped — that is the lighter-weight path
    that does not require a live cross-encoder.

    Returns the chunks list so tests can override/inspect.
    """
    import src.query.retrieve as _retr

    chunks: list[Any] = []

    monkeypatch.setattr(_retr, "milvus_search_only", lambda *a, **k: chunks)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)
    return chunks


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``get_reranker()`` to a deterministic identity-style stub.

    Defensive: the empty-hit ``fake_milvus_search`` short-circuits before
    rerank in the production code path, so this fixture is a belt-and-
    suspenders no-op for tests that don't need any chunks to flow through
    the cross-encoder. Tests that DO need real chunks override
    ``fake_milvus_search`` with a non-empty list first; this stub then
    re-scores deterministically without loading the cross-encoder.

    The attribute set is ``raising=False`` because not every src.query
    module exposes ``get_reranker`` at the top level (it lives in
    ``src.query.rerank`` in some layouts and ``src.query.reranker`` in
    others — the monkeypatch is permissive so the fixture can be requested
    by any test without forcing the import).
    """

    class _FakeReranker:
        def rerank(self, query: str, chunks: list[Any]) -> list[Any]:
            for i, c in enumerate(chunks):
                if hasattr(c, "score_rerank"):
                    try:
                        c.score_rerank = (getattr(c, "score_dense", 0.5) or 0.5) + 0.01 * i
                    except AttributeError:
                        pass
            return chunks

    fake = _FakeReranker()

    # Try both well-known dotted paths; importing the parent module is wrapped
    # in try/except because ``monkeypatch.setattr("a.b.c", ...)`` imports
    # ``a.b`` first and the import error is unrecoverable even with
    # ``raising=False``. Tolerate either layout silently.
    import importlib

    for dotted in ("src.query.rerank", "src.query.reranker"):
        try:
            mod = importlib.import_module(dotted)
        except ImportError:
            continue
        if hasattr(mod, "get_reranker"):
            monkeypatch.setattr(mod, "get_reranker", lambda: fake)
    return None


@pytest.fixture
def audit_log_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``QUERY_LOG_ROOT`` to ``tmp_path/queries``.

    ``src/query/query_logger.make_query_logger`` reads the env var
    ``QUERY_LOG_ROOT`` (default ``logs/queries``). Setting it to a tmp_path
    sub-dir ensures all audit-trail writes go under the test sandbox and
    are cleaned up automatically.

    Returns the resolved root path so tests can ``rglob`` into it.
    """
    root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(root))
    return root


@pytest.fixture
def ephemeral_chats_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sandbox ``chats.db`` under ``tmp_path``.

    ``src.server.chats_store.CHATS_DB_PATH`` is a module-level constant that
    ``open_chats_conn`` reads at call time (NOT at import time). Therefore
    monkeypatching the module attribute is sufficient — no need for an env
    var (chats_store does not honour one).

    The fixture also runs the migration so the schema is present.
    """
    db = tmp_path / "chats.db"

    # Apply migrations first so 0002_chats.sql lands.
    from src.server.migrations import migrate

    migrate.run(db)

    import src.server.chats_store as _cs

    monkeypatch.setattr(_cs, "CHATS_DB_PATH", db)
    return db


# ---------------------------------------------------------------------------
# Phase 5: fake_queue, tmp_chats_db, log_tail_samples
# ---------------------------------------------------------------------------


class _FakeQueueEntry:
    """Minimal stand-in for QueueEntry — only run_id is needed by tests."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class FakeDiscoverQueue:
    """Records enqueue() calls and returns scriptable get_state() shapes.

    The cap behaviour (MAX_QUERIES_PER_REQUEST) lives in the real
    DiscoverStartRequest model's field_validator.  This fake does NOT
    re-implement the cap — it only records calls and returns state.

    Usage in tests:
        fq = FakeDiscoverQueue()
        fq.script[run_id] = {"state": "done", "run_id": run_id, ...}
        entry = fq.enqueue(some_request)
        assert fq.get_state(entry.run_id) == fq.script[entry.run_id]
    """

    def __init__(self) -> None:
        from src.server.discover_queue import DiscoverStartRequest  # noqa: PLC0415

        self._DiscoverStartRequest = DiscoverStartRequest
        self.calls: list[Any] = []
        self.url_calls: list[dict[str, Any]] = []
        self.script: dict[str, dict[str, Any]] = {}

    def enqueue(self, req: DiscoverStartRequest) -> _FakeQueueEntry:
        self.calls.append(req)
        run_id = uuid.uuid4().hex[:12]
        self.script[run_id] = {"state": "queued", "run_id": run_id, "position": 0, "queue_depth": 1}
        return _FakeQueueEntry(run_id=run_id)

    def enqueue_urls(self, *, urls: list[str], collection: str, partition: str) -> _FakeQueueEntry:
        self.url_calls.append({"urls": urls, "collection": collection, "partition": partition})
        run_id = uuid.uuid4().hex[:12]
        self.script[run_id] = {"state": "queued", "run_id": run_id, "position": 0, "queue_depth": 1}
        return _FakeQueueEntry(run_id=run_id)

    def get_state(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.script:
            raise KeyError(f"run_id {run_id!r} not scripted in FakeDiscoverQueue")
        return self.script[run_id]


@pytest.fixture
def fake_queue() -> FakeDiscoverQueue:
    """FakeDiscoverQueue instance scoped to a single test."""
    return FakeDiscoverQueue()


_MAPS_DDL = """
CREATE TABLE IF NOT EXISTS maps (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    collection  TEXT NOT NULL,
    snapshot    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_maps_collection ON maps(collection);
"""


@pytest.fixture
def tmp_chats_db(tmp_path: Path) -> Path:
    """Yield a path to a fresh temp sqlite file with the maps table DDL applied.

    Each test gets an isolated DB under tmp_path.  The maps DDL is inlined
    here (the real migration file 0005_maps.sql is built in Plan 02).
    """
    db = tmp_path / "chats.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(_MAPS_DDL)
        conn.commit()
    finally:
        conn.close()
    return db


@pytest.fixture
def log_tail_samples() -> dict[str, list[str]]:
    """Representative log_tail lines per INGEST-03 stage (verbatim from RESEARCH.md Flag 2).

    "fetching"  : Deep Research trigger line (Phase A)
    "parsing"   : ragctl parse_start / progress line (Phase B)
    "chunking"  : Found N file(s) + filename bracket (Phase C start)
    "embedding" : parents/children embedding line (Phase C mid)
    "unknown"   : only pre-stage chrome, no stage marker
    """
    return {
        "fetching": ["[discover:2305.12345] triggering Deep Research mode=fast…"],
        "parsing": [
            "  → parse_start: trading pdf=/x.pdf",
            "  [progress] trading:1/3 pdf:1/3 (parse_pending=2)",
        ],
        "chunking": [
            "Found 193 file(s) to process.",
            "[007__1_strategy_layer_overview.txt]",
        ],
        "embedding": [
            "[007__1_strategy_layer_overview.txt]",
            "  1 parents, 2 children — embedding…",
        ],
        "unknown": ["[discover:2305.12345] creating notebook…"],
    }


@pytest.fixture
def client_with_chats(
    ephemeral_chats_db: Path,
    audit_log_root: Path,
    fake_genai_client: types.SimpleNamespace,
    fake_milvus_search: list[Any],
    fake_reranker: None,
) -> Iterator[Any]:
    """``TestClient(app)`` with all externals stubbed and ephemeral state.

    Composes the lower-level fixtures into the single client the e2e tests
    consume. Production-app composer is used (``src.server.app.app``) so the
    lifespan runs migrations and all routers are wired — the same surface
    the GUI hits.

    Note: ``client_with_chats`` does NOT enter ``TestClient``'s lifespan
    context (``with TestClient(app) as client``) because ``app.lifespan``
    re-runs migrations against the production ``parents/chats.db`` (env-
    independent), which would touch production state. Instead we monkeypatch
    ``CHATS_DB_PATH`` BEFORE the client is created so any code path that
    re-reads the constant picks up the sandbox path. Migrations on the
    sandbox were already applied by ``ephemeral_chats_db``.

    For Phase 1 tests that only exercise the chat-related routers (which
    open their own connection via ``open_chats_conn`` at request time), this
    is sufficient.
    """
    from fastapi.testclient import TestClient

    from src.server.app import app

    yield TestClient(app)
