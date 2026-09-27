"""RED — implemented in 03-02-PLAN.

Tests for 0004_corpus_fts.sql migration.

Pin the migration shape before it lands:
  - PRAGMA user_version bumps to >= 4
  - corpus_fts + corpus_fts_meta virtual tables exist after migration
  - Re-running migrate.run is idempotent (no error, version stable)

Analog:
  - tests/server/test_migrations_0003.py (Phase 2 — same patterns)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_all(db_path: Path) -> None:
    """Run the full migration chain on db_path."""
    from src.server.migrations import migrate

    migrate.run(db_path)


def test_user_version_bumps_to_at_least_4(tmp_path: Path) -> None:
    """RED — after 0004 migration, user_version >= 4."""
    db = tmp_path / "chats.db"
    _run_all(db)
    conn = sqlite3.connect(str(db))
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 4, f"expected user_version >= 4 after 0004; got {v}"
    finally:
        conn.close()


def test_corpus_fts_virtual_table_exists(tmp_path: Path) -> None:
    """RED — corpus_fts virtual table is created by 0004."""
    db = tmp_path / "chats.db"
    _run_all(db)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_fts'"
        ).fetchone()
        assert row is not None, "corpus_fts virtual table missing"
    finally:
        conn.close()


def test_corpus_fts_meta_watermark_exists(tmp_path: Path) -> None:
    """RED — corpus_fts_meta watermark row is created by 0004."""
    db = tmp_path / "chats.db"
    _run_all(db)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='corpus_fts_meta'"
        ).fetchone()
        assert row is not None, "corpus_fts_meta table missing"
    finally:
        conn.close()


def test_idempotent_re_apply_no_error(tmp_path: Path) -> None:
    """RED — re-running migrate.run after 0004 is a no-op (no error, version stable)."""
    db = tmp_path / "chats.db"
    _run_all(db)
    # second run — must not error
    _run_all(db)
    conn = sqlite3.connect(str(db))
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 4
    finally:
        conn.close()
