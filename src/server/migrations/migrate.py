"""Sqlite migration runner for parents/chats.db.

Applies all *.sql files in src/server/migrations/ in numeric-prefix order,
tracked via PRAGMA user_version. Each migration runs inside its own
BEGIN/COMMIT transaction with ROLLBACK on exception — partial application
is impossible.

Wired into src/server/app.py lifespan (D-19): failure raises → uvicorn fails
to bind → systemd marks unit failed → deploy /api/health curl gate fails
→ deploy aborts (INFRA-08).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "parents" / "chats.db"


def _parse_statements(sql: str) -> list[str]:
    """Parse semicolon-delimited SQL into a list of non-empty executable statements.

    Handles:
    - ``-- ...`` line comments (stripped BEFORE splitting on ``;`` so a
      semicolon inside a comment never splits a statement).
    - SQLite ``CREATE TRIGGER ... BEGIN ...; ...; END;`` compound blocks
      (the inner semicolons are NOT statement boundaries — the whole
      ``CREATE TRIGGER`` lives in one statement until the matching ``END;``).
    - Blank lines (ignored).

    Returns only statements with actual SQL content.
    """
    # 1. Strip ``--`` line comments. Anything after ``--`` on a line is a
    #    comment; remove just that tail. This must run BEFORE we split on
    #    ``;`` so a ``;`` inside a comment is never treated as a separator.
    stripped_lines: list[str] = []
    for line in sql.splitlines():
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        if line.strip():
            stripped_lines.append(line)
    cleaned = "\n".join(stripped_lines)

    # 2. Walk char-by-char, splitting on ``;`` UNLESS we are inside a
    #    ``BEGIN ... END`` block (case-insensitive, whole-word). Trigger
    #    bodies contain ``;`` separators that must not split the statement.
    result: list[str] = []
    buf: list[str] = []
    depth = 0  # nesting count of unmatched BEGIN keywords
    upper = cleaned.upper()
    i = 0
    n = len(cleaned)

    def _is_word_boundary(text: str, start: int, end: int) -> bool:
        before_ok = start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_")
        after_ok = end >= len(text) or not (text[end].isalnum() or text[end] == "_")
        return before_ok and after_ok

    while i < n:
        ch = cleaned[i]
        # Match BEGIN / END as whole words (case-insensitive via ``upper``).
        if upper.startswith("BEGIN", i) and _is_word_boundary(upper, i, i + 5):
            depth += 1
            buf.append(cleaned[i : i + 5])
            i += 5
            continue
        if upper.startswith("END", i) and _is_word_boundary(upper, i, i + 3) and depth > 0:
            depth -= 1
            buf.append(cleaned[i : i + 3])
            i += 3
            continue
        if ch == ";" and depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                # Collapse interior whitespace for log/exec hygiene.
                result.append(" ".join(stmt.split()))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    # Tail (statement without trailing ``;``).
    tail = "".join(buf).strip()
    if tail:
        result.append(" ".join(tail.split()))
    return result


def run(db_path: Path) -> None:
    """Apply all pending migrations to ``db_path`` in numeric-prefix order.

    Idempotent across restarts — already-applied migrations are skipped via
    PRAGMA user_version. Each migration runs inside an explicit BEGIN/COMMIT
    transaction with ROLLBACK on exception — partial application is impossible.

    SQLite PRAGMA statements that change journal mode (e.g. ``PRAGMA
    journal_mode = WAL``) cannot run inside a BEGIN block.  These are executed
    before the transaction opens. All other SQL runs inside BEGIN/COMMIT.
    The user_version bump is always inside the transaction so a DDL failure
    leaves user_version unchanged.

    Raises:
        sqlite3.Error: If any migration's SQL fails. Caller (lifespan in
            app.py) MUST let this propagate so uvicorn fails to bind.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        pending = sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql"))
        for sql_path in pending:
            version = int(sql_path.stem.split("_", 1)[0])
            if version <= current:
                continue
            sql = sql_path.read_text()
            logger.info(
                "Applying migration %s (user_version %d -> %d)",
                sql_path.name,
                current,
                version,
            )
            statements = _parse_statements(sql)
            # Separate PRAGMA journal_mode changes (cannot run inside BEGIN)
            # from all other DDL (must run inside BEGIN for atomicity).
            pragma_stmts = [
                s for s in statements
                if s.upper().startswith("PRAGMA JOURNAL_MODE")
            ]
            ddl_stmts = [
                s for s in statements
                if not s.upper().startswith("PRAGMA JOURNAL_MODE")
            ]
            # 1. Run journal_mode PRAGMAs outside the transaction (idempotent).
            for stmt in pragma_stmts:
                conn.execute(stmt)
            # 2. Run DDL + user_version bump inside an explicit transaction.
            conn.execute("BEGIN")
            try:
                for stmt in ddl_stmts:
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {version}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                logger.error("Migration %s failed; rolled back.", sql_path.name)
                raise
            current = version
    finally:
        conn.close()
