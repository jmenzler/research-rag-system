-- 0002_chats.sql — Phase 1 chat persistence (D-16).
-- Four tables, no FTS5 (lands in Phase 2 / HIST-02 as 0003_chats_fts.sql).
-- All FKs CASCADE on delete (CONTEXT.md "Claude's Discretion" — single editable per-chat dataset).

CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    retriever   TEXT NOT NULL DEFAULT 'milvus',
    collections TEXT,
    version     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    chat_id      TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    query_id     TEXT,
    message_uuid TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    version      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_query_id ON messages(query_id);

CREATE TABLE IF NOT EXISTS citations (
    message_id   TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    marker       INTEGER NOT NULL,
    child_id     TEXT,
    parent_id    TEXT,
    paper_id     TEXT,
    score_dense  REAL,
    score_sparse REAL,
    score_rerank REAL,
    resolved     INTEGER NOT NULL CHECK (resolved IN (0, 1)),
    PRIMARY KEY (message_id, marker)
);

CREATE INDEX IF NOT EXISTS idx_citations_message_id ON citations(message_id);

CREATE TABLE IF NOT EXISTS message_meta (
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    PRIMARY KEY (message_id, key)
);
