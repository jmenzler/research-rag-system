-- 0005_maps.sql — Phase 5 saved Maps (graph-page entities, not chat-bound).

-- Standalone maps table: no FOREIGN KEY to chats (D-09 single-user, graph-page
-- entity whose lifecycle is independent of any conversation).
-- All CREATE statements use IF NOT EXISTS (idempotent on re-run).

CREATE TABLE IF NOT EXISTS maps (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    collection  TEXT NOT NULL,
    snapshot    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_maps_collection ON maps(collection);

PRAGMA user_version = 5;
