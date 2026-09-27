-- 0003_chats_fts.sql — Phase 2 FTS5 + archived + per-turn retriever provenance.
-- Idempotent: every CREATE uses IF NOT EXISTS. The migration runner
-- (src/server/migrations/migrate.py) skips this entire file on subsequent
-- boots via PRAGMA user_version, so the IF NOT EXISTS guards only fire on
-- the first apply.

-- 1. Add chats.archived column (boolean as INTEGER, default 0 = active).
ALTER TABLE chats ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;

-- 2. Add messages.retriever column (text, nullable so the column-add
--    is fast; backfilled in step 3).
ALTER TABLE messages ADD COLUMN retriever TEXT;

-- 3. Backfill messages.retriever from chats.retriever for existing Phase 1
--    rows (RESEARCH §Pitfall 3). Runs BEFORE triggers are created so the
--    UPDATE does not double-fire the FTS sync.
UPDATE messages
   SET retriever = (SELECT c.retriever FROM chats c WHERE c.id = messages.chat_id)
 WHERE retriever IS NULL;

-- 4. FTS5 virtual table — denormalized title + content per message row.
CREATE VIRTUAL TABLE IF NOT EXISTS chats_fts USING fts5(
    chat_id UNINDEXED,
    title,
    content,
    tokenize = 'porter unicode61'
);

-- 5. Sync triggers — FOUR triggers (RESEARCH §Pitfall 2).
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO chats_fts(rowid, chat_id, title, content)
        SELECT new.rowid, c.id, c.title, new.content
          FROM chats c
         WHERE c.id = new.chat_id;
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM chats_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM chats_fts WHERE rowid = old.rowid;
    INSERT INTO chats_fts(rowid, chat_id, title, content)
        SELECT new.rowid, c.id, c.title, new.content
          FROM chats c
         WHERE c.id = new.chat_id;
END;

-- chats_title_au keeps chats_fts.title in sync after a rename. If the chat
-- has no messages yet, insert a title-only "sentinel" row with rowid =
-- -1 * chats.rowid so the chat is still discoverable via title search
-- (negative rowids cannot collide with messages.rowid, which is always > 0).
CREATE TRIGGER IF NOT EXISTS chats_title_au AFTER UPDATE OF title ON chats BEGIN
    DELETE FROM chats_fts WHERE chat_id = new.id;
    INSERT INTO chats_fts(rowid, chat_id, title, content)
        SELECT m.rowid, new.id, new.title, m.content
          FROM messages m
         WHERE m.chat_id = new.id;
    INSERT INTO chats_fts(rowid, chat_id, title, content)
        SELECT -new.rowid, new.id, new.title, ''
         WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = new.id);
END;

-- 6. Backfill chats_fts with rows for messages that existed before this
--    migration. Triggers handle new rows from this point forward.
INSERT INTO chats_fts(rowid, chat_id, title, content)
    SELECT m.rowid, c.id, c.title, m.content
      FROM messages m
      JOIN chats c ON c.id = m.chat_id
     WHERE NOT EXISTS (SELECT 1 FROM chats_fts f WHERE f.rowid = m.rowid);

-- 7. Filter indexes for sidebar performance (HIST-04).
CREATE INDEX IF NOT EXISTS idx_chats_archived ON chats(archived);
CREATE INDEX IF NOT EXISTS idx_messages_retriever ON messages(retriever);
