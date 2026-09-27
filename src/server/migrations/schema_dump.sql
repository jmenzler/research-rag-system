CREATE TABLE chats ( id TEXT PRIMARY KEY, title TEXT, retriever TEXT NOT NULL DEFAULT 'milvus', collections TEXT, version INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP , archived INTEGER NOT NULL DEFAULT 0)
CREATE VIRTUAL TABLE chats_fts USING fts5( chat_id UNINDEXED, title, content, tokenize = 'porter unicode61' )
CREATE TABLE 'chats_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID
CREATE TABLE 'chats_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2)
CREATE TABLE 'chats_fts_data'(id INTEGER PRIMARY KEY, block BLOB)
CREATE TABLE 'chats_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)
CREATE TABLE 'chats_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID
CREATE TRIGGER chats_title_au AFTER UPDATE OF title ON chats BEGIN DELETE FROM chats_fts WHERE chat_id = new.id; INSERT INTO chats_fts(rowid, chat_id, title, content) SELECT m.rowid, new.id, new.title, m.content FROM messages m WHERE m.chat_id = new.id; INSERT INTO chats_fts(rowid, chat_id, title, content) SELECT -new.rowid, new.id, new.title, '' WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = new.id); END
CREATE TABLE citations ( message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE, marker INTEGER NOT NULL, child_id TEXT, parent_id TEXT, paper_id TEXT, score_dense REAL, score_sparse REAL, score_rerank REAL, resolved INTEGER NOT NULL CHECK (resolved IN (0, 1)), PRIMARY KEY (message_id, marker) )
CREATE VIRTUAL TABLE corpus_fts USING fts5( paper_id UNINDEXED, title, authors, year, short_cite, arxiv_id, doi, tokenize='porter unicode61' )
CREATE TABLE 'corpus_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID
CREATE TABLE 'corpus_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2, c3, c4, c5, c6)
CREATE TABLE 'corpus_fts_data'(id INTEGER PRIMARY KEY, block BLOB)
CREATE TABLE 'corpus_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)
CREATE TABLE 'corpus_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID
CREATE TABLE corpus_fts_meta ( id INTEGER PRIMARY KEY CHECK (id = 1), watermark INTEGER NOT NULL DEFAULT 0 )
CREATE INDEX idx_chats_archived ON chats(archived)
CREATE INDEX idx_citations_message_id ON citations(message_id)
CREATE INDEX idx_maps_collection ON maps(collection)
CREATE INDEX idx_messages_chat_id ON messages(chat_id)
CREATE INDEX idx_messages_query_id ON messages(query_id)
CREATE INDEX idx_messages_retriever ON messages(retriever)
CREATE TABLE maps ( id TEXT PRIMARY KEY, name TEXT NOT NULL, collection TEXT NOT NULL, snapshot TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP )
CREATE TABLE message_meta ( message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (message_id, key) )
CREATE TABLE messages ( id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), content TEXT NOT NULL, query_id TEXT, message_uuid TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, version INTEGER NOT NULL , retriever TEXT)
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN DELETE FROM chats_fts WHERE rowid = old.rowid; END
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN INSERT INTO chats_fts(rowid, chat_id, title, content) SELECT new.rowid, c.id, c.title, new.content FROM chats c WHERE c.id = new.chat_id; END
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN DELETE FROM chats_fts WHERE rowid = old.rowid; INSERT INTO chats_fts(rowid, chat_id, title, content) SELECT new.rowid, c.id, c.title, new.content FROM chats c WHERE c.id = new.chat_id; END