-- 0004_corpus_fts.sql — Phase 3 corpus_fts for the corpus-browser FTS5 search.
-- Idempotent: every CREATE uses IF NOT EXISTS / INSERT OR IGNORE.
--
-- This is a regular (not contentless) FTS5 table: it has an auto-generated content
-- table so SELECT returns column values (not just rowid). The sync is driven by
-- rebuild_corpus_fts() in papers_view.py, which reads the watermark from
-- corpus_fts_meta and re-INSERTs the entire corpus after the parents.sqlite
-- sources table changes.
--
-- Cross-DB triggers are impossible (parents.sqlite != chats.db), so the
-- watermark-driven rebuild is the sole sync mechanism. No triggers are used.

-- 1. FTS5 table — paper_id is UNINDEXED (never matched), all other columns
--    are searchable. Regular table (no content=''), so SELECT returns values.
CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
    paper_id UNINDEXED,
    title,
    authors,
    year,
    short_cite,
    arxiv_id,
    doi,
    tokenize='porter unicode61'
);

-- 2. Watermark tracker — single-row table that stores the max(sources.updated_at)
--    value at which the FTS was last rebuilt. Rebuild compares current
--    max(sources.updated_at) against this value; if current > watermark, rebuild.
CREATE TABLE IF NOT EXISTS corpus_fts_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    watermark INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO corpus_fts_meta (id, watermark) VALUES (1, 0);
