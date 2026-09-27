-- 0001_schema_version.sql — initialise WAL mode + version marker.
-- Phase 0: no real tables yet. Phase 1 will add 0002_chats.sql with the
-- chats / messages / turns tables.
PRAGMA journal_mode = WAL;
