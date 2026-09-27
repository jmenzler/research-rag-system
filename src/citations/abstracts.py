"""Abstract lookup against ``abstracts.sqlite`` (corpusid PK, zstd-blob).

Only ~21% of kept papers carry an abstract (the dump ships the OA-redistributable
subset; niche/older papers skew missing), so :func:`fetch` returns a partial map;
callers back-fill gaps via :mod:`src.citations.s2_client`.
"""

from __future__ import annotations

import sqlite3
import sys

if sys.version_info >= (3, 14):
    from compression import zstd  # type: ignore[import-not-found]
else:
    from backports import zstd

# sqlite caps host params per statement (SQLITE_MAX_VARIABLE_NUMBER, 999 on the
# stock build), so the IN-clause is chunked well under that.
_CHUNK = 900


def fetch(con: sqlite3.Connection, corpus_ids: list[int]) -> dict[int, str]:
    """Return ``{corpusid: abstract}`` for the ids that have one stored.

    Missing ids (no abstract in the dump) are simply absent from the result.
    """
    ids = list({int(c) for c in corpus_ids})
    out: dict[int, str] = {}
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start : start + _CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = con.execute(
            f"SELECT corpusid, abstract_zstd FROM abstracts WHERE corpusid IN ({placeholders})",
            chunk,
        ).fetchall()
        for cid, blob in rows:
            if blob is not None:
                out[int(cid)] = zstd.decompress(blob).decode("utf-8")
    return out
