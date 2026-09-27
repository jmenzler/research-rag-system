"""RAG-corpus membership for citation candidates.

The citation graph is the full S2 bulk dump — every walked node exists in it.
What callers actually want from ``in_corpus`` is whether a paper is already in
the *RAG retrieval corpus* (``parents.sqlite``), so a discovery walk can flag
genuinely new work. That answer is not in the graph store; it is derived here by
mapping each candidate's corpus_id → arxiv (``paper_meta.sqlite``) and testing
membership against the set of ingested arxiv ids.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

# Match both the canonical ``1234.56789`` form and the underscore-munged
# ``1234_56789`` that appears in ingested source-dir / file names.
_ARXIV_RE = re.compile(r"(\d{4})[._](\d{4,6})")


def normalize_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    s = re.sub(r"v\d+$", "", str(value).strip().lower().strip("/"))
    m = _ARXIV_RE.search(s)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def load_ingested_arxiv(corpus_db: str) -> set[str]:
    """Set of normalized arxiv ids present in the RAG corpus.

    Sources carry their arxiv id inconsistently (structured ``arxiv_id``/``url``
    columns, the ``id`` column for hash-ingested PDFs, the chunk ``source_file``
    path, or only the on-disk ``sources/<notebook>/<dir>`` name), so all of these
    are scanned and unioned.
    """
    ingested: set[str] = set()
    con = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    try:
        for col in ("id", "arxiv_id", "url"):
            try:
                for (value,) in con.execute(f"SELECT {col} FROM sources"):
                    n = normalize_arxiv(value)
                    if n:
                        ingested.add(n)
            except sqlite3.OperationalError:
                pass
        try:
            for (source_file,) in con.execute("SELECT DISTINCT source_file FROM parents"):
                n = normalize_arxiv(source_file)
                if n:
                    ingested.add(n)
        except sqlite3.OperationalError:
            pass
    finally:
        con.close()
    try:
        repo_root = Path(corpus_db).resolve().parent.parent
        for path in (repo_root / "sources").glob("*/*"):
            n = normalize_arxiv(path.name)
            if n:
                ingested.add(n)
    except OSError:
        pass
    return ingested


def membership(
    meta_con: sqlite3.Connection, corpus_ids: list[int], ingested: set[str]
) -> dict[int, bool | None]:
    """Map corpus_id → in-RAG-corpus.

    ``True`` ingested, ``False`` has an arxiv id but not ingested, ``None`` no
    resolvable arxiv (membership undetermined — books, journal-only papers).
    """
    out: dict[int, bool | None] = {}
    ids = [int(c) for c in corpus_ids]
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        q = f"SELECT corpusid, arxiv_id FROM paper_meta WHERE corpusid IN ({placeholders})"
        for cid, arxiv_id in meta_con.execute(q, chunk):
            n = normalize_arxiv(arxiv_id)
            out[int(cid)] = (n in ingested) if n else None
    for cid in ids:
        out.setdefault(cid, None)
    return out
