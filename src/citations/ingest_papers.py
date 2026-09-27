"""Filter S2 papers dump into local DuckDB.

Reads every `.gz` shard under `<base>/<release>/papers/`, keeps rows whose
``s2fieldsofstudy.category`` intersects :data:`TARGET_FIELDS`, and writes them
to ``citations.duckdb`` along with a ``corpus_in_scope`` index table.

Run from project root after the papers snapshot has finished downloading:

    python -m src.citations.ingest_papers --release 2026-05-12
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import cast

import duckdb

from src.citations import DEFAULT_BASE, DEFAULT_RELEASE, TARGET_FIELDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--db-path", default=None)
    args = ap.parse_args()

    base = f"{args.base}/{args.release}"
    papers_dir = f"{base}/papers"
    db_path: str = args.db_path or f"{base}/citations.duckdb"

    gz = sorted(glob.glob(f"{papers_dir}/*.gz"))
    incomplete = [f for f in gz if os.path.exists(f + ".aria2")]
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} aria2c download still in progress in {papers_dir}"
        )
    if not gz:
        raise SystemExit(f"no .gz files in {papers_dir}")
    print(f"papers/*.gz count: {len(gz)}")

    tgt_sql = "(" + ",".join(f"'{t}'" for t in TARGET_FIELDS) + ")"
    fos_filter = (
        "list_aggregate("
        f"list_transform(s2fieldsofstudy, x -> CASE WHEN x.category IN {tgt_sql} "
        "THEN 1 ELSE 0 END), 'sum') > 0"
    )
    file_list_sql = "[" + ",".join(f"'{f}'" for f in gz) + "]"

    con = duckdb.connect(db_path)
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET memory_limit = '20GB';")
    con.execute("SET threads = 12;")

    t0 = time.time()
    print("\n=== creating filtered papers table ===")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE papers AS
        SELECT
            corpusid,
            externalids,
            title,
            year,
            venue,
            authors,
            referencecount,
            citationcount,
            influentialcitationcount,
            s2fieldsofstudy
        FROM read_json_auto({file_list_sql},
            format='newline_delimited',
            maximum_object_size=33554432)
        WHERE {fos_filter}
        """
    )
    n_papers = cast(tuple[int], con.execute("SELECT count(*) FROM papers").fetchone())[0]
    print(f"  filtered papers: {n_papers:,}  ({time.time() - t0:.1f}s)")

    print("\n=== building corpus_in_scope ===")
    con.execute(
        """
        CREATE OR REPLACE TABLE corpus_in_scope AS
        SELECT corpusid FROM papers;
        """
    )
    con.execute("CREATE UNIQUE INDEX idx_scope_pk ON corpus_in_scope(corpusid);")
    con.execute("CREATE UNIQUE INDEX idx_papers_pk ON papers(corpusid);")
    con.execute("CREATE INDEX idx_papers_arxiv ON papers((externalids.ArXiv));")
    con.execute("CREATE INDEX idx_papers_doi ON papers((externalids.DOI));")

    db_size = os.path.getsize(db_path)
    print("\n=== stats ===")
    print(f"  rows: {n_papers:,}")
    print(f"  db size: {db_size / 1024**3:.2f} GB")
    print(f"  total elapsed: {time.time() - t0:.1f}s")

    print("\n=== top 5 by citationcount ===")
    rows = con.execute(
        """
        SELECT corpusid, title, citationcount, year
        FROM papers ORDER BY citationcount DESC LIMIT 5
        """
    ).fetchall()
    for r in rows:
        print(f"  {r[0]:>10}  cit={r[2]:>7} y={r[3]}  {r[1][:70]}")


if __name__ == "__main__":
    main()
