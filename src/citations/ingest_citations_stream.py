"""Stream-download + filtered ingest of S2 citations dataset.

For each citations shard listed in the cached manifest:
  1. aria2c download
  2. ``INSERT INTO citations SELECT ... WHERE both ends IN corpus_in_scope``
  3. delete raw ``.gz`` (unless ``--keep-raw``)

State persists in ``_log/citations_state.json`` so the job is restartable.

Run from project root after :mod:`src.citations.ingest_papers` finishes:

    nohup python -m src.citations.ingest_citations_stream \\
        > ./data/snapshots/<release>/_log/citations_stream.stdout 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import cast

import duckdb

from src.citations import DEFAULT_BASE, DEFAULT_RELEASE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--limit", type=int, default=0, help="only process N files (0 = all)")
    ap.add_argument("--keep-raw", action="store_true", help="don't delete .gz after ingest")
    args = ap.parse_args()

    base = Path(args.base) / args.release
    citations_dir = base / "citations"
    citations_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(base / "citations.duckdb")
    state_path = base / "_log" / "citations_state.json"
    log_path = base / "_log" / "citations_stream.log"

    manifest = json.loads((base / "_manifests" / "citations.json").read_text())
    all_files: list[str] = manifest["files"]
    if args.limit:
        all_files = all_files[: args.limit]

    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"done": [], "failed": []}
    done_set = set(state["done"])

    con = duckdb.connect(db_path)
    con.execute("SET memory_limit = '20GB';")
    con.execute("SET threads = 12;")

    scope_row = con.execute(
        """
        SELECT count(*) FROM information_schema.tables
        WHERE table_name = 'corpus_in_scope'
        """
    ).fetchone()
    has_scope = cast(tuple[int], scope_row)[0]
    if not has_scope:
        raise SystemExit("corpus_in_scope table missing — run ingest_papers first")

    scope_n = cast(tuple[int], con.execute("SELECT count(*) FROM corpus_in_scope").fetchone())[0]
    print(f"corpus_in_scope rows: {scope_n:,}")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS citations (
            citingcorpusid BIGINT,
            citedcorpusid  BIGINT,
            isinfluential  BOOLEAN
        );
        """
    )

    log_fh = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        log_fh.write(line + "\n")

    log(f"begin: {len(all_files)} files, {len(done_set)} already done")
    t_start = time.time()

    for i, url in enumerate(all_files):
        fname = url.split("?")[0].rsplit("/", 1)[1]
        if fname in done_set:
            continue

        local = citations_dir / fname
        t0 = time.time()

        if not local.exists():
            log(f"[{i + 1}/{len(all_files)}] download {fname}")
            r = subprocess.run(
                [
                    "aria2c", "-x", "4", "-s", "4",
                    "--continue=true", "--max-tries=10", "--retry-wait=10",
                    "--console-log-level=warn", "--summary-interval=0",
                    "-d", str(citations_dir), "-o", fname, url,
                ],
                capture_output=True,
            )
            if r.returncode != 0:
                log(f"  FAIL download: {r.stderr.decode()[-500:]}")
                state["failed"].append(fname)
                state_path.write_text(json.dumps(state, indent=2))
                continue
        else:
            log(f"[{i + 1}/{len(all_files)}] reuse local {fname}")

        size_mb = local.stat().st_size / 1024**2
        t_dl = time.time() - t0

        t1 = time.time()
        try:
            con.execute(
                f"""
                INSERT INTO citations
                SELECT citingcorpusid, citedcorpusid, isinfluential
                FROM read_json_auto('{local}',
                    format='newline_delimited',
                    maximum_object_size=33554432)
                WHERE citingcorpusid IN (SELECT corpusid FROM corpus_in_scope)
                  AND citedcorpusid  IN (SELECT corpusid FROM corpus_in_scope)
                """
            )
        except Exception as e:
            log(f"  FAIL ingest: {e}")
            state["failed"].append(fname)
            state_path.write_text(json.dumps(state, indent=2))
            continue
        t_ing = time.time() - t1

        if not args.keep_raw:
            local.unlink()

        state["done"].append(fname)
        state_path.write_text(json.dumps(state, indent=2))

        elapsed = time.time() - t_start
        done_n = len(state["done"])
        log(
            f"  size={size_mb:.0f}MB  dl={t_dl:.1f}s  ingest={t_ing:.1f}s  "
            f"total_elapsed={elapsed / 60:.1f}min"
        )

        if done_n % 5 == 0:
            tot_cit = cast(tuple[int], con.execute("SELECT count(*) FROM citations").fetchone())[0]
            db_gb = os.path.getsize(db_path) / 1024**3
            rate = done_n / max(elapsed, 1)
            remaining_s = (len(all_files) - done_n) / max(rate, 0.001)
            log(
                f"  cumulative: rows={tot_cit:,} db={db_gb:.1f}GB  "
                f"ETA={remaining_s / 60:.0f}min"
            )

    log_fh.close()


if __name__ == "__main__":
    main()
