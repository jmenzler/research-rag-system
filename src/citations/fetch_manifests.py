"""Fetch S2 dataset manifests and prepare aria2c input files.

For each dataset name passed (default: papers, citations, paper-ids, abstracts),
hit ``/datasets/v1/release/latest/dataset/<name>`` and write:
  - ``<base>/<release>/_manifests/<dataset>.json``
  - ``<base>/<release>/_log/aria2_<dataset>.txt`` (aria2c -i input format)

Reads ``SEMANTIC_SCHOLAR_API_KEY`` from project ``.env`` via ``src.config``.

Run from project root:

    python -m src.citations.fetch_manifests --datasets papers,citations,paper-ids,abstracts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests  # type: ignore[import-untyped]

import src.config  # noqa: F401 — loads .env
from src.citations import DEFAULT_BASE, DEFAULT_RELEASE

_API_BASE = "https://api.semanticscholar.org/datasets/v1/release/latest/dataset"
_REQUEST_SPACING_S = 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=DEFAULT_RELEASE)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument(
        "--datasets",
        default="papers,citations,paper-ids,abstracts",
        help="comma-separated dataset names",
    )
    args = ap.parse_args()

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise SystemExit("SEMANTIC_SCHOLAR_API_KEY missing — populate .env")

    out_root = Path(args.base) / args.release
    manifests_dir = out_root / "_manifests"
    log_dir = out_root / "_log"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets.split(","):
        (out_root / ds).mkdir(parents=True, exist_ok=True)

    headers = {"x-api-key": api_key}
    for ds in args.datasets.split(","):
        url = f"{_API_BASE}/{ds}"
        print(f">> fetching {ds}")
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        payload = r.json()
        (manifests_dir / f"{ds}.json").write_text(json.dumps(payload))

        files = payload["files"]
        input_path = log_dir / f"aria2_{ds}.txt"
        with input_path.open("w") as out:
            for file_url in files:
                fname = file_url.split("?")[0].rsplit("/", 1)[1]
                out.write(file_url + "\n")
                out.write(f"  out={fname}\n")
                out.write(f"  dir={out_root}/{ds}/\n")
        print(f"   {ds}: {len(files)} files  ->  {input_path}")

        time.sleep(_REQUEST_SPACING_S)


if __name__ == "__main__":
    main()
