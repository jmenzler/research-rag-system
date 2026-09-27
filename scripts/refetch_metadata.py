"""Lightweight metadata-only refetch for sources with weak attribution.

Old `meta.json` files lack the `page_metadata` field (Fix 1 only populates
it on fresh fetches). For HTML sources whose authors fell back to a
single-word sitename, host-derived brand, or empty, an HTTP GET + a
trafilatura pass can often recover a real `og:article:author` /
JSON-LD `Person.name` signal.

This script does NOT re-run the full spider chain (MinerU, NotebookLM,
stealth tiers). It just pulls the raw HTML, runs `harvest_page_metadata`,
and writes the result into `meta["page_metadata"]`. Run the regular
`python -m src.fetch.metadata --all --force` afterwards to fold the new
signal into `meta["metadata"]`.

Modes:
  --list      Print candidate count + sample (no HTTP).
  --execute   Plain httpx GET. Fast but skips CF-walled hosts.
  --stealth   Scrapling StealthyFetcher with solve_cloudflare. Slow but
              recovers the hosts the plain fetch skipped (403 / CF / etc).
              Operates only on candidates that have no page_metadata yet
              (i.e. the --execute leftovers).

Politeness:
  - Per-host minimum 1s spacing (concurrent across hosts allowed).
  - 10s timeout per request in --execute, 30s in --stealth.
  - Skips non-200 responses silently in --execute.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetch.metadata import _platform_match  # noqa: E402
from src.fetch.page_metadata import harvest_page_metadata  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _is_weak_attribution(md: dict[str, Any]) -> bool:
    """True if the existing metadata block looks like a fallback, not a
    real author signal — i.e. would benefit from a trafilatura refetch."""
    authors = md.get("authors") or []
    if not authors:
        return True  # empty → any signal helps
    if len(authors) > 1:
        return False  # multi-author = probably real
    # Single-entry: weak if it's a sitename/brand (no space → single token,
    # or matches a generic short tail like 'Reddit', 'GitHub', etc.)
    only = authors[0]
    if not isinstance(only, str):
        return True
    if " " not in only:
        return True  # 'QuantStart', 'Mergify', 'GitHub'
    if len(only.split()) <= 2 and only.istitle():
        # 'Dean Markwick' is two-word title-case — could be real, could be
        # sitename. Refetch is cheap; let trafilatura confirm.
        return True
    return False


def _is_candidate(meta: dict[str, Any], doc_dir: Path) -> bool:
    if (doc_dir / "pdf.txt").exists():
        return False  # PDF source — text-scan handles it
    url = meta.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return False
    if _platform_match(url) is not None:
        return False  # Wikipedia/Reddit/HN/Stack* won't gain
    if meta.get("page_metadata"):
        return False  # already harvested
    md = meta.get("metadata") or {}
    if not isinstance(md, dict):
        return False
    return _is_weak_attribution(md)


def _walk_candidates(sources_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return list of (meta_path, parsed_meta) for refetch candidates."""
    out: list[tuple[Path, dict[str, Any]]] = []
    for mp in sources_root.rglob("meta.json"):
        if "_quarantine" in str(mp):
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_candidate(meta, mp.parent):
            out.append((mp, meta))
    return out


def _list_mode(sources_root: Path) -> int:
    candidates = _walk_candidates(sources_root)
    log.info("scanned %s — %d refetch candidates", sources_root, len(candidates))
    by_host: dict[str, int] = defaultdict(int)
    for _, meta in candidates:
        host = urlparse(meta.get("url", "")).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        by_host[host] += 1
    top_hosts = sorted(by_host.items(), key=lambda kv: -kv[1])[:25]
    print()
    print(f"=== {len(candidates)} candidates across {len(by_host)} hosts ===")
    print()
    print("Top 25 hosts:")
    for host, n in top_hosts:
        print(f"  {n:5d}  {host}")
    print()
    print("Sample 10:")
    for mp, meta in candidates[:10]:
        md = meta.get("metadata") or {}
        print(f"  [{mp.parent.name[:50]:50s}]")
        print(f"    url:    {meta.get('url','')[:75]}")
        print(f"    auth:   {md.get('authors')}")
    return 0


def _execute_mode(sources_root: Path, *, max_per_host_per_sec: float = 1.0) -> int:
    candidates = _walk_candidates(sources_root)
    log.info("refetching %d candidates", len(candidates))

    headers = {"User-Agent": "Mozilla/5.0 (rag-system metadata refetch)"}
    last_hit_at: dict[str, float] = {}
    min_spacing = 1.0 / max_per_host_per_sec

    ok = 0
    skipped = 0
    failed = 0
    written = 0

    with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
        for i, (mp, meta) in enumerate(candidates, 1):
            url = meta["url"]
            host = urlparse(url).netloc.lower()

            # Per-host rate limit
            last = last_hit_at.get(host, 0.0)
            wait = min_spacing - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)

            try:
                resp = client.get(url)
                last_hit_at[host] = time.monotonic()
            except Exception as exc:
                failed += 1
                if i % 50 == 0 or failed <= 5:
                    log.warning("[%d/%d] FAIL %s: %s",
                                i, len(candidates), host, type(exc).__name__)
                continue

            if resp.status_code != 200 or not resp.text:
                skipped += 1
                if skipped <= 5 or i % 100 == 0:
                    log.info("[%d/%d] skip %d %s",
                             i, len(candidates), resp.status_code, host)
                continue

            try:
                pm = harvest_page_metadata(resp.text)
            except Exception as exc:
                failed += 1
                log.warning("[%d/%d] harvest failed for %s: %s",
                            i, len(candidates), host, exc)
                continue

            if not pm:
                ok += 1
                continue

            meta["page_metadata"] = pm
            mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
            written += 1
            ok += 1
            if written <= 10 or i % 100 == 0:
                log.info("[%d/%d] wrote page_metadata for %s (authors=%r)",
                         i, len(candidates), mp.parent.name[:40],
                         pm.get("authors"))

    log.info("done. ok=%d written=%d skipped=%d failed=%d",
             ok, written, skipped, failed)
    return 0


def _stealth_mode(sources_root: Path, *, max_per_host_per_sec: float = 0.5) -> int:
    """Re-fetch candidates that the plain httpx pass couldn't handle.

    Uses Scrapling's StealthyFetcher (headless Chrome + CF solver). Much
    slower than the httpx path (~3-10s per request) so we restrict to
    candidates that still lack ``page_metadata`` after --execute ran.
    """
    from scrapling.fetchers import StealthyFetcher  # noqa: PLC0415

    candidates = _walk_candidates(sources_root)
    log.info("stealth refetching %d remaining candidates", len(candidates))

    last_hit_at: dict[str, float] = {}
    min_spacing = 1.0 / max_per_host_per_sec
    ok = 0
    written = 0
    failed = 0

    for i, (mp, meta) in enumerate(candidates, 1):
        url = meta["url"]
        host = urlparse(url).netloc.lower()

        last = last_hit_at.get(host, 0.0)
        wait = min_spacing - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)

        try:
            resp = StealthyFetcher.fetch(
                url,
                headless=True,
                solve_cloudflare=True,
                network_idle=True,
                timeout=30_000,
            )
            last_hit_at[host] = time.monotonic()
        except Exception as exc:
            failed += 1
            if failed <= 5 or i % 20 == 0:
                log.warning("[%d/%d] STEALTH FAIL %s: %s",
                            i, len(candidates), host, type(exc).__name__)
            continue

        html = getattr(resp, "body", None) or getattr(resp, "html_content", None)
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        if not html:
            failed += 1
            continue

        try:
            pm = harvest_page_metadata(html)
        except Exception as exc:
            failed += 1
            log.warning("[%d/%d] harvest failed for %s: %s",
                        i, len(candidates), host, exc)
            continue

        ok += 1
        if not pm:
            continue

        meta["page_metadata"] = pm
        mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
        written += 1
        log.info("[%d/%d] stealth wrote page_metadata for %s (authors=%r)",
                 i, len(candidates), mp.parent.name[:40], pm.get("authors"))

    log.info("stealth done. ok=%d written=%d failed=%d", ok, written, failed)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true",
                   help="Just count + sample candidates (no HTTP)")
    p.add_argument("--execute", action="store_true",
                   help="Plain httpx refetch loop (fast, skips CF)")
    p.add_argument("--stealth", action="store_true",
                   help="Stealth refetch loop (slow, handles CF)")
    p.add_argument("--sources-root", default=None,
                   help="Override sources/ root (default: <repo>/sources)")
    p.add_argument("--rate", type=float, default=1.0,
                   help="Max requests per host per second (default: 1.0)")
    args = p.parse_args()

    modes_set = sum([args.list, args.execute, args.stealth])
    if modes_set != 1:
        p.error("specify exactly one of --list / --execute / --stealth")

    if args.sources_root:
        root = Path(args.sources_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1] / "sources"

    if not root.exists():
        log.error("sources root not found: %s", root)
        return 1

    if args.list:
        return _list_mode(root)
    if args.execute:
        return _execute_mode(root, max_per_host_per_sec=args.rate)
    return _stealth_mode(root, max_per_host_per_sec=min(args.rate, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
