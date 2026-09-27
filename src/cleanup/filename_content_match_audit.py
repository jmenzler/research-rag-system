"""Filename↔content topic-match audit. Thin wrapper around `src.cleanup` primitives.

For every staged source file, runs two checks:
1. Fail-page detection (literal tokens + max-paragraph floor) → strong evidence
2. Filename-token-vs-content overlap → ratio low = filename promises X but
   content delivers Y, classified as poisoned only when combined (Rule 6) in
   the manifest builder.

Outputs `logs/filename_content_match.json` consumed by build_cleanup_manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanup.filename_match import descriptive_tokens, filename_match_ratio
from src.cleanup.quality import detect_fail_signature
from src.ingest import _strip_tail_sections

ROOT = Path(__file__).resolve().parent.parent.parent

# `sources/<collection>/<slug>/<leaf>.txt` — pdf.txt | nlm.txt | web.txt.
SOURCE_TEXT_GLOB = "sources/*/*/*.txt"
RATIO_THRESHOLD = 0.40
MIN_DESCRIPTIVE_TOKENS = 4  # below this, filename signal is unreliable (URL/arxiv-only)


def main() -> None:
    files = sorted(
        f for f in ROOT.glob(SOURCE_TEXT_GLOB)
        if not f.name.endswith(".ctx.json")
        and not any(p.startswith("_") for p in f.relative_to(ROOT / "sources").parts)
    )
    fail_page: list[tuple[str, str]] = []
    name_mismatch: list[tuple[float, int, str, list[str], list[str]]] = []
    skipped_id_only: list[str] = []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        clean, _ = _strip_tail_sections(text)
        # Pass 1: fail-page (literal sigs + max-paragraph floor)
        sig = detect_fail_signature(clean)
        if sig:
            fail_page.append((str(f.relative_to(ROOT)), sig))
            continue
        # Pass 2: filename-vs-content match
        toks = descriptive_tokens(str(f))
        if len(toks) < MIN_DESCRIPTIVE_TOKENS:
            skipped_id_only.append(str(f.relative_to(ROOT)))
            continue
        ratio = filename_match_ratio(str(f), clean)
        misses = [t for t in toks if t not in clean.lower()]
        if ratio < RATIO_THRESHOLD:
            name_mismatch.append((ratio, len(toks), str(f.relative_to(ROOT)), toks, misses))

    name_mismatch.sort()

    print(f"scanned: {len(files)} files")
    print(
        f"FAIL-PAGE detections: {len(fail_page)} "
        "(Cloudflare, 403, captcha, max-paragraph<70...)"
    )
    print(
        f"NAME-MISMATCH detections "
        f"(ratio < {RATIO_THRESHOLD}, "
        f"≥{MIN_DESCRIPTIVE_TOKENS} tokens): "
        f"{len(name_mismatch)}"
    )
    print(
        f"skipped (filename has <{MIN_DESCRIPTIVE_TOKENS} descriptive tokens "
        f"— arxiv/URL): {len(skipped_id_only)}"
    )
    print()
    print("=== FAIL PAGES ===")
    for path, sig in fail_page[:50]:
        print(f"  [{sig}]  {path}")
    print()
    print("=== NAME MISMATCHES (worst first) ===")
    for ratio, n_tok, path, toks, misses in name_mismatch[:50]:
        print(f"{ratio:.2f} ({n_tok:>2d} tok) {path}")
        print(f"           misses: {misses[:8]}")

    out = ROOT / "logs" / "filename_content_match.json"
    out.write_text(json.dumps({
        "fail_pages": [{"path": p, "reason": r} for p, r in fail_page],
        "name_mismatches": [
            {"ratio": r, "n_tokens": n, "path": p, "tokens": t, "misses": m}
            for r, n, p, t, m in name_mismatch
        ],
        "skipped_id_only": skipped_id_only,
    }, indent=2))
    print(f"\nfull report -> {out}")

    from src.pipeline.pipeline_log import StageLogger  # noqa: PLC0415
    with StageLogger("_corpus_", "audit") as sl:
        sl.item_ok("filename_match",
                   n_files=len(files),
                   fail_pages=len(fail_page),
                   name_mismatches=len(name_mismatch),
                   skipped_id_only=len(skipped_id_only))


if __name__ == "__main__":
    main()
