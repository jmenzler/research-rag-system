"""Parse source PDFs to text via MinerU.

Walks ``sources/<collection>/<slug>/source.pdf`` and writes ``pdf.txt`` next
to each one. Idempotent — skips PDFs that already have ``pdf.txt``.

Usage:
    # Parse every PDF in a collection:
    uv run python -m src.pdf_parsers.cli --collection trading

    # Specific PDFs:
    uv run python -m src.pdf_parsers.cli sources/trading/<slug>/source.pdf [...]

    # Glob (relative to repo root):
    uv run python -m src.pdf_parsers.cli --paths "sources/trading/*/source.pdf"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.pdf_parsers import mineru as mineru_parser
from src.pipeline.pipeline_log import StageLogger

REPO_ROOT = Path(__file__).resolve().parent.parent


def page_count(pdf: Path) -> int:
    import subprocess
    try:
        out = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True, check=False, timeout=10,
        )
        for line in out.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
    except Exception:
        pass
    return -1


def parse_one(
    pdf: Path, out_path: Path,
    no_vlm_fallback: bool = False,
    pass_mode: str = "auto",
    force: bool = False,
) -> tuple[bool, str, float]:
    """Returns (ok, status_msg, elapsed_seconds).

    pass_mode:
      - "auto": current behaviour — parse_pdf() (pipeline + VLM fallback inside one call)
      - "pipeline-only": parse_pipeline_only(); write pdf.txt if clean, write
        pdf.needs_vlm marker if broken fonts (no VLM). ok=True either way.
      - "vlm": only process PDFs with a pdf.needs_vlm marker; delete marker on success.
    """
    t0 = time.monotonic()
    _ = force  # used by main loop's skip-if-exists; accepted here for API consistency
    marker = pdf.parent / "pdf.needs_vlm"

    try:
        if pass_mode == "pipeline-only":
            text = mineru_parser.parse_pipeline_only(pdf)
            if mineru_parser.has_broken_fonts(text):
                marker.write_text("")
                return True, "deferred (broken fonts)", time.monotonic() - t0
            out_path.write_text(text)
            marker.unlink(missing_ok=True)
            return True, f"{len(text):,} chars", time.monotonic() - t0

        if pass_mode == "vlm":
            if not marker.exists():
                return True, "skip (no marker)", time.monotonic() - t0
            text = mineru_parser.parse_vlm_only(pdf)
            out_path.write_text(text)
            marker.unlink()
            return True, f"{len(text):,} chars", time.monotonic() - t0

        # pass_mode == "auto"
        text = (
            mineru_parser.parse_pipeline_only(pdf)
            if no_vlm_fallback
            else mineru_parser.parse_pdf(pdf)
        )
    except Exception as e:
        return False, f"FAIL {type(e).__name__}: {e}", time.monotonic() - t0

    out_path.write_text(text)
    return True, f"{len(text):,} chars", time.monotonic() - t0


def collect_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.collection:
        return sorted(
            (REPO_ROOT / "sources" / args.collection).glob("*/source.pdf")
        )
    if args.paths:
        return sorted(REPO_ROOT.glob(args.paths)) if not Path(args.paths).is_absolute() \
            else sorted(Path("/").glob(args.paths.lstrip("/")))
    if args.pdfs:
        return [p.resolve() for p in args.pdfs]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdfs", nargs="*", type=Path,
                    help="Specific PDF paths to parse.")
    ap.add_argument("--collection",
                    help="Collection root under sources/ to parse all PDFs in "
                         "(e.g. trading).")
    ap.add_argument("--paths", help="Glob pattern (relative to repo root or absolute).")
    ap.add_argument("--out-name", default="pdf.txt",
                    help="Sidecar filename to write (default: pdf.txt).")
    ap.add_argument("--force", action="store_true",
                    help="Re-parse even if the output file already exists.")
    ap.add_argument("--no-vlm-fallback", action="store_true",
                    help="Mineru only: skip VLM fallback, run pipeline-only "
                         "(faster; misses formulas on broken-font PDFs).")
    ap.add_argument("--pass", dest="pass_mode", default="auto",
                    choices=["auto", "pipeline-only", "vlm"],
                    help="Two-pass mode for orchestrated runs (default: auto). "
                         "pipeline-only writes pdf.needs_vlm markers for broken-font PDFs. "
                         "vlm processes only marked PDFs.")
    args = ap.parse_args()

    pdfs = collect_pdfs(args)
    if not pdfs:
        print("No PDFs found. Pass --collection, --paths, or positional PDF paths.",
              file=sys.stderr)
        sys.exit(2)

    import json
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    progress_path = log_dir / f"parse_pdfs_progress_{int(time.time())}.jsonl"
    print(f"Parsing {len(pdfs)} PDFs with parser=mineru, "
          f"pass={args.pass_mode}, out_name={args.out_name}")
    print(f"Progress log: {progress_path}")
    skipped = ok = failed = 0
    total_elapsed = 0.0
    run_t0 = time.monotonic()

    def fmt_eta(remaining: int, avg_per: float) -> str:
        if avg_per <= 0 or remaining <= 0:
            return "?"
        secs = remaining * avg_per
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    stage_label = f"parse_{args.pass_mode}" if args.pass_mode != "auto" else "parse"
    sl = StageLogger(args.collection or "ad_hoc", stage_label)
    with progress_path.open("w") as plog, sl:
        for i, pdf in enumerate(pdfs, 1):
            out_path = pdf.parent / args.out_name
            wall_so_far = time.monotonic() - run_t0
            done = i - 1
            avg_per = (total_elapsed / done) if done else 0.0
            eta = fmt_eta(len(pdfs) - done, avg_per)
            if out_path.exists() and not args.force:
                skipped += 1
                print(f"[{i:>3}/{len(pdfs)}] SKIP (exists)  {pdf.parent.name}  "
                      f"[wall={wall_so_far/60:.1f}m eta≈{eta}]", flush=True)
                plog.write(json.dumps({"i": i, "n": len(pdfs), "name": pdf.parent.name,
                                       "status": "skip"}) + "\n")
                plog.flush()
                sl.item_skip(pdf.parent.name, reason="exists")
                continue
            pages = page_count(pdf)
            size_kb = pdf.stat().st_size / 1024
            print(f"[{i:>3}/{len(pdfs)}] {pdf.parent.name}  ({pages}p, {size_kb:.0f}K)  "
                  f"[wall={wall_so_far/60:.1f}m avg={avg_per:.0f}s/doc eta≈{eta}]",
                  flush=True)
            success, status, elapsed = parse_one(
                pdf, out_path,
                no_vlm_fallback=args.no_vlm_fallback,
                pass_mode=args.pass_mode,
                force=args.force,
            )
            total_elapsed += elapsed
            if success:
                ok += 1
                rate = pages / elapsed if pages > 0 and elapsed > 0 else 0
                print(f"          ok  {elapsed:.1f}s  {status}  ({rate:.2f}p/s)", flush=True)
                plog.write(json.dumps({
                    "i": i, "n": len(pdfs), "name": pdf.parent.name,
                    "status": "ok", "pages": pages, "elapsed_s": round(elapsed, 1),
                    "chars": status,
                }) + "\n")
                sl.item_ok(pdf.parent.name, elapsed_s=elapsed,
                           pages=pages, chars=status)
            else:
                failed += 1
                print(f"          {status}", flush=True)
                plog.write(json.dumps({
                    "i": i, "n": len(pdfs), "name": pdf.parent.name,
                    "status": "fail", "error": status,
                }) + "\n")
                sl.item_fail(pdf.parent.name, elapsed_s=elapsed, reason=status)
            plog.flush()

    print(f"\nDone — ok={ok}  skipped={skipped}  failed={failed}  "
          f"wall={total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
