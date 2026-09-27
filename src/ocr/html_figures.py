"""HTML image preprocessing — download + OCR formula images in fetched pages.

Extracted from ``scripts.fetch_sources.py``.  The public entry point is
``preprocess_html_images``, called by ``src.fetch.html._fetch_and_extract``
during web-text extraction.
"""

from __future__ import annotations

import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests  # type: ignore[import-untyped]

from src.ocr.surya_formula import (
    _extract_svg_formula,
    _looks_like_formula_alt,
    _surya_batch_ocr,
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA}

# <img> tag extraction: src (required) + optional alt text
_IMG_RE = re.compile(
    r"""<img[^>]+src=["']([^"']+)["'](?:\s[^>]*?\salt=["']([^"']*)["'])?[^>]*>""",
    re.I,
)

# Min usable figure dimension.  Anything below this is icon/sprite/nav glyph
# noise (Arkham, Substack-style sites have hundreds per page).  Skipping
# them avoids downloading + OCR'ing meaningless thumbnails.  Math formula
# SVGs are exempt — they're often <200px tall.
_MIN_DIM_PX = 300


def preprocess_html_images(
    html: str,
    base_url: str,
    figures_dir: Path,
    delay_s: float,
    *,
    pace_fn: object = None,
) -> str:
    """Replace ``<img>`` tags with inline formula text or ``[Figure: …]`` markers.

    Two-pass: downloads images first, then batch-OCRs all non-SVG images
    via surya, then injects markers.  Recovery chain per image:
    1. MathJax/KaTeX SVG → embedded LaTeX from ``<title>``
    2. Alt-text heuristic
    3. Surya batch formula recognition (CPU, ~0.15s/image amortised)
    4. ``[Figure: …]`` fallback
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_host = urlparse(base_url).netloc

    # --- pass 1: download all images -----------------------------------------
    downloads: list[tuple[str, Path, str | None, str | None]] = []
    # (img_tag, out_file, svg_formula_or_None, alt_formula_or_None)
    ocr_queue: list[Path] = []

    for i, m in enumerate(_IMG_RE.finditer(html)):
        img_tag = m.group(0)
        src = m.group(1)
        alt = m.group(2) or ""
        if not src or src.startswith("data:"):
            continue
        parsed = urlparse(src)
        img_url = urljoin(base_url, src)
        host = urlparse(img_url).netloc
        try:
            # Pacing only matters across distinct hosts. The page already
            # returned all image refs in one HTML payload — there's no anti-
            # bot reason to space requests to the same host. Without this
            # short-circuit, a page with 300+ images takes 300+ seconds.
            if host != base_host:
                if pace_fn is not None:
                    pace_fn(host, delay_s)  # type: ignore[operator]
                else:
                    time.sleep(delay_s)
            r = requests.get(img_url, headers=HEADERS, timeout=10, stream=True,
                             allow_redirects=True)
            r.raise_for_status()
        except Exception:
            continue
        ext = None
        ct = r.headers.get("content-type", "")
        if ct:
            ext = mimetypes.guess_extension(ct.split(";")[0].strip())
        if not ext:
            ext = Path(parsed.path).suffix
        if not ext or ext == ".":
            ext = ".png"
        elif not ext.startswith("."):
            ext = f".{ext}"
        out_file = figures_dir / f"picture_{i}{ext}"
        try:
            with out_file.open("wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception:
            continue

        # Drop tiny icons/sprites. SVGs may carry math formulas (handled below)
        # so don't size-filter them. For raster images, peek dimensions cheaply
        # via Pillow; if it fails (corrupt file, unsupported format), keep.
        # Formula-likely rasters (wide+short equation strips) are exempt from
        # the dim filter so they survive even when below _MIN_DIM_PX.
        is_formula_raster = False
        if ext != ".svg":
            try:
                from PIL import Image  # noqa: PLC0415
                with Image.open(out_file) as im:
                    w, h = im.size
                is_formula_raster = h > 0 and (w / h) > 2.0 and h < 200
                if not is_formula_raster and (w < _MIN_DIM_PX or h < _MIN_DIM_PX):
                    out_file.unlink(missing_ok=True)
                    continue
            except Exception:
                pass

        svg_formula = None
        alt_formula = None
        if ext == ".svg":
            svg_formula = _extract_svg_formula(out_file)
        elif alt and _looks_like_formula_alt(alt):
            alt_formula = alt.strip()

        downloads.append((img_tag, out_file, svg_formula, alt_formula))
        # Surya OCR is ~0.15s/image; on blog corpora that means minutes of
        # wallclock for chart/photo noise. Only OCR rasters that look like
        # rendered equations (wide+short aspect). Alt-formula rasters are
        # already captured via alt_formula above and don't need OCR.
        if svg_formula is None and alt_formula is None and is_formula_raster:
            ocr_queue.append(out_file)

    # --- pass 2: batch OCR via surya -----------------------------------------
    ocr_results: dict[str, str | None] = {}
    if ocr_queue:
        ocr_results = _surya_batch_ocr(ocr_queue)

    # --- pass 3: inject markers ----------------------------------------------
    modified = html
    for img_tag, out_file, svg_formula, alt_formula in downloads:
        formula = svg_formula or alt_formula or ocr_results.get(str(out_file))
        if formula:
            marker = f"${formula}$"
        else:
            marker = f"[Figure: {out_file.name}]"
        modified = modified.replace(img_tag, marker, 1)

    return modified
