"""Surya-based formula recognition for image preprocessing.

Extracted from ``scripts/fetch_sources.py`` — private helpers used by
``src.ocr.html_figures.preprocess_html_images``.
"""

from __future__ import annotations

import re
from pathlib import Path

# --- Module-level surya predictor cache ---

_surya_predictor: object | None = None


def _get_surya_predictor() -> object:
    """Lazy-init the surya formula-recognition batch predictor (module-level cache)."""
    global _surya_predictor
    if _surya_predictor is None:
        from surya.foundation import (  # type: ignore[import-not-found]
            FoundationPredictor,  # noqa: PLC0415
        )
        from surya.recognition import (  # type: ignore[import-not-found]
            RecognitionPredictor,  # noqa: PLC0415
        )
        _surya_predictor = RecognitionPredictor(FoundationPredictor(device="cpu"))
    return _surya_predictor


# --- SVG formula extraction ---


def _extract_svg_formula(filepath: Path) -> str | None:
    """If an SVG contains a MathJax/KaTeX formula, return the LaTeX source text.

    Wikipedia and other MathJax sites embed the original LaTeX in the SVG
    ``<title>`` element::

        <title>{\\displaystyle \\sigma_t^2 = \\alpha_0 + ...}</title>

    KaTeX embeds it in an ``<annotation>`` element.  For Medium/Investopedia
    formula-images (server-side PNGs) this returns *None* — the text is lost.
    """
    try:
        text = filepath.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    m = re.search(
        r"<title[^>]*>\s*\{\\displaystyle\s+(.*?)\}\s*</title>", text, re.S
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r'<annotation[^>]*encoding="application/x-tex"[^>]*>(.*?)</annotation>',
        text, re.S,
    )
    if m:
        return m.group(1).strip()
    return None


# --- SVG rasteriser (cairosvg) ---


def _rasterize_svg(svg_path: Path) -> Path | None:
    """Convert *svg_path* to a temporary PNG via cairosvg (needs libcairo).

    Monkey-patches ``ctypes.util.find_library`` so cairocffi can discover the
    Homebrew-installed Cairo on macOS.
    """
    import ctypes  # noqa: PLC0415
    import ctypes.util  # noqa: PLC0415
    _orig_find = ctypes.util.find_library
    def _patched_find(name: str) -> str | None:
        result = _orig_find(name)
        if result is not None:
            return result
        for prefix in ("/opt/homebrew/lib", "/usr/local/lib"):
            lib = Path(prefix) / f"lib{name}.dylib"
            if lib.exists():
                return str(lib)
            lib = Path(prefix) / f"lib{name}.2.dylib"
            if lib.exists():
                return str(lib)
        return None
    ctypes.util.find_library = _patched_find
    try:
        import cairosvg  # type: ignore[import-not-found]  # noqa: PLC0415
    except Exception:
        return None
    png = svg_path.with_suffix(".png")
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(png))
    except Exception:
        return None
    return png if png.exists() else None


# --- Batch OCR ---


def _surya_batch_ocr(image_paths: list[Path]) -> dict[str, str | None]:
    """Run surya formula recognition on a batch of images.

    Returns a dict mapping ``str(path)`` → LaTeX text (or *None* if
    no formula was recognised).  One model-load amortised across the batch.
    """
    from PIL import Image  # noqa: PLC0415
    try:
        from surya.common.surya.schema import (  # type: ignore[import-not-found]
            TaskNames,  # noqa: PLC0415
        )
    except Exception:
        return {}
    predictor = _get_surya_predictor()
    images: list[object] = []
    valid_indices: list[int] = []
    for idx, p in enumerate(image_paths):
        try:
            img = Image.open(str(p))
            if img.mode == "P":
                img = img.convert("RGBA")  # type: ignore[assignment]
            images.append(img)
            valid_indices.append(idx)
        except Exception:
            continue
    if not images:
        return {}
    tasks = [TaskNames.block_without_boxes] * len(images)
    bboxes = [[[0, 0, img.width, img.height]] for img in images]  # type: ignore[attr-defined]
    try:
        predictions = predictor(images, tasks, bboxes=bboxes)  # type: ignore[operator]
    except Exception:
        return {}
    results: dict[str, str | None] = {}
    for vi, pred in zip(valid_indices, predictions):
        key = str(image_paths[vi])
        raw = pred.text_lines[0].text
        raw = raw.replace("<math>", "").replace("</math>", "")
        raw = raw.replace('<math display="block">', "").strip()
        results[key] = raw if raw else None
    return results


# --- Alt-text formula heuristic ---


def _looks_like_formula_alt(alt: str) -> bool:
    """Heuristic: does ``alt`` text look like a formula rather than a caption?"""
    if not alt:
        return False
    # LaTeX math delimiters or common formula markers
    if re.search(
        r"[\\$][a-zA-Z]|\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|"
        r"\\sigma|\\epsilon|\\theta|\\lambda|\\mu|\\pi|\\rho|\\omega|"
        r"\\gamma|\\delta", alt,
    ):
        return True
    # Math operators / relation symbols
    if re.search(
        r"[=<>+\-*/^_{}|]|\\times|\\cdot|\\leq|\\geq|\\neq|\\approx", alt,
    ):
        return True
    return len(alt) > 5 and re.search(r"[^a-zA-Z\s]{3,}", alt) is not None
