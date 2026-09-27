"""HTML pre-processing for trafilatura: math, headings, images.

Promoted from ``spikes/scrapling_spider_pipeline.py``.  Pure regex-based
transformations with zero Scrapling dependency — modify HTML before
trafilatura sees it so math, heading hierarchy and content images survive
extraction.
"""

from __future__ import annotations

import html as _html_mod
import re
from urllib.parse import urljoin

# ---------------------------------------------------------------------------
# Math: <math> blocks + Wikipedia fallback images → inline LaTeX
# ---------------------------------------------------------------------------

_MATHML_ANNOTATION_RE = re.compile(
    r"<annotation\s+encoding=[\"']application/x-tex[\"'][^>]*>(.*?)</annotation>",
    re.I | re.S,
)
_MATH_TAG_RE = re.compile(r"<math[\s>][\s\S]*?</math>", re.I)

_WP_MATH_FALLBACK_RE = re.compile(
    r"""<img\s+[^>]*\bclass=["'][^"']*mwe-math-fallback[^"']*["']"""
    r"""[^>]*\salt=["']\{?\\displaystyle\s+(.*?)\}?["'][^>]*>""",
    re.I,
)


def inline_math(html: str) -> str:
    """Replace ``<math>`` blocks and Wikipedia math-fallback ``<img>`` with ``$…$`` / ``$$…$$``.

    Handles three patterns:
    1. arXiv: ``<annotation encoding="application/x-tex">`` child element
    2. Wikipedia: ``<math alttext="{\\displaystyle ...}">`` attribute
    3. Wikipedia: ``<img class="mwe-math-fallback-..." alt="...">``
    """

    def _replace_math_tag(m: re.Match[str]) -> str:
        block = m.group(0)
        if 'style="display: none"' in block or "style='display: none'" in block:
            return " "
        # Pattern 1: arXiv annotation
        anno = _MATHML_ANNOTATION_RE.search(block)
        if anno:
            latex = anno.group(1).strip()
            latex = latex.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            is_display = any(
                kw in latex
                for kw in (r"\\", r"\begin", r"\frac", r"\sum", r"\int",
                           r"\prod", r"\lim", r"\big", r"\left")
            )
            return f" {'$$' if is_display else '$'}{latex}{'$$' if is_display else '$'} "
        # Pattern 2: Wikipedia alttext attribute
        alt_m = re.search(
            r"""\salttext=["']\{?\\displaystyle\s+(.*?)\}?["']""", block, re.I,
        )
        if alt_m:
            latex = alt_m.group(1).strip()
            latex = latex.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            is_display = any(
                kw in latex
                for kw in (r"\\", r"\begin", r"\frac", r"\sum", r"\int",
                           r"\prod", r"\lim", r"\big", r"\left")
            )
            return f" {'$$' if is_display else '$'}{latex}{'$$' if is_display else '$'} "
        # Pattern 3: fallback — strip tags, keep text content
        inner = re.sub(r"<[^>]+>", " ", block)
        inner = re.sub(r"\s+", " ", inner).strip()
        return f" $$ {inner} $$ " if inner else " "

    html = _MATH_TAG_RE.sub(_replace_math_tag, html)

    def _replace_wp_fallback(m: re.Match[str]) -> str:
        latex = m.group(1).strip()
        latex = latex.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        is_display = any(
            kw in latex
            for kw in (r"\\", r"\begin", r"\frac", r"\sum", r"\int",
                       r"\prod", r"\lim", r"\big", r"\left")
        )
        delim = "$$" if is_display else "$"
        return " " + delim + latex + delim + " "

    return _WP_MATH_FALLBACK_RE.sub(_replace_wp_fallback, html)


# ---------------------------------------------------------------------------
# Headings: prepend markdown # prefixes so hierarchy survives trafilatura
# ---------------------------------------------------------------------------

_HEADING_CONTENT_RE = re.compile(
    r"(<\s*h[1-6][^>]*>)\s*(.*?)\s*(<\s*/\s*h[1-6]\s*>)", re.I | re.S,
)


def headings_to_markdown(html: str) -> str:
    """Prepend ``#`` / ``##`` / … to heading text content in-place."""

    def _repl(m: re.Match[str]) -> str:
        open_tag, body, close_tag = m.groups()
        level_m = re.search(r"h([1-6])", open_tag, re.I)
        level = int(level_m.group(1)) if level_m else 1
        level = min(level, 4)
        prefix = "#" * level
        return f"{open_tag}{prefix} {body}{close_tag}"

    return _HEADING_CONTENT_RE.sub(_repl, html)


# ---------------------------------------------------------------------------
# Images: collect <img> into registry, replace orphans with markers
# ---------------------------------------------------------------------------

_IMG_SRC_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["'][^>]*>""", re.I)
_MIN_IMG_DIM = 200

_SPAN_IMG_RE = re.compile(
    r"""<span\s+class=["']mw-default-size["'][^>]*>\s*<a\s+[^>]*>\s*<img\s+[^>]*>\s*</a>\s*</span>""",
    re.I,
)
_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def collect_images(html: str, base_url: str) -> dict[int, dict[str, str]]:
    """Scan HTML for ``<img>`` tags and build an image registry.

    Skips math-fallback images, ``data:`` URIs, UI chrome / icons / logos.
    Captions are taken from nearby ``<figcaption>`` or
    ``<div class=\"thumbcaption\">`` elements.

    Returns ``{n: {"url": str, "label": str}}`` indexed by image number.
    """
    registry: dict[int, dict[str, str]] = {}

    for m in _IMG_SRC_RE.finditer(html):
        tag = m.group(0)
        src_m = re.search(r"""src=["']([^"']+)["']""", tag, re.I)
        if not src_m:
            continue
        src_url = urljoin(base_url, _html_mod.unescape(src_m.group(1)))
        if src_url.startswith("data:"):
            continue
        if re.search(r"\bmwe-math-fallback\b", tag, re.I):
            continue
        if re.search(r"/(?:static|skins|resources|assets)/", src_url):
            continue
        if re.search(r"/(?:logo|icon|button|badge|pixel)(?:[.-]|$)", src_url, re.I):
            continue

        # Dimension filtering
        w = 0
        px_m = re.search(r"/(\d+)px-", src_url)
        url_w = int(px_m.group(1)) if px_m else 0
        if url_w:
            w = url_w
        else:
            w_m = re.search(r"""width=["']?(\d+)""", tag, re.I)
            w = int(w_m.group(1)) if w_m else 0
        if w and w < _MIN_IMG_DIM:
            continue

        alt_m = re.search(r"""alt=["']([^"']*)["']""", tag, re.I)
        alt = (alt_m.group(1) or "").strip() if alt_m else ""

        # Caption from nearby <figcaption> or <div class="thumbcaption">
        ctx_start = max(0, m.start() - 600)
        ctx_end = min(len(html), m.end() + 600)
        ctx = html[ctx_start:ctx_end]
        cap_m = re.search(
            r"""<(?:figcaption|div\s+class=["'][^"']*thumbcaption[^"']*["'])\b[^>]*>"""
            r"""(.*?)</(?:figcaption|div)>""",
            ctx, re.I | re.S,
        )
        caption = ""
        if cap_m:
            caption = re.sub(r"<[^>]+>", " ", cap_m.group(1))
            caption = re.sub(r"\s+", " ", caption).strip().rstrip(".")

        n = len(registry)
        registry[n] = {"url": src_url}
        label = caption or alt or f"Image {n}"
        registry[n]["label"] = label

    return registry


def replace_span_imgs(html: str, base_url: str, registry: dict[int, dict[str, str]]) -> str:
    """Replace span-wrapped orphan images with ``__IMG_N__`` markers.

    trafilatura drops ``<span>`` content, so convert these to text markers
    that survive extraction and are swapped back in ``postprocess_markdown``.
    """

    def _repl(m: re.Match[str]) -> str:
        src_m = re.search(r"""src=["']([^"']+)["']""", m.group(0), re.I)
        if not src_m:
            return m.group(0)
        src_url = urljoin(base_url, _html_mod.unescape(src_m.group(1)))
        for n, info in registry.items():
            url = _html_mod.unescape(info["url"])
            if url == src_url or url.replace("https://", "http://") == src_url.replace("https://", "http://"):
                return f" __IMG_{n}__ "
        return m.group(0)

    return _SPAN_IMG_RE.sub(_repl, html)


def _url_to_registry_key(registry: dict[int, dict[str, str]]) -> dict[str, int]:
    """Build lookup from URL → registry index.

    Normalises http/https, decodes HTML entities, and stores path-only keys
    so both absolute and relative URLs match.
    """
    mapping: dict[str, int] = {}
    for n, info in registry.items():
        url = info["url"]
        for variant in {url, _html_mod.unescape(url)}:
            mapping[variant] = n
            if variant.startswith("https://"):
                mapping["http://" + variant[8:]] = n
            elif variant.startswith("http://"):
                mapping["https://" + variant[4:]] = n
            if "://" in variant:
                path = variant.split("/", 3)[-1] if variant.count("/") >= 3 else variant
                mapping[path] = n
    return mapping


def _escape_md_alt(label: str) -> str:
    """Escape ``[`` and ``]`` in image alt text so they don't break markdown."""
    return label.replace("[", "&#91;").replace("]", "&#93;")


def postprocess_markdown(
    text: str, registry: dict[int, dict[str, str]], base_url: str,
) -> str:
    """Replace trafilatura image URLs and ``__IMG_N__`` markers with ``![label](img:N)``."""

    url_to_n = _url_to_registry_key(registry)

    def _replace_img_ref(m: re.Match[str]) -> str:
        url = m.group(2)
        n = url_to_n.get(url)
        if n is None and not url.startswith("http"):
            n = url_to_n.get(urljoin(base_url, url))
        if n is not None:
            label = registry[n].get("label", f"Image {n}")
            return f"![{_escape_md_alt(label)}](img:{n})"
        # Try with /Npx- normalised (wikimedia resizing)
        clean = re.sub(r"/\d+px-", "/", url)
        n = url_to_n.get(clean)
        if n is None and not clean.startswith("http"):
            n = url_to_n.get(urljoin(base_url, clean))
        if n is not None:
            label = registry[n].get("label", f"Image {n}")
            return f"![{_escape_md_alt(label)}](img:{n})"
        return m.group(0)

    text = _IMG_MD_RE.sub(_replace_img_ref, text)

    # Pass 2: __IMG_N__ markers (from span-wrapped orphan images)
    for n, info in sorted(registry.items(), reverse=True):
        label = info.get("label", f"Image {n}")
        text = text.replace(f"__IMG_{n}__", f"![{_escape_md_alt(label)}](img:{n})")

    return text


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


def extract_web_text(
    html_bytes: bytes, *, base_url: str = "",
) -> tuple[str, dict[int, dict[str, str]]]:
    """Return ``(text, image_registry)`` from trafilatura extraction.

    Pre-processes HTML (headings, math, images), runs trafilatura, then
    post-processes markdown to replace raw image URLs with ``![label](img:N)``
    references.
    """
    try:
        import trafilatura  # noqa: PLC0415
    except Exception:
        return "", {}

    html_str: str = html_bytes.decode("utf-8", errors="replace")

    # Build image registry from raw HTML (before any pre-processing)
    img_registry = collect_images(html_str, base_url)

    # Replace span-wrapped orphan images with markers
    html_str = replace_span_imgs(html_str, base_url, img_registry)

    # Math + headings
    html_str = headings_to_markdown(html_str)
    html_str = inline_math(html_str)

    raw = trafilatura.extract(
        html_str,
        include_comments=False,
        include_tables=True,
        include_formatting=True,
        include_images=True,
    )
    if not raw:
        return "", img_registry

    from src.text_cleanup import clean_nlm_text, tail_strip  # noqa: PLC0415

    text = tail_strip(raw).strip()
    text = clean_nlm_text(text)

    text = postprocess_markdown(text, img_registry, base_url)
    return text, img_registry


# ---------------------------------------------------------------------------
# Convenience: HTML pre-processing only (used by structured extraction path)
# ---------------------------------------------------------------------------


def preprocess_html(html: str, *, base_url: str = "") -> tuple[str, dict[int, dict[str, str]]]:
    """Pre-process HTML for extraction (math, headings, image markers).

    Returns ``(processed_html, image_registry)``. The returned HTML can be
    passed to ``trafilatura`` or ``extract_structured``.
    """
    img_registry = collect_images(html, base_url)
    html = replace_span_imgs(html, base_url, img_registry)
    html = headings_to_markdown(html)
    html = inline_math(html)
    return html, img_registry
