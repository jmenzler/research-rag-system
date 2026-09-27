"""Text quality gates — used at extraction time and during post-extraction audit.

Two complementary signals:
1. Literal fail-page token match (`FAIL_PAGE_SIGNATURES`) — strong, catches
   localized Cloudflare challenges, 404s, vendor-specific block notices.
2. Longest-paragraph floor (`MIN_LONGEST_PARAGRAPH`) — catches nav/menu pages
   without literal tokens (Token Terminal, stock-quote dashboards, GitHub nav).

Threshold calibrated empirically against the 621-file trading corpus:
  - T&F paywall-landing nav: max_paragraph=63
  - PyEventBT documentation nav: max_paragraph=24
  - User-authored markdown bullet notes: max_paragraph=74+
  - Real academic paper bodies (Docling-extracted): max_paragraph=100+
70 is the cleanest split between bullet-only menus and bullet-heavy real prose.
"""

from __future__ import annotations

from src.cleanup.signatures import FAIL_PAGE_SIGNATURES

MIN_TEXT_LEN = 400          # extracted text below this = likely empty/error response
MIN_LONGEST_PARAGRAPH = 70  # max-paragraph floor for substantive prose
PDF_MAGIC = b"%PDF-"        # rejects HTML-served-as-PDF (silent 403→login redirect)


def pdf_is_openable(path: object) -> tuple[bool, str]:
    """True if pypdfium2 can open the PDF and report at least one page.

    Catches silent-truncation downloads (partial response cached as success) where
    the magic bytes are intact but the trailing xref/trailer are missing. We use
    pypdfium2 directly because that's MinerU's loader; "opens here" matches "opens
    in the parser." Some real PDFs lack a literal `%%EOF` yet pypdfium2 reads
    them — a pure text-grep gate would falsely reject those.
    """
    try:
        import pypdfium2 as pp  # type: ignore[import-untyped]  # noqa: PLC0415
    except Exception as e:
        return True, f"pypdfium2_unavailable:{type(e).__name__}"  # fail-open
    try:
        doc = pp.PdfDocument(str(path))
        n = len(doc)
    except Exception as e:
        return False, f"pdf_unopenable:{type(e).__name__}"
    if n < 1:
        return False, "pdf_zero_pages"
    return True, "ok"


def longest_paragraph_chars(text: str) -> int:
    """Length of the longest newline-delimited run.

    Real article bodies have at least one prose paragraph that crosses the floor.
    Nav pages and bullet-only block pages are all-short-lines.
    """
    return max((len(p.strip()) for p in text.split("\n") if p.strip()), default=0)


def detect_fail_signature(text: str) -> str | None:
    """Return a fail reason if the text looks like a failed extraction, else None.

    Two-pass: literal-token first (cheap), then paragraph-floor (catches token-free nav).
    Token check uses the lowercased first 1500 chars to limit cost on large docs.
    """
    lc = text.lower()[:1500]
    for sig in FAIL_PAGE_SIGNATURES:
        if sig in lc:
            return sig
    if longest_paragraph_chars(text) < MIN_LONGEST_PARAGRAPH:
        return f"max_paragraph<{MIN_LONGEST_PARAGRAPH}"
    return None


def quality_score(text: str) -> int:
    """Single integer for ordering retry attempts: pass=very-high, fail=length.

    Used by the retry chain to pick "best of attempts" when no fetcher passed.
    Sentinel +10000 ensures any passing attempt beats any failing attempt.
    """
    if detect_fail_signature(text) is None and len(text) >= MIN_TEXT_LEN:
        return 10000 + len(text)
    return len(text)
