"""Multi-fetcher retry orchestration with quality-gated stop-early.

Each fetcher attempts to retrieve text from a URL. The orchestrator runs them
in order, stops at the first quality-passing result, and falls back to the
best-of-attempts if none pass — that way we never lose retrievable content
just because the canonical fetcher returned a fail-page.

Used by `src/fetch/html.py` to chain trafilatura → trafilatura(browser UA)
→ opencli → curl(browser UA) for stubborn web sources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.cleanup.quality import detect_fail_signature, quality_score

# Browser UA pool — first try a benign default, then escalate to a real Chrome UA
# for sites that 403 on requests-style UAs (some CDN configs, ScienceDirect, etc).
DEFAULT_UA = "Mozilla/5.0 (compatible; rag-system/1.0)"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class FetchAttempt:
    """One fetcher's result. `text` empty if fetch errored out before any content."""

    fetcher: str
    text: str
    elapsed_s: float
    error: str | None = None
    extras: dict[str, str | int] = field(default_factory=dict)

    @property
    def passes_gate(self) -> bool:
        return bool(self.text) and detect_fail_signature(self.text) is None

    @property
    def score(self) -> int:
        return quality_score(self.text) if self.text else -1


@dataclass
class FetchResult:
    """Final orchestrator output: chosen attempt + full attempt log."""

    chosen: FetchAttempt | None
    attempts: list[FetchAttempt]

    @property
    def ok(self) -> bool:
        return self.chosen is not None and self.chosen.passes_gate

    @property
    def reason(self) -> str:
        if self.chosen and self.chosen.passes_gate:
            return f"ok:{self.chosen.fetcher}"
        if self.chosen:
            return (
                f"best_of_attempts:{self.chosen.fetcher}:"
                f"{detect_fail_signature(self.chosen.text) or 'short'}"
            )
        return "all_fetchers_failed"


# A fetcher takes (url) and returns a FetchAttempt. Errors must be caught and
# returned via the `error` field — no exceptions should leak out.
Fetcher = Callable[[str], FetchAttempt]


def run_chain(url: str, fetchers: list[Fetcher]) -> FetchResult:
    """Try each fetcher in order; stop at first quality-passing result.

    All attempts are recorded in `attempts` for debugging. If none pass the
    quality gate, the highest-scoring attempt is still chosen — better to keep
    a partial extraction than discard all signal entirely.
    """
    attempts: list[FetchAttempt] = []
    for fetcher in fetchers:
        attempt = fetcher(url)
        attempts.append(attempt)
        if attempt.passes_gate:
            return FetchResult(chosen=attempt, attempts=attempts)

    # No fetcher passed; pick highest-score attempt as fallback.
    candidates = [a for a in attempts if a.text]
    if not candidates:
        return FetchResult(chosen=None, attempts=attempts)
    best = max(candidates, key=lambda a: a.score)
    return FetchResult(chosen=best, attempts=attempts)
