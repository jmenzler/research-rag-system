"""Resolve arbitrary seed inputs to S2 corpus IDs.

Accepts arxiv IDs, arxiv URLs, DOIs, DOI URLs, S2 paper-id SHAs, S2 corpus
IDs, raw publisher URLs, and free-form title strings; returns a
:class:`SeedResult` with the canonical S2 ``corpusid`` plus provenance.

Lookups go against the local ``paper_meta.sqlite`` resolver index — a lean
``corpusid → {arxiv_id, doi, title, year, citationcount}`` table decoupled
from the KuzuDB citation graph (resolution is a point lookup, not a graph
op). If a seed is unresolved there (paper too new for the snapshot, or an S2
SHA which the index does not carry), the caller can fall back to the S2 Graph
API — see :mod:`src.citations.s2_client`.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Literal

SeedKind = Literal[
    "arxiv_id",
    "doi",
    "s2_id",
    "corpus_id",
    "url",
    "title",
]


@dataclass(frozen=True)
class SeedResult:
    corpus_id: int
    title: str
    arxiv_id: str | None
    doi: str | None
    confidence: float
    source: SeedKind


class SeedUnresolvableError(RuntimeError):
    """Raised when a seed input cannot be mapped to any S2 paper."""


_ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?([0-9]{4}\.[0-9]{4,6})(?:v\d+)?$", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"^https?://arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})(?:v\d+)?(?:\.pdf)?/?$",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_DOI_URL_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_S2_ID_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_CORPUS_ID_RE = re.compile(r"^\d{1,12}$")


def classify_input(s: str) -> SeedKind:
    s = s.strip()
    if _ARXIV_URL_RE.match(s):
        return "url"
    if _DOI_URL_RE.match(s):
        return "url"
    if s.startswith("http"):
        return "url"
    if _ARXIV_ID_RE.match(s):
        return "arxiv_id"
    if _DOI_RE.match(s):
        return "doi"
    if _S2_ID_RE.match(s):
        return "s2_id"
    if _CORPUS_ID_RE.match(s):
        return "corpus_id"
    return "title"


def _normalize_arxiv(s: str) -> str:
    m = _ARXIV_ID_RE.match(s) or _ARXIV_URL_RE.match(s)
    if not m:
        raise SeedUnresolvableError(f"not an arxiv id/url: {s}")
    return m.group(1).lower()


def _normalize_doi(s: str) -> str:
    m = _DOI_RE.match(s) or _DOI_URL_RE.match(s)
    if not m:
        raise SeedUnresolvableError(f"not a doi/doi-url: {s}")
    return m.group(1).lower()


def _row_to_result(row: tuple[int, str | None, str | None, str | None],
                   confidence: float, source: SeedKind) -> SeedResult:
    return SeedResult(
        corpus_id=int(row[0]),
        title=row[1] or "",
        arxiv_id=row[2],
        doi=row[3],
        confidence=confidence,
        source=source,
    )


def resolve(con: sqlite3.Connection, seed: str) -> SeedResult:
    """Resolve a single seed via the local ``paper_meta.sqlite`` index.

    Raises :class:`SeedUnresolvableError` when nothing matches; the caller can
    fall back to the S2 Graph API for fresh papers.
    """
    kind = classify_input(seed)

    if kind in ("arxiv_id", "url") and (_ARXIV_ID_RE.match(seed) or _ARXIV_URL_RE.match(seed)):
        arxiv = _normalize_arxiv(seed)
        row = con.execute(
            "SELECT corpusid, title, arxiv_id, doi FROM paper_meta WHERE arxiv_id = ? LIMIT 1",
            [arxiv],
        ).fetchone()
        if row is not None:
            return _row_to_result(row, 1.0, "arxiv_id")

    if kind in ("doi", "url") and (_DOI_RE.match(seed) or _DOI_URL_RE.match(seed)):
        doi = _normalize_doi(seed)
        row = con.execute(
            "SELECT corpusid, title, arxiv_id, doi FROM paper_meta WHERE lower(doi) = ? LIMIT 1",
            [doi],
        ).fetchone()
        if row is not None:
            return _row_to_result(row, 1.0, "doi")

    if kind == "corpus_id":
        row = con.execute(
            "SELECT corpusid, title, arxiv_id, doi FROM paper_meta WHERE corpusid = ? LIMIT 1",
            [int(seed)],
        ).fetchone()
        if row is not None:
            return _row_to_result(row, 1.0, "corpus_id")

    if kind == "s2_id":
        # The resolver index carries arxiv/doi external ids but not the S2
        # paper-id SHA, so a 40-char hex id can only be resolved via the S2
        # Graph API fallback.
        raise SeedUnresolvableError(
            f"s2 paper-id SHA not in local index (use corpus_id/arxiv/doi, "
            f"or the S2 API fallback): {seed!r}"
        )

    if kind == "title":
        row = con.execute(
            "SELECT corpusid, title, arxiv_id, doi FROM paper_meta "
            "WHERE lower(title) = lower(?) ORDER BY citationcount DESC LIMIT 1",
            [seed],
        ).fetchone()
        if row is not None:
            return _row_to_result(row, 0.85, "title")

    raise SeedUnresolvableError(f"no local match for {kind}: {seed!r}")
