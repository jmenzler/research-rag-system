"""Paper reliability score (Layer-Q) — transparent weighted composite.

Inverse of :mod:`src.citations.scoring`'s niche bias: reliability rewards the
vetting/impact that niche demotes. Four normalized dimensions — impact
(FWCI-style: influential cites over the expected value for the paper's
field/type/age cohort), venue (peer-review tier × homegrown SJR-lite prestige),
author (h-index), grounding (reference count) — combined into a 0–100 composite
plus a verdict.

Pure and DB-free: the caller does the cohort/venue/author lookups and passes
primitives, so scoring is unit-testable without any store. Raw signals are
stored elsewhere and the score is computed here at query time (weights tunable
without re-ingest), mirroring the niche scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# publicationtypes that imply peer review.
_PEER_REVIEWED = {"JournalArticle", "Conference", "Review"}


def paper_age_years(
    pub_date: str | None, year: int | None, snapshot: date
) -> tuple[float, bool]:
    """Age in years at the snapshot date. Returns (age, date_known).

    Uses the full ``publicationdate`` when present (sub-year precision); else
    falls back to the year at ~mid-year (date_known=False, lower confidence).
    """
    if pub_date:
        try:
            return max(0.0, (snapshot - date.fromisoformat(pub_date)).days / 365.25), True
        except ValueError:
            pass
    if year:
        return max(0.0, snapshot.year - int(year) + 0.4), False
    return 0.0, False


def cohort_tier(publication_types: list[str] | None, has_venue: bool) -> str:
    """Coarse peer-review class used to key FWCI cohorts (and the venue tier)."""
    if set(publication_types or []) & _PEER_REVIEWED:
        return "reviewed"
    return "published" if has_venue else "preprint"

# Saturation half-points (value at which a signal reaches 0.5).
_HINDEX_HALF = 20.0
_REFCOUNT_HALF = 20.0
_VENUE_PRESTIGE_HALF = 5.0
# Below this age (years) the citation window is too short to trust impact.
_IMPACT_MIN_AGE = 0.5


@dataclass(frozen=True)
class Weights:
    venue: float = 0.30
    impact: float = 0.35
    author: float = 0.20
    grounding: float = 0.15


DEFAULT_WEIGHTS = Weights()


@dataclass(frozen=True)
class QualityScore:
    composite: float          # 0–100
    verdict: str
    # dimension sub-scores (0–1)
    impact: float
    venue: float
    author: float
    grounding: float
    # transparency: raw values + derived
    impact_fwci: float | None
    impact_confidence: str    # "ok" | "low"
    influential_cites: int
    venue_tier: float
    venue_label: str
    venue_prestige: float
    max_hindex: int
    mean_hindex: float
    n_authors: int
    reference_count: int
    age_years: float
    date_known: bool


def _saturate(x: float, half: float) -> float:
    """Map [0, ∞) → [0, 1) with value `half` mapping to 0.5 (Michaelis–Menten)."""
    x = max(0.0, x)
    return x / (x + half)


def _venue_tier(publication_types: list[str] | None, has_venue: bool) -> tuple[float, str]:
    types = set(publication_types or [])
    if types & _PEER_REVIEWED:
        return 1.0, "peer-reviewed"
    if has_venue:
        return 0.5, "published"
    return 0.2, "preprint"


def _age_bucket(age_years: float) -> str:
    if age_years < 0.25:
        return "0-3mo"
    if age_years < 0.5:
        return "3-6mo"
    if age_years < 1.0:
        return "6-12mo"
    if age_years < 2.0:
        return "1-2y"
    if age_years < 3.0:
        return "2-3y"
    if age_years < 5.0:
        return "3-5y"
    if age_years < 10.0:
        return "5-10y"
    return "10y+"


def age_bucket(age_years: float) -> str:
    """Public alias — the build pipeline keys baselines by the same buckets."""
    return _age_bucket(age_years)


def _verdict(tier_label: str, composite: float, confidence: str) -> str:
    if confidence == "low":
        return f"recent {tier_label}, unproven"
    if composite >= 70:
        return f"established, {tier_label}, high field-normalized impact"
    if composite >= 45:
        return f"{tier_label}, moderate impact"
    return f"{tier_label}, low signal"


def score_quality(
    *,
    influential_cites: int,
    reference_count: int,
    publication_types: list[str] | None,
    has_venue: bool,
    venue_prestige: float,
    expected_infl_cites: float | None,
    author_hindexes: list[int],
    age_years: float,
    date_known: bool,
    weights: Weights = DEFAULT_WEIGHTS,
) -> QualityScore:
    """Compute the reliability composite from already-looked-up primitives.

    ``expected_infl_cites`` is the cohort mean for the paper's
    (field, pubtype, age-bucket); ``None`` when the cohort is unknown/empty.
    """
    # Impact — FWCI-style: influential cites over cohort expectation.
    confidence = "ok"
    fwci: float | None
    if age_years < _IMPACT_MIN_AGE or not expected_infl_cites or expected_infl_cites <= 0:
        # Citation window too short, or no cohort baseline: impact unreliable.
        fwci = None
        impact = 0.0
        confidence = "low"
    else:
        fwci = influential_cites / expected_infl_cites
        impact = fwci / (fwci + 1.0)  # FWCI=1 (world avg) → 0.5
        if age_years < 1.0:
            confidence = "low"

    # Venue — peer-review tier scaled by prestige.
    tier, tier_label = _venue_tier(publication_types, has_venue)
    prestige_norm = _saturate(venue_prestige, _VENUE_PRESTIGE_HALF)
    venue = tier * (0.6 + 0.4 * prestige_norm)

    # Author — heavyweight (max) blended with team depth (mean).
    max_h = max(author_hindexes) if author_hindexes else 0
    mean_h = (sum(author_hindexes) / len(author_hindexes)) if author_hindexes else 0.0
    author = 0.6 * _saturate(max_h, _HINDEX_HALF) + 0.4 * _saturate(mean_h, _HINDEX_HALF)

    # Grounding — reference thoroughness.
    grounding = _saturate(reference_count, _REFCOUNT_HALF)

    # Composite — when impact is untrustworthy (young/no cohort), redistribute its
    # weight to venue + author so a fresh preprint isn't judged on absent citations.
    w = weights
    if confidence == "low":
        spread = w.impact
        eff_impact, eff_venue, eff_author, eff_ground = (
            0.0, w.venue + spread * 0.6, w.author + spread * 0.4, w.grounding,
        )
    else:
        eff_impact, eff_venue, eff_author, eff_ground = (
            w.impact, w.venue, w.author, w.grounding,
        )
    total_w = eff_impact + eff_venue + eff_author + eff_ground
    composite01 = (
        eff_impact * impact + eff_venue * venue
        + eff_author * author + eff_ground * grounding
    ) / total_w if total_w > 0 else 0.0
    composite = round(100.0 * composite01, 1)

    return QualityScore(
        composite=composite,
        verdict=_verdict(tier_label, composite, confidence),
        impact=round(impact, 4),
        venue=round(venue, 4),
        author=round(author, 4),
        grounding=round(grounding, 4),
        impact_fwci=round(fwci, 3) if fwci is not None else None,
        impact_confidence=confidence,
        influential_cites=int(influential_cites),
        venue_tier=tier,
        venue_label=tier_label,
        venue_prestige=round(venue_prestige, 3),
        max_hindex=int(max_h),
        mean_hindex=round(mean_h, 2),
        n_authors=len(author_hindexes),
        reference_count=int(reference_count),
        age_years=round(age_years, 2),
        date_known=date_known,
    )
