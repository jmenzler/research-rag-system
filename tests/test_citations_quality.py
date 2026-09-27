"""Unit tests for the paper reliability scorer (pure, DB-free)."""

from __future__ import annotations

import pytest

from src.citations.quality import Weights, age_bucket, score_quality


def _base(**over: object) -> dict[str, object]:
    args: dict[str, object] = {
        "influential_cites": 10,
        "reference_count": 30,
        "publication_types": ["JournalArticle"],
        "has_venue": True,
        "venue_prestige": 5.0,
        "expected_infl_cites": 10.0,
        "author_hindexes": [20],
        "age_years": 5.0,
        "date_known": True,
    }
    args.update(over)
    return args


def test_fwci_world_average_maps_to_half() -> None:
    # influential == expected → FWCI 1.0 → impact 0.5.
    s = score_quality(**_base(influential_cites=10, expected_infl_cites=10.0))  # type: ignore[arg-type]
    assert s.impact_fwci == pytest.approx(1.0)
    assert s.impact == pytest.approx(0.5)


def test_fwci_above_field_average_scores_higher() -> None:
    lo = score_quality(**_base(influential_cites=5, expected_infl_cites=10.0))  # type: ignore[arg-type]
    hi = score_quality(**_base(influential_cites=40, expected_infl_cites=10.0))  # type: ignore[arg-type]
    assert hi.impact > lo.impact
    assert hi.impact_fwci == pytest.approx(4.0)


def test_young_paper_impact_is_low_confidence_and_excluded() -> None:
    s = score_quality(**_base(age_years=0.2, influential_cites=0, expected_infl_cites=None))  # type: ignore[arg-type]
    assert s.impact_confidence == "low"
    assert s.impact == 0.0
    assert s.impact_fwci is None
    assert "unproven" in s.verdict


def test_low_confidence_redistributes_weight_to_venue_author() -> None:
    # A fresh paper at a strong venue with strong authors should still score
    # respectably — impact's weight moves to venue+author, not a flat penalty.
    strong = score_quality(
        **_base(  # type: ignore[arg-type]
            age_years=0.3,
            influential_cites=0,
            expected_infl_cites=None,
            venue_prestige=10.0,
            author_hindexes=[60, 40],
        )
    )
    weak = score_quality(
        **_base(  # type: ignore[arg-type]
            age_years=0.3,
            influential_cites=0,
            expected_infl_cites=None,
            has_venue=False,
            publication_types=[],
            venue_prestige=0.0,
            author_hindexes=[2],
        )
    )
    assert strong.composite > weak.composite


def test_venue_tiers() -> None:
    reviewed = score_quality(**_base(publication_types=["Conference"], has_venue=True))  # type: ignore[arg-type]
    published = score_quality(**_base(publication_types=["Dataset"], has_venue=True))  # type: ignore[arg-type]
    preprint = score_quality(**_base(publication_types=[], has_venue=False))  # type: ignore[arg-type]
    assert reviewed.venue_label == "peer-reviewed" and reviewed.venue_tier == 1.0
    assert published.venue_label == "published" and published.venue_tier == 0.5
    assert preprint.venue_label == "preprint" and preprint.venue_tier == 0.2
    assert reviewed.venue > published.venue > preprint.venue


def test_author_max_dominates_blend() -> None:
    one_star = score_quality(**_base(author_hindexes=[60, 1, 1]))  # type: ignore[arg-type]
    all_weak = score_quality(**_base(author_hindexes=[3, 3, 3]))  # type: ignore[arg-type]
    assert one_star.author > all_weak.author


def test_grounding_thin_reference_list_scores_low() -> None:
    thin = score_quality(**_base(reference_count=2))  # type: ignore[arg-type]
    thorough = score_quality(**_base(reference_count=80))  # type: ignore[arg-type]
    assert thin.grounding < 0.2 < thorough.grounding


def test_weights_change_composite() -> None:
    row = _base(influential_cites=40, expected_infl_cites=10.0, author_hindexes=[2])
    iw = Weights(impact=0.9, venue=0.04, author=0.03, grounding=0.03)
    aw = Weights(impact=0.04, venue=0.03, author=0.9, grounding=0.03)
    impact_heavy = score_quality(**row, weights=iw)  # type: ignore[arg-type]
    author_heavy = score_quality(**row, weights=aw)  # type: ignore[arg-type]
    # High impact + weak authors: impact-weighted composite must exceed author-weighted.
    assert impact_heavy.composite > author_heavy.composite


def test_age_bucket_boundaries() -> None:
    assert age_bucket(0.1) == "0-3mo"
    assert age_bucket(0.4) == "3-6mo"
    assert age_bucket(0.9) == "6-12mo"
    assert age_bucket(1.5) == "1-2y"
    assert age_bucket(20.0) == "10y+"
