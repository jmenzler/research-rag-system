"""Notebook → audience-string + glossary-domain mapping.

Add new notebooks here. The `domain` field selects which
`glossary_<domain>.md` and `examples_<domain>.md` to load.
"""

from __future__ import annotations

from typing import TypedDict


class AudienceSpec(TypedDict):
    audience: str
    domain: str  # selects glossary_<domain>.md / examples_<domain>.md


# Trading partitions — quant finance domain
_TRADING = AudienceSpec(
    audience="quant finance researchers and market microstructure practitioners",
    domain="trading",
)

NOTEBOOK_AUDIENCE: dict[str, AudienceSpec] = {
    "finance_research": _TRADING,
    "machine_learning": AudienceSpec(
        audience="machine-learning researchers",
        domain="default",
    ),
    "research_briefs": AudienceSpec(
        audience="researchers reviewing synthesized literature briefs",
        domain="research_brief",
    ),
}

DEFAULT_AUDIENCE = AudienceSpec(
    audience="domain researchers",
    domain="default",
)


def for_notebook(notebook: str) -> AudienceSpec:
    return NOTEBOOK_AUDIENCE.get(notebook, DEFAULT_AUDIENCE)
