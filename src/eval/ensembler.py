"""Majority-vote ensembler — mirrors ``ragas.metrics.base.Ensember.from_discrete``.

Used by ``context_precision`` and ``context_recall`` when ``n>1`` samples are
generated for the same input. We default ``n=1`` everywhere (RAGAS does too
for these metrics), so the ensembler is effectively a no-op pass-through.
The implementation is here for parity completeness.

Implementation diverges from RAGAS in one safe way: we copy each item dict
before mutating ``[attribute]`` so caller's data is untouched. RAGAS mutates
``inputs[0][i]`` in place.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any


def majority_vote_discrete(
    inputs: list[list[dict[str, Any]]],
    attribute: str,
) -> list[dict[str, Any]]:
    """Combine N parallel samples (each a list of L dicts) into a single list
    of L dicts, where ``attribute`` is the majority-vote across the N samples.

    On ties, ``Counter.most_common(1)`` returns the first-encountered key
    (Python 3.7+ Counter preserves insertion order), so the first sample's
    verdict wins on a tie — same as RAGAS.

    Args:
        inputs:    List of N samples; each sample is a list of L item dicts.
        attribute: The dict key whose value gets majority-voted.

    Returns:
        List of L dicts taken from ``inputs[0]`` with ``attribute`` overwritten
        by the majority vote. Other fields are preserved from sample 0.

    Edge cases (mirror RAGAS):
        - Empty ``inputs`` → ``[]``.
        - Samples of unequal length → warn, return ``inputs[0]``.
        - Any sample-item missing ``attribute`` → warn, return ``inputs[0]``.
        - Single sample (n=1) → returned as-is.
    """
    if not inputs:
        return []
    if not all(len(item) == len(inputs[0]) for item in inputs):
        warnings.warn("All inputs must have the same length", UserWarning, stacklevel=2)
        return inputs[0]
    if not all(attribute in item for inp in inputs for item in inp):
        warnings.warn(
            f"All inputs must have {attribute} attribute", UserWarning, stacklevel=2
        )
        return inputs[0]
    if len(inputs) == 1:
        return inputs[0]

    result: list[dict[str, Any]] = []
    for i in range(len(inputs[0])):
        # Copy the first sample's dict so we don't mutate caller's data.
        item = dict(inputs[0][i])
        verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
        item[attribute] = Counter(verdicts).most_common(1)[0][0]
        result.append(item)
    return result
