"""RAGAS evaluation quality thresholds."""
from __future__ import annotations

RAGAS_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}
