"""Apples-to-apples replacement for RAGAS' four core metrics.

Same prompts, same arithmetic, same example shots — but LLM calls go through
provider-side JSON schema enforcement (Gemini ``response_schema`` or OpenAI
``json_schema`` ``response_format``) via ``src.query.usage_track``. No
RAGAS / instructor parse-and-retry loop, no 30-minute stalls when a judge
returns malformed JSON.

Public API (re-exported from submodules):
- ``faithfulness(row, judge_model)`` — async, score in [0, 1] or NaN
- ``answer_relevancy(row, judge_model)`` — async, score in [0, 1] or NaN
- ``context_precision(row, judge_model)`` — async, score in [0, 1] or NaN
- ``context_recall(row, judge_model)`` — async, score in [0, 1] or NaN
- ``run_our_metrics(rows, judge_model, max_workers)`` — async runner

Submodules can be imported directly:
- ``src.eval.schemas`` — Pydantic output models
- ``src.eval.prompts`` — instruction strings + renderer
- ``src.eval.ensembler`` — ``majority_vote_discrete``
- ``src.eval.metrics`` — the four metric functions
- ``src.eval.runner`` — orchestrator
"""

from __future__ import annotations
