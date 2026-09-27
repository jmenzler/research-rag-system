"""Rewriter env knobs — feature flags and token caps for the adaptive query rewriter.

Mirrors the pattern in ``retrieval.py`` (flat module, os.getenv reads).
Read by ``rewriter_configs.py::_baseline_rewriter_config``.
"""

from __future__ import annotations

import os

# Model used for the router/rewriter LLM call.
REWRITER_MODEL: str = os.getenv("REWRITER_MODEL", "deepseek-v4-flash")

# Feature-gate flags — default False so the refactor is a no-op at deploy time.
# Single global config: flip true via env to A/B test a prompt section
# (HyDE, stepback, etc.) across ALL queries. No per-intent gating.
REWRITER_EMIT_HYDE: bool = os.getenv("REWRITER_EMIT_HYDE", "false").lower() == "true"
REWRITER_EMIT_STEPBACK: bool = os.getenv("REWRITER_EMIT_STEPBACK", "false").lower() == "true"
REWRITER_EMIT_DISAMBIGUATION: bool = (
    os.getenv("REWRITER_EMIT_DISAMBIGUATION", "false").lower() == "true"
)
REWRITER_EMIT_FILTERS: bool = os.getenv("REWRITER_EMIT_FILTERS", "false").lower() == "true"

# Token caps.
REWRITER_DENSE_MAX_TOKENS: int = int(os.getenv("REWRITER_DENSE_MAX_TOKENS", "40"))
REWRITER_HYDE_MAX_TOKENS: int = int(os.getenv("REWRITER_HYDE_MAX_TOKENS", "80"))
REWRITER_STEPBACK_MAX_TOKENS: int = int(os.getenv("REWRITER_STEPBACK_MAX_TOKENS", "30"))
REWRITER_SUB_QUERIES_MIN: int = int(os.getenv("REWRITER_SUB_QUERIES_MIN", "1"))
REWRITER_SUB_QUERIES_MAX: int = int(os.getenv("REWRITER_SUB_QUERIES_MAX", "4"))
REWRITER_MAX_OUTPUT_TOKENS: int = int(os.getenv("REWRITER_MAX_OUTPUT_TOKENS", "2048"))
