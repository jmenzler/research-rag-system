"""Model identifiers and embedding provider selection."""
from __future__ import annotations

import logging
import os

from src.models import Usage

logger = logging.getLogger(__name__)

# Generation / judging / OCR / reranking
GEN_MODEL: str = os.getenv("GEN_MODEL", "deepseek-v4-pro")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite-preview")
OCR_MODEL: str = os.getenv("OCR_MODEL", "gemini-3.1-flash-lite-preview")
RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
TABLE_SUMMARY_MODEL: str = os.getenv("TABLE_SUMMARY_MODEL", "gemini-3.1-flash-lite-preview")
DECOMPOSE_MODEL: str = os.getenv("DECOMPOSE_MODEL", "deepseek-v4-flash")

# Embedding provider — format is always ``provider:model/name``.
EMBEDDING_PROVIDER: str = os.getenv(
    "EMBEDDING_PROVIDER", "gemini:gemini-embedding-2"
)

_EMBED_DIM_BY_PROVIDER: dict[str, int] = {
    "gemini:gemini-embedding-2": 3072,
    "openrouter:perplexity/pplx-embed-v1-4b": 2560,
    "openrouter:perplexity/pplx-embed-v1-0.6b": 1024,
    "openrouter:qwen/qwen3-embedding-8b": 4096,
    "openrouter:qwen/qwen3-embedding-4b": 2048,
    "openrouter:openai/text-embedding-3-large": 3072,
    "openrouter:openai/text-embedding-3-small": 1536,
    "openrouter:baai/bge-m3": 1024,
}

EMBED_MODEL: str = EMBEDDING_PROVIDER.split(":", 1)[1]
EMBED_DIM: int = _EMBED_DIM_BY_PROVIDER.get(EMBEDDING_PROVIDER, 3072)

if EMBEDDING_PROVIDER not in _EMBED_DIM_BY_PROVIDER:
    raise RuntimeError(
        f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
        f"Known: {sorted(_EMBED_DIM_BY_PROVIDER.keys())}"
    )


# ---------------------------------------------------------------------------
# LLM pricing table — USD per 1M tokens.
# Source: provider pricing pages, May 2026. Verify quarterly.
# cache_miss: input tokens that missed the cache (full price).
# cache_hit:  input tokens that hit the cache (discounted).
# output:     completion tokens.
# reasoning:  chain-of-thought tokens (same rate as output for most models).
# ---------------------------------------------------------------------------

_LLM_PRICES_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {
        "cache_miss": 0.27,
        "cache_hit": 0.054,
        "output": 1.10,
        "reasoning": 1.10,
    },
    "deepseek-v4-flash": {
        "cache_miss": 0.14,
        "cache_hit": 0.028,
        "output": 0.28,
        "reasoning": 0.28,
    },
    "gemini-2.5-flash": {
        "cache_miss": 0.30,
        "cache_hit": 0.075,
        "output": 2.50,
        "reasoning": 2.50,
    },
    "gemini-3-flash-preview": {
        "cache_miss": 0.30,
        "cache_hit": 0.075,
        "output": 2.50,
        "reasoning": 2.50,
    },
    "gemini-3-pro-preview": {
        "cache_miss": 1.25,
        "cache_hit": 0.31,
        "output": 10.00,
        "reasoning": 10.00,
    },
    "gemini-3-flash-lite-preview": {
        "cache_miss": 0.10,
        "cache_hit": 0.025,
        "output": 0.40,
        "reasoning": 0.40,
    },
    # Same price tier as gemini-3-flash-lite-preview — verified May 2026.
    # Added here because JUDGE_MODEL / OCR_MODEL / TABLE_SUMMARY_MODEL all
    # default to this model (project policy: no 2.5 defaults).
    "gemini-3.1-flash-lite-preview": {
        "cache_miss": 0.10,
        "cache_hit": 0.025,
        "output": 0.40,
        "reasoning": 0.40,
    },
}

# Emit each "unknown model" warning once per process.
_warned_models: set[str] = set()


def compute_cost_usd(model: str, usage: Usage) -> float | None:
    """Compute cost in USD for one LLM call given *usage* and the pricing table.

    Returns ``None`` for unknown models (logs a warning once per process).
    When ``usage.input_cache_hit + usage.input_cache_miss == 0``, treats all
    input tokens as cache-miss (conservative / correct upper bound).
    """
    rates = _LLM_PRICES_PER_M_TOKENS.get(model)
    if rates is None:
        if model not in _warned_models:
            logger.warning(
                "compute_cost_usd: unknown model %r — returning None. "
                "Add it to _LLM_PRICES_PER_M_TOKENS in src/config/models.py.",
                model,
            )
            _warned_models.add(model)
        return None

    cache_split_known = (usage.input_cache_hit + usage.input_cache_miss) > 0
    if cache_split_known:
        cache_miss_tokens = usage.input_cache_miss
        cache_hit_tokens = usage.input_cache_hit
    else:
        # Treat all input as cache-miss (worst-case, never underestimates).
        cache_miss_tokens = usage.input
        cache_hit_tokens = 0

    cost = (
        cache_miss_tokens * rates["cache_miss"]
        + cache_hit_tokens * rates["cache_hit"]
        + usage.output * rates["output"]
        + usage.reasoning * rates["reasoning"]
    ) / 1_000_000.0
    return cost
