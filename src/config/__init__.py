"""Configuration constants for the RAG system.

Loads ``.env`` from the project root via python-dotenv. All constants are
module-level. ``GEMINI_API_KEY`` is read eagerly but validation is deferred
to ``validate_api_key()`` so that modules depending only on Milvus/chunking
can be imported without a key present.

Submodules:
- ``env``             — API keys, Milvus URI, key validation
- ``models``          — gen/judge/ocr/rerank/decompose model IDs + embedding provider
- ``chunking``        — chunk sizes, text field budget
- ``retrieval``       — retrieval/reranking params, CRAG flags
- ``intent_policies`` — per-intent retrieval policy multipliers + overrides
- ``quality``         — RAGAS quality thresholds
- ``paths``           — storage paths
- ``rewriter``        — rewriter env knobs (REWRITER_EMIT_*, token caps)
- ``rewriter_configs`` — global RewriterConfig dataclass + env-driven baseline
"""

from src.config.chunking import (
    CHILD_MIN_TOKENS,
    CHILD_OVERLAP,
    CHILD_TOKENS,
    CHUNK_TEXT_BUDGET_CHARS,
    MERGE_UNDER_TOKENS,
    MILVUS_TEXT_MAX_CHARS,
    PARENT_TOKENS,
)
from src.config.env import (
    DEEPSEEK_API_KEY,
    GEMINI_API_KEY,
    MILVUS_URI,
    OPENROUTER_API_KEY,
    validate_api_key,
)
from src.config.intent_policies import INTENT_MULTIPLIERS, INTENT_OVERRIDES
from src.config.models import (
    DECOMPOSE_MODEL,
    EMBED_DIM,
    EMBED_MODEL,
    EMBEDDING_PROVIDER,
    GEN_MODEL,
    JUDGE_MODEL,
    OCR_MODEL,
    RERANK_MODEL,
    TABLE_SUMMARY_MODEL,
)
from src.config.paths import (
    COLLECTIONS,
    PARENTS_DB,
    SOURCES_DIR,
    collection_dir,
    collection_glob_content_list,
    doc_dir,
    iter_doc_dirs,
    quarantine_dir,
)
from src.config.quality import RAGAS_THRESHOLDS
from src.config.retrieval import (
    CRAG_THRESHOLD,
    DECOMPOSE_MAX_SUBQUERIES,
    MAX_CHILDREN_PER_PARENT,
    MMR_LAMBDA,
    MMR_POOL_MULT,
    MMR_TOP_K,
    RERANK_MIN_KEEP,
    RERANK_SCORE_THRESHOLD,
    RERANK_SCORE_THRESHOLD_NVIDIA,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
    USE_CRAG_LITE,
    USE_MMR,
    USE_STEPBACK,
)
from src.config.rewriter import (
    REWRITER_DENSE_MAX_TOKENS,
    REWRITER_EMIT_DISAMBIGUATION,
    REWRITER_EMIT_FILTERS,
    REWRITER_EMIT_HYDE,
    REWRITER_EMIT_STEPBACK,
    REWRITER_HYDE_MAX_TOKENS,
    REWRITER_MAX_OUTPUT_TOKENS,
    REWRITER_MODEL,
    REWRITER_STEPBACK_MAX_TOKENS,
    REWRITER_SUB_QUERIES_MAX,
    REWRITER_SUB_QUERIES_MIN,
)
from src.config.rewriter_configs import RewriterConfig

__all__ = [
    # env
    "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
    "MILVUS_URI", "validate_api_key",
    # models
    "GEN_MODEL", "JUDGE_MODEL", "OCR_MODEL", "RERANK_MODEL", "TABLE_SUMMARY_MODEL",
    "DECOMPOSE_MODEL", "DECOMPOSE_MAX_SUBQUERIES",
    "EMBEDDING_PROVIDER", "EMBED_MODEL", "EMBED_DIM",
    # chunking
    "PARENT_TOKENS", "CHILD_TOKENS", "CHILD_OVERLAP", "CHILD_MIN_TOKENS",
    "MERGE_UNDER_TOKENS", "MILVUS_TEXT_MAX_CHARS", "CHUNK_TEXT_BUDGET_CHARS",
    # retrieval
    "RETRIEVE_TOP_K", "RERANK_TOP_K", "RERANK_SCORE_THRESHOLD",
    "RERANK_SCORE_THRESHOLD_NVIDIA", "RERANK_MIN_KEEP",
    "MAX_CHILDREN_PER_PARENT",
    "USE_STEPBACK", "USE_CRAG_LITE", "CRAG_THRESHOLD",
    "USE_MMR", "MMR_LAMBDA", "MMR_TOP_K", "MMR_POOL_MULT",
    "DECOMPOSE_MAX_SUBQUERIES",
    # intent policies
    "INTENT_MULTIPLIERS", "INTENT_OVERRIDES",
    # quality
    "RAGAS_THRESHOLDS",
    # paths
    "PARENTS_DB", "SOURCES_DIR", "COLLECTIONS",
    "collection_dir", "doc_dir", "quarantine_dir",
    "iter_doc_dirs", "collection_glob_content_list",
    # rewriter knobs
    "REWRITER_MODEL", "REWRITER_EMIT_HYDE", "REWRITER_EMIT_STEPBACK",
    "REWRITER_EMIT_DISAMBIGUATION", "REWRITER_EMIT_FILTERS",
    "REWRITER_DENSE_MAX_TOKENS", "REWRITER_HYDE_MAX_TOKENS",
    "REWRITER_STEPBACK_MAX_TOKENS", "REWRITER_SUB_QUERIES_MIN",
    "REWRITER_SUB_QUERIES_MAX", "REWRITER_MAX_OUTPUT_TOKENS",
    # rewriter configs
    "RewriterConfig",
]
