"""BM25-adversarial noise injection for robustness sensitivity scoring.

For each eval row with baseline faithfulness, this module:
1. Extracts content nouns from the question
2. Searches Milvus via BM25 for lexically-similar but (likely) irrelevant chunks
3. Filters out chunks already in the original retrieved set
4. Injects up to 3 noise chunks into the contexts
5. Re-synthesizes the answer with noisy contexts
6. Re-scores faithfulness on the augmented answer
7. Reports ΔF = baseline_F - F_with_noise

Small delta (≤0.05): the system is robust to lexical distractors.
Large delta (≥0.20): the system is vulnerable to relevance noise.

Toggle: only runs when ``extras={"noise": True}`` is passed to ``run_our_metrics``.
Default off to keep the regular eval cheap (~$0.05/eval at N=10).
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from typing import Any

from src.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Stopwords filtered from noun extraction.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "is", "at", "which", "on", "in", "a", "an", "and", "or",
    "but", "not", "to", "of", "for", "with", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "that", "this", "what",
    "these", "those", "can", "will", "just", "should", "now",
    "also", "has", "have", "are", "was", "were", "been", "does",
    "its", "does", "much", "very",
})

# Min word length for content nouns.
_MIN_WORD_LEN = 4

# Default max content nouns extracted per question.
_MAX_NOUNS = 8

# BM25 search top-K per noun.
_BM25_TOP_K = 20

# Max noise chunks to inject.
_MAX_NOISE_CHUNKS = 3


def _extract_content_nouns(question: str, max_nouns: int = _MAX_NOUNS) -> list[str]:
    """Extract content-bearing nouns from a question string.

    Uses simple regex: 4+ character alphabetic words, excluding stopwords.
    Returns deduplicated tokens in order of first appearance.
    """
    words = re.findall(r"[a-zA-Z]{4,}", question.lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for w in words:
        if w not in _STOPWORDS and w not in seen:
            seen.add(w)
            ordered.append(w)
            if len(ordered) >= max_nouns:
                break
    return ordered


def _bm25_search(
    query: str,
    collection: str,
    notebook: str,
    top_k: int = _BM25_TOP_K,
) -> list[str]:
    """Run BM25-only sparse search and return parent chunk texts.

    Two-step lookup matching the production retrieval path:
    1. BM25 over Milvus' ``sparse_embedding`` field returns child chunks
       (``id``, ``parent_chunk_id``).
    2. Parent texts are resolved via the parents sqlite db keyed on
       ``parent_chunk_id`` — Milvus only stores child rows, parent text
       lives in sqlite (per the Phase 1 architecture).

    Uses a single sparse ``AnnSearchRequest`` (no dense vector — only one
    request in ``hybrid_search`` reqs means BM25 is the sole ranker).
    """
    from pymilvus import AnnSearchRequest, RRFRanker  # noqa: PLC0415

    from src.config import PARENTS_DB  # noqa: PLC0415
    from src.milvus_client import get_client  # noqa: PLC0415
    from src.query.retrieve import _lookup_parents  # noqa: PLC0415

    client = get_client()

    if not notebook:
        expr = ""
    elif "," in notebook:
        nbs = [n.strip() for n in notebook.split(",") if n.strip()]
        quoted = ", ".join('"' + n + '"' for n in nbs)
        expr = "notebook in [" + quoted + "]"
    else:
        expr = f'notebook == "{notebook}"'

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=expr or None,
    )

    results = client.hybrid_search(
        collection_name=collection,
        reqs=[sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["parent_chunk_id"],
    )

    if not results or not results[0]:
        return []

    hits = results[0]
    parent_ids: list[str] = []
    seen_pids: set[str] = set()
    for hit in hits:
        pid = hit.get("entity", {}).get("parent_chunk_id")
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            parent_ids.append(pid)

    if not parent_ids:
        return []

    parents = _lookup_parents(parent_ids=parent_ids, db_path=PARENTS_DB)
    return [parents[pid].text for pid in parent_ids if pid in parents]


async def run_noise_sensitivity(
    row: dict[str, Any],
    judge_model: str,
    collection: str,
    notebook: str,
    *,
    milvus_client: Any = None,  # injectable for testing  # noqa: ANN401
) -> dict[str, Any] | None:
    """Run noise sensitivity for a single eval row.

    Returns None when noise injection fails (no BM25 hits, no baseline F, etc.).

    Returns a dict with:
        noise_delta_faithfulness: baseline_F - F_with_noise
        noise_baseline_f: original faithfulness score
        noise_f_with_noise: faithfulness after noise injection
        noise_n_injected: number of noise chunks injected
    """
    from src.eval.metrics import faithfulness  # noqa: PLC0415
    from src.query.generate import synthesize_answer  # noqa: PLC0415

    question: str = row["question"]
    original_contexts: list[str] = list(row.get("contexts", []))
    baseline_f = row.get("faithfulness")

    if baseline_f is None or not isinstance(baseline_f, (int, float)):
        return None
    if math.isnan(float(baseline_f)):
        return None

    # Extract nouns for BM25 search.
    nouns = _extract_content_nouns(question)
    if not nouns:
        nouns = [question]

    # BM25 search: combine all noun tokens into a query.
    bm25_query = " ".join(nouns[:5])
    noise_texts = _bm25_search(bm25_query, collection, notebook, top_k=_BM25_TOP_K)

    # Filter out texts already in the original contexts.
    noise_texts = [t for t in noise_texts if t not in original_contexts]

    if not noise_texts:
        return None

    # Take up to _MAX_NOISE_CHUNKS, shuffle to randomize position.
    noise_texts = random.sample(
        noise_texts, min(len(noise_texts), _MAX_NOISE_CHUNKS)
    )
    n_noise = len(noise_texts)

    # Build augmented contexts: original + noise (randomized order).
    augmented_contexts = list(original_contexts) + noise_texts
    random.shuffle(augmented_contexts)

    # Build fake RetrievedChunk objects for synthesis.
    fake_chunks: list[RetrievedChunk] = []
    for i, text in enumerate(augmented_contexts):
        from src.models import ChildChunk, ParentChunk  # noqa: PLC0415

        parent = ParentChunk(
            id=f"noise_parent_{i}",
            source_file="noise_injection",
            text=text,
            page_number=0,
            modality="text",
            notebook="noise",
        )
        child = ChildChunk(
            id=f"noise_child_{i}",
            parent_id=parent.id,
            text=text[:500],
            source_file="noise_injection",
            page_number=0,
            notebook="noise",
            modality="text",
        )
        fake_chunks.append(
            RetrievedChunk(
                child=child,
                parent=parent,
                dense_score=0.0,
                sparse_score=0.0,
                rerank_score=0.0,
            )
        )

    # Re-synthesize with noise.
    try:
        t0 = time.perf_counter()
        noise_answer, _, __ = synthesize_answer(question, fake_chunks)
        synth_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "noise: re-synthesized in %dms with %d noise chunks",
            synth_ms,
            n_noise,
        )
    except Exception as exc:
        logger.warning("noise: re-synthesis failed: %s", exc)
        return None

    # Re-score faithfulness on the noise-augmented answer.
    noise_row: dict[str, Any] = {
        "user_input": question,
        "response": noise_answer,
        "retrieved_contexts": augmented_contexts,
    }
    try:
        f_with_noise = await faithfulness(noise_row, judge_model)
    except Exception as exc:
        logger.warning("noise: faithfulness re-scoring failed: %s", exc)
        return None

    if math.isnan(f_with_noise):
        return None

    delta = float(baseline_f) - f_with_noise

    return {
        "noise_delta_faithfulness": round(delta, 4),
        "noise_baseline_f": float(baseline_f),
        "noise_f_with_noise": round(f_with_noise, 4),
        "noise_n_injected": n_noise,
    }


async def compute_all_noise(
    rows: list[dict[str, Any]],
    judge_model: str,
    collection: str,
    notebook: str,
    extras: dict[str, bool] | None = None,
) -> list[dict[str, Any] | None]:
    """Run noise sensitivity for all rows (opt-in, toggled by extras["noise"]).

    Returns a list parallel to *rows* with per-row noise results or None.
    """
    if not extras or not extras.get("noise"):
        return []

    results: list[dict[str, Any] | None] = []
    for row in rows:
        try:
            result = await run_noise_sensitivity(
                row, judge_model, collection, notebook,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("noise: run_noise_sensitivity failed for row: %s", exc)
            results.append(None)
    return results
