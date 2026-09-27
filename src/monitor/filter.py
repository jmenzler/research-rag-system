"""LLM relevance classification for arXiv papers via Gemini Flash."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.llm import ChatMessage, LLMClient
from src.monitor.paper import Paper

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "monitor_relevance.txt"
_MODEL = "gemini-3.1-flash-lite-preview"
_MAX_WORKERS = 5


@dataclass
class RelevanceResult:
    paper: Paper
    relevant: bool
    summary: str


def _classify_one(paper: Paper, system_prompt: str, client: LLMClient) -> RelevanceResult:
    user_msg = f"Title: {paper.title}\n\nAbstract:\n{paper.abstract}"
    try:
        resp = client.generate(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_msg),
            ],
            json_mode=True,
            temperature=0.0,
            max_tokens=256,
        )
        parsed = json.loads(resp.text)
        relevant = bool(parsed.get("relevant", False))
        summary = str(parsed.get("summary", ""))[:120]
    except Exception:  # noqa: BLE001
        logger.warning(
            "classify failed for %s — defaulting to not relevant",
            paper.id,
            exc_info=True,
        )
        relevant = False
        summary = ""
    return RelevanceResult(paper=paper, relevant=relevant, summary=summary)


def classify_papers(papers: list[Paper]) -> list[RelevanceResult]:
    """Classify each paper for relevance using Gemini Flash.

    Runs up to _MAX_WORKERS calls in parallel. All papers are returned;
    callers filter by ``result.relevant``.
    """
    if not papers:
        return []

    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    client = LLMClient(model=_MODEL)
    results: list[RelevanceResult] = [None] * len(papers)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_classify_one, paper, system_prompt, client): i
            for i, paper in enumerate(papers)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "classify_papers: unexpected error for paper at index %d", idx, exc_info=True
                )
                results[idx] = RelevanceResult(
                    paper=papers[idx], relevant=False, summary=""
                )

    relevant_count = sum(1 for r in results if r.relevant)
    logger.info(
        "classify_papers: %d/%d papers relevant", relevant_count, len(papers)
    )
    return results
