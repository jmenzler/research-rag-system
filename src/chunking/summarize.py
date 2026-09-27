"""Table summarizer: LLM-generated retrieval-surrogate summaries for oversize tables.

Tables that exceed the chunk text budget can't be stored as a single atomic chunk.
Following the literature consensus (Ragie, KX/Unstructured, LangChain table
benchmark), we generate a short paragraph summary for embedding while the full
table is stored separately for retrieval-time lookup.

Usage::

    from src.chunking.summarize import summarize_table

    summary = summarize_table(
        caption="Table 1: Entity bank composition.",
        markdown_body="| Type | Count | ...",
        breadcrumb="Paper Title > Section 3 > Results",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src import config
from src.llm import ChatMessage, LLMClient
from src.llm.messages import LLMError

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "table_summary"


def _load_template() -> str:
    """Load the base summarization prompt template (cached in memory)."""
    template_path = _PROMPT_DIR / "base.md"
    return template_path.read_text(encoding="utf-8")


# Loaded once at module import — the template is static.
_BASE_TEMPLATE: str | None = None


def _get_template() -> str:
    global _BASE_TEMPLATE
    if _BASE_TEMPLATE is None:
        _BASE_TEMPLATE = _load_template()
    return _BASE_TEMPLATE


def _body_hash(caption: str, markdown_body: str) -> str:
    """Deterministic hash for idempotency cache keys."""
    return hashlib.sha256(f"{caption}\n\n{markdown_body}".encode()).hexdigest()


def _read_cache(cache_path: Path) -> dict[str, str]:
    """Read the idempotency cache file. Returns empty dict if missing/corrupt."""
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("summarize_table: cache read failed for %s: %s", cache_path, exc)
    return {}


def _write_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Atomically write the idempotency cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(cache_path)


def summarize_table(
    *,
    caption: str,
    markdown_body: str,
    breadcrumb: str,
    client: LLMClient | None = None,
    model: str | None = None,
    cache_path: Path | None = None,
) -> str:
    """Return a 100-150 token retrieval-surrogate summary for an oversize table.

    Args:
        caption: Original table caption (may be empty).
        markdown_body: GFM markdown table body (header + separator + rows).
        breadcrumb: Section breadcrumb (e.g. "Title > Section 3 > Results").
        client: Shared ``LLMClient`` instance (avoids repeated construction).
        model: Override model (default: ``config.TABLE_SUMMARY_MODEL``).
        cache_path: Optional path to a JSON idempotency cache. Reads on entry,
            writes on success. Cache key is ``sha256(caption + body)``.

    Returns:
        Summary string (100-150 tokens, plain text), or ``""`` on any failure.
    """
    # --- Cache check ---
    bh = _body_hash(caption, markdown_body)
    if cache_path is not None:
        cache = _read_cache(cache_path)
        if bh in cache:
            logger.info("summarize_table: cache hit for %s", bh[:12])
            return cache[bh]

    # --- Build client ---
    if client is None:
        client = LLMClient(model=model or config.TABLE_SUMMARY_MODEL)

    # --- Build prompt ---
    template = _get_template()
    prompt = template.format(
        breadcrumb=breadcrumb,
        caption=caption if caption else "Untitled table",
        markdown_body=markdown_body,
    )

    # --- Call LLM ---
    try:
        resp = client.generate(
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=400,
            json_mode=False,
        )
    except LLMError as exc:
        logger.warning("summarize_table: LLM call failed: %s", exc)
        return ""

    summary = resp.text.strip()

    # --- Cache write ---
    if cache_path is not None and summary:
        cache = _read_cache(cache_path)
        cache[bh] = summary
        _write_cache(cache_path, cache)

    return summary
