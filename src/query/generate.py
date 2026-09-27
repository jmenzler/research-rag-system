"""Generation module for the RAG system.

Programmatic ``generate()`` builds a multimodal Gemini prompt from retrieved
chunks, calls the API, parses inline citations, logs the query, and returns
a ``RAGResponse``.

CLI defaults to RAW mode — print reranked parent chunks with [file:page]
headers, no LLM in the loop. Pass ``--synthesize`` to also produce an
LLM-synthesized answer on top.

CLI usage:
    # default: raw chunks only
    python -m src.generate \\
        --query "How is CEX hedge inventory rebalanced?" \\
        --collection notes --notebook example_topic

    # opt-in synthesis
    python -m src.generate \\
        --query "..." --collection trading --notebook example_notebook \\
        --synthesize
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai as _openai
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src import config
from src.models import Citation, RAGResponse, RetrievedChunk, Usage
from src.retrieval_metrics import _norm_source


def _is_retryable_error(exc: BaseException) -> bool:
    """Retry on Gemini 429 (rate limit) and 503 (overloaded / preview models)."""
    return isinstance(exc, APIError) and getattr(exc, "code", None) in (429, 503)


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _generate_content_with_retry(
    client: genai.Client,
    contents: Sequence[types.Content | types.Part | str],
    system_instruction: str,
    model: str,
) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=model,
        contents=contents,  # type: ignore[arg-type]
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=8000,
        ),
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_QUERIES_LOG = _LOGS_DIR / "queries.jsonl"

# Module-level lock — serializes concurrent writes to logs/queries.jsonl.
# Required when evaluate.py runs _build_rows_from_pipeline with pipeline_workers > 1.
_QUERIES_LOG_LOCK = threading.Lock()

# Two-stage parse: find every [...] block, then extract each `file:N` token inside.
# Handles [a:0], [a:0, b:image], and [a:0, b:5, c:image] correctly.
_CITATION_BLOCK_RE = re.compile(r"\[([^\[\]]+)\]")
_CITATION_TOKEN_RE = re.compile(r"([^\s,\[\]]+):(image|\d+)")
# Numbered citation: [1], [3, 5, 7]. Must be preceded by a non-identifier char
# (whitespace/punct/start) so we don't catch `array[3]` or `matrix[1,2]` —
# those have a letter directly before the opening bracket. Citations live in
# prose, surrounded by spaces or sentence punctuation.
_NUMERIC_BLOCK_RE = re.compile(r"(?<![A-Za-z0-9_])\[((?:\d+\s*,\s*)*\d+)\]")
# Sources block lines look like:
#   [1] foo.txt
#   [1] Sun & Boyd 2018 — foo.txt
# The optional `cite — ` prefix is captured separately so we can distinguish
# the human-friendly short_cite from the canonical filename used as the key.
_SOURCES_LINE_RE = re.compile(
    r"^\s*\[(\d+)\]\s+(?:(.+?)\s+[—–-]\s+)?(.+?)\s*$",
    re.MULTILINE,
)
_SOURCES_DIR = Path(__file__).resolve().parent.parent.parent / "sources"


def _clean_source_name(path_or_name: str) -> str:
    """Return a slug-bearing identifier for a source path.

    Post-flatten layout: ``sources/<collection>/<slug>/<leaf>`` where leaf ∈
    {source.pdf, pdf.txt, nlm.txt}. The leaf alone is meaningless — use the
    parent directory name (the slug) as the citation key. Falls through to
    the leaf filename for any non-canonical input (old logs, raw filenames).
    """
    p = Path(path_or_name)
    if p.parent.name and p.name in {"source.pdf", "pdf.txt", "nlm.txt"}:
        return p.parent.name
    return p.name


_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_SYSTEM_INSTRUCTION = (
    # ---- Role ----
    "You are a domain-expert research synthesist. Your job is not to summarise "
    "documents but to **explain the underlying mechanism** that the documents "
    "collectively describe — at the level of a senior researcher writing the "
    "paragraph their colleague needs to read.\n\n"
    # ---- Procedure (front-loaded so the model plans before stitching) ----
    "PROCEDURE — follow internally before writing the answer:\n"
    "1. Cluster the documents by what they actually say about the question. "
    "Different chunks may describe the same mechanism in different language, "
    "or may emphasise different facets (definition vs. derivation vs. "
    "implementation vs. critique).\n"
    "2. Identify the **load-bearing** facts: equations, parameters, empirical "
    "thresholds, named techniques. These are what the user came for.\n"
    "3. Notice contrasts. Where do the sources agree? Where do they differ in "
    "scope, assumptions, parameter choices? A real synthesis surfaces this — "
    "it does not flatten everything into one voice.\n"
    "4. Build a single integrated explanation that uses each document as "
    "evidence for a part of the whole. Aim for: load-bearing facts in the "
    "first paragraph; mechanism and assumptions in the body; contrasts and "
    "limits at the close.\n\n"
    # ---- Anti-extractive guardrails ----
    "AVOID:\n"
    "- Paragraph-per-chunk structure (a sign you are stitching, not synthesising).\n"
    "- Quoting math or prose verbatim when restatement in your own framing is "
    "  clearer. Equations are fine to keep symbolic and faithful, but the prose "
    "  around them must be yours.\n"
    "- Filler hedges ('this captures the trade-off', 'the model balances X '\n"
    "  'against Y') unless they add information not in the document text.\n\n"
    # Citation rules deliberately omitted from the system prompt — they live
    # in the footer (after the documents) so they are recency-weighted in
    # attention. Inline [N] only; the server renders the trailing Sources
    # block from metadata. Author names from the cite attribute may be used
    # naturally in prose ('Li et al. show that... [1]')."
)


# Trailing reminder appended after the <documents> block. Recency-weighted
# rules (the model attends most to the last instructions before generating)
# go here: the abstention policy and the strict format constraints. The
# system prompt above carries the role + procedure; this carries the rails.
_USER_PROMPT_FOOTER = (
    "\n\n--- ANSWER REQUIREMENTS ---\n"
    "Strict grounding. Every substantive claim must come from the documents "
    "above. Do not import outside knowledge.\n\n"
    "Abstention. If a document is missing the specific mechanism the question "
    "asks for, say so directly: \"The documents describe X but do not specify Y.\" "
    "Do not pad with adjacent material to manufacture an answer. Partial "
    "evidence → partial answer + explicit gap. No evidence → say so plainly.\n\n"
    "Citations. Inline only, format [N] or [N, M]. Do not emit a trailing "
    "Sources/References block — the system renders one. Do not write document "
    "indices, slugs, or filenames outside the [N] bracket.\n\n"
    "Math. Keep formulas faithful to the source using $...$ or $$...$$. "
    "Surrounding prose must be your own framing, not chunk-text restated.\n\n"
    "Stop after the final paragraph of your answer."
)


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def _provider_for(model: str) -> str:
    """Return 'gemini', 'deepseek', or 'openrouter' based on model name prefix."""
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openrouter"


_META_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}


def _slug_for(source_file: str) -> str:
    """Return the slug-bearing directory name for a source path.

    Post-flatten layout: ``sources/<collection>/<slug>/<leaf>`` where leaf is
    one of source.pdf / pdf.txt / nlm.txt — the slug is the parent dir.
    Non-canonical inputs fall through to the leaf stem.
    """
    p = Path(source_file)
    if p.parent.name and p.name in {"source.pdf", "pdf.txt", "nlm.txt"}:
        return p.parent.name
    return p.stem


def _load_meta(source_file: str, notebook: str) -> dict[str, Any] | None:
    """Load meta.json for a chunk, keyed by (source_file, notebook).

    Returns the parsed JSON object (top-level dict) or None if missing /
    unreadable. Cached by (source_file, notebook) — same source path can
    appear under multiple notebooks (e.g. before/after retag).
    """
    key = (source_file, notebook)
    if key in _META_CACHE:
        return _META_CACHE[key]

    slug = _slug_for(source_file)
    meta_path = _SOURCES_DIR / notebook / slug / "meta.json"
    meta: dict[str, Any] | None = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
    _META_CACHE[key] = meta
    return meta


def _lookup_short_cite(source_file: str, notebook: str) -> str | None:
    """Author-year cite (e.g. "Li et al. 2025") if present in meta.json."""
    meta = _load_meta(source_file, notebook)
    if not meta:
        return None
    md = meta.get("metadata") or {}
    sc = md.get("short_cite")
    return sc.strip() if isinstance(sc, str) and sc.strip() else None


def _lookup_title(source_file: str, notebook: str) -> str | None:
    """Document title if present in meta.json (top-level ``title`` field)."""
    meta = _load_meta(source_file, notebook)
    if not meta:
        return None
    title = meta.get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


def _xml_attr_escape(s: str) -> str:
    """Minimal XML attribute escape — handles quote/ampersand/angle brackets."""
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _doc_open_tag(i: int, parent: object) -> str:
    """Build ``<document index="N" title="..." cite="...">``.

    Page number, source slug, and modality are intentionally omitted from the
    prompt — the LLM doesn't need them to reason. Title and short_cite (when
    available) help prose quality (e.g. "Li et al. show...") without leaking
    rerank score or position-marker noise. The trailing Sources block is
    server-rendered after the LLM completes.
    """
    src: str = parent.source_file  # type: ignore[attr-defined]
    nb: str = parent.notebook  # type: ignore[attr-defined]
    title = _lookup_title(src, nb)
    cite = _lookup_short_cite(src, nb)

    parts = [f'<document index="{i}"']
    if title:
        parts.append(f'title="{_xml_attr_escape(title)}"')
    if cite:
        parts.append(f'cite="{_xml_attr_escape(cite)}"')
    return " ".join(parts) + ">"


def _build_text_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    """Plain-text prompt for OpenAI-compatible providers (no image bytes).

    Layout: question → documents → footer (strict-grounding + format rules).
    The footer goes *after* the documents on purpose — LLMs attend most
    strongly to the last instructions before they begin generating, so the
    rails belong there. Role + procedure live in the system prompt.
    """
    parts = [f"Question: {query}\n\n<documents>"]
    for i, rc in enumerate(retrieved, 1):
        parts.append(f"{_doc_open_tag(i, rc.parent)}\n{rc.parent.text}\n</document>")
    parts.append("</documents>")
    parts.append(_USER_PROMPT_FOOTER)
    return "\n".join(parts)


def _usage_from_gemini(response: Any) -> Usage:  # noqa: ANN401  — duck-typed SDK response
    """Extract token usage from a google-genai GenerateContentResponse.

    Gemini reports ``prompt_token_count`` (input), ``candidates_token_count``
    (visible output), and ``thoughts_token_count`` (reasoning, when the model
    is a thinking variant like 2.5-pro). Missing fields default to 0.

    Also reads ``cached_content_token_count`` → ``input_cache_hit`` when
    prompt caching is active (90% input discount on cache hits).
    """
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return Usage()
    inp = int(getattr(um, "prompt_token_count", 0) or 0)
    out = int(getattr(um, "candidates_token_count", 0) or 0)
    reasoning = int(getattr(um, "thoughts_token_count", 0) or 0)
    total = int(getattr(um, "total_token_count", 0) or 0)
    if total == 0:
        total = inp + out + reasoning
    cache_hit = int(getattr(um, "cached_content_token_count", 0) or 0)
    cache_miss = max(inp - cache_hit, 0) if cache_hit > 0 else 0
    return Usage(
        input=inp,
        output=out,
        reasoning=reasoning,
        total=total,
        input_cache_hit=cache_hit,
        input_cache_miss=cache_miss,
    )


def _usage_from_openai_compat(response: Any) -> Usage:  # noqa: ANN401  — duck-typed SDK response
    """Extract token usage from an OpenAI-compatible chat completion response.

    Handles DeepSeek (always), OpenRouter (provider-dependent), and reasoning
    models that report ``completion_tokens_details.reasoning_tokens``. Missing
    fields default to 0 — providers vary in what they report.

    Also reads DeepSeek-specific cache fields (``prompt_cache_hit_tokens`` /
    ``prompt_cache_miss_tokens``) when present (50× price differential).
    """
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    inp = int(getattr(u, "prompt_tokens", 0) or 0)
    out = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", 0) or 0)
    reasoning = 0
    details = getattr(u, "completion_tokens_details", None)
    if details is not None:
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
    if total == 0:
        total = inp + out + reasoning
    cache_hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
    return Usage(
        input=inp,
        output=out,
        reasoning=reasoning,
        total=total,
        input_cache_hit=cache_hit,
        input_cache_miss=cache_miss,
    )


def _generate_openai_compat(
    query: str, retrieved: list[RetrievedChunk], model: str
) -> tuple[str, int, Usage]:
    """Call OpenAI-compat API (DeepSeek/OpenRouter); return (answer, latency_ms, usage)."""
    provider = _provider_for(model)
    if provider == "deepseek":
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it.")
        client = _openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )
    else:
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or export it.")
        client = _openai.OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    prompt = _build_text_prompt(query, retrieved)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=8000,
        timeout=180.0,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    answer = response.choices[0].message.content or ""
    usage = _usage_from_openai_compat(response)
    return answer, latency_ms, usage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_mime_type(image_path: str) -> str:
    """Return the MIME type for *image_path* based on file extension.

    Raises:
        ValueError: if the extension is not one of the supported image types.
    """
    suffix = Path(image_path).suffix.lower()
    mime = _MIME_MAP.get(suffix)
    if mime is None:
        raise ValueError(
            f"Unsupported image extension {suffix!r} in {image_path!r}. "
            f"Supported: {list(_MIME_MAP)}"
        )
    return mime


def _build_contents(query: str, retrieved: list[RetrievedChunk]) -> list[types.Part | str]:
    """Construct the ordered list of content parts for the prompt.

    Layout:
    1. User query (text Part).
    2. <documents> ... </documents> with text or image chunks.
    3. Trailing footer (strict-grounding + format rules) — placed after the
       documents on purpose so the rails are recency-weighted in attention.
    """
    parts: list[types.Part | str] = [
        types.Part.from_text(text=f"Question: {query}\n"),
        types.Part.from_text(text="<documents>"),
    ]

    for i, rc in enumerate(retrieved, start=1):
        parent = rc.parent
        open_tag = _doc_open_tag(i, parent)

        if parent.modality == "image":
            parts.append(types.Part.from_text(text=open_tag))
            image_path = parent.image_path
            if image_path is None:
                parts.append(types.Part.from_text(text=f"(image bytes unavailable) {parent.text}"))
            else:
                mime = _detect_mime_type(image_path)
                image_bytes = Path(image_path).read_bytes()
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
            parts.append(types.Part.from_text(text="</document>"))
        else:
            parts.append(types.Part.from_text(text=f"{open_tag}\n{parent.text}\n</document>"))

    parts.append(types.Part.from_text(text="</documents>"))
    parts.append(types.Part.from_text(text=_USER_PROMPT_FOOTER))
    return parts


def _strip_llm_sources_block(answer: str) -> str:
    """Remove any trailing ``Sources:`` / ``References:`` block from the answer.

    Defensive: the prompt tells the LLM not to emit one, but models occasionally
    do anyway. We rebuild the block deterministically from metadata; whatever
    the LLM wrote would be redundant or wrong, so strip it.
    """
    # Find the last "\nSources:" or "\nReferences:" (case-insensitive) and
    # discard from there to end-of-answer. Trailing whitespace gets trimmed.
    body = answer
    for marker in ("\nsources:", "\nreferences:"):
        idx = body.lower().rfind(marker)
        if idx >= 0:
            body = body[:idx]
    return body.rstrip()


def _cited_indices(answer_body: str) -> list[int]:
    """Return the list of cited indices in first-appearance order.

    Parses every ``[N]`` / ``[N, M]`` block in *answer_body* (which should be
    the answer with any LLM Sources block already stripped). The strict
    ``_NUMERIC_BLOCK_RE`` rejects `array[3]`-style false positives.
    """
    seen: set[int] = set()
    out: list[int] = []
    for block in _NUMERIC_BLOCK_RE.finditer(answer_body):
        for tok in block.group(1).split(","):
            tok = tok.strip()
            if not tok.isdigit():
                continue
            n = int(tok)
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _render_sources_block(cited_indices: list[int], retrieved: list[RetrievedChunk]) -> str:
    """Build the trailing ``Sources:`` block deterministically from metadata.

    Format per line:
        [N] cite — title (p. P)        (when cite + title both present)
        [N] cite (p. P)                (cite only)
        [N] title (p. P)               (title only)
        [N] slug (p. P)                (fallback)

    Page suffix omitted when modality=="image" or page_number==0.
    Lines emitted in the same order indices appear in *cited_indices*.
    """
    lines: list[str] = ["Sources:"]
    for n in cited_indices:
        if n < 1 or n > len(retrieved):
            # Bad index from the LLM — show it but flag the fallback.
            lines.append(f"[{n}] (no document at this index)")
            continue
        rc = retrieved[n - 1]
        parent = rc.parent
        cite = _lookup_short_cite(parent.source_file, parent.notebook)
        title = _lookup_title(parent.source_file, parent.notebook)
        slug = _slug_for(parent.source_file)

        # Pick the most informative human label available.
        if cite and title:
            head = f'{cite} — "{title}"'
        elif cite:
            head = cite
        elif title:
            head = title
        else:
            head = slug

        # Page suffix: meaningful only for paginated text sources.
        if parent.modality == "image":
            tail = ""
        elif parent.page_number > 0:
            tail = f" (p. {parent.page_number})"
        else:
            tail = ""

        lines.append(f"[{n}] {head}{tail}")
    return "\n".join(lines)


def _parse_citations(answer: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Extract structured Citation entries from *answer*.

    The authoritative ``index → chunk`` map is the 1-based position in
    *retrieved* (matches the prompt's ``<document index="N">``). We strip
    any LLM-emitted Sources block first and parse only the prose body —
    so abbreviated or hallucinated source names in a stray Sources block
    cannot poison the structured citations.

    Page number and modality come from the matching ``RetrievedChunk``.
    Dedup key is ``(source_file, page_number, modality)`` — two chunks
    with the same slug on different pages count as distinct citations,
    but the same slug+page across multiple [N] references collapses.
    First-occurrence order preserved.

    Also keeps backwards compatibility with the legacy ``[file.txt:PAGE]``
    format (older trace logs and non-conforming providers).
    """
    body = _strip_llm_sources_block(answer)

    source_by_index: dict[int, str] = {
        i: _clean_source_name(rc.parent.source_file) for i, rc in enumerate(retrieved, start=1)
    }
    page_by_index: dict[int, int] = {
        i: rc.parent.page_number for i, rc in enumerate(retrieved, start=1)
    }
    modality_by_index: dict[int, str] = {
        i: rc.parent.modality for i, rc in enumerate(retrieved, start=1)
    }

    # Modality lookup by slug for the legacy ``[file.txt:PAGE]`` parser below.
    modality_by_source: dict[str, str] = {}
    for rc in retrieved:
        sf = _clean_source_name(rc.parent.source_file)
        if sf not in modality_by_source:
            modality_by_source[sf] = rc.parent.modality

    seen: set[tuple[str, int, str]] = set()
    citations: list[Citation] = []

    # Pass 1: numeric format [N] / [N, M, ...].
    for block in _NUMERIC_BLOCK_RE.finditer(body):
        for tok in block.group(1).split(","):
            tok = tok.strip()
            if not tok.isdigit():
                continue
            idx = int(tok)
            source_file = source_by_index.get(idx)
            if not source_file:
                continue
            page_number = page_by_index.get(idx, 0)
            modality = modality_by_index.get(idx, "text")
            key = (source_file, page_number, modality)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        source_file=source_file,
                        page_number=page_number,
                        modality=modality,
                    )
                )

    # Pass 2 (backwards compat): legacy [file.txt:PAGE] format
    for block in _CITATION_BLOCK_RE.finditer(body):
        for m in _CITATION_TOKEN_RE.finditer(block.group(1)):
            source_file = m.group(1)
            page_raw = m.group(2)

            if page_raw == "image":
                page_number = 0
                modality = "image"
            else:
                page_number = int(page_raw)
                modality = modality_by_source.get(source_file, "text")

            key = (source_file, page_number, modality)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        source_file=source_file,
                        page_number=page_number,
                        modality=modality,
                    )
                )

    return citations


def _append_log(
    *,
    query: str,
    retrieved: list[RetrievedChunk],
    answer: str,
    citations: list[Citation],
    latency_ms: int,
    audit_dir: str | None = None,
) -> None:
    """Append one JSONL line to ``logs/queries.jsonl``.

    Creates the file (and ``logs/`` directory) if they do not exist.
    Fails loudly on I/O error.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "query": query,
        "retrieved_ids": [rc.child.id for rc in retrieved],
        # Rank-ordered, normalized, deduped source files. The retriever-agnostic
        # eval signal — see src/retrieval_metrics.py module docstring.
        "retrieved_sources": list(
            dict.fromkeys(_norm_source(rc.parent.source_file) for rc in retrieved)
        ),
        "modalities": [rc.parent.modality for rc in retrieved],
        # Persist parent texts so RAGAS evaluation can replay this query
        # without re-running the pipeline. Adds ~5-15 KB per query but
        # makes the log self-contained.
        "contexts": [rc.parent.text for rc in retrieved],
        "answer": answer,
        "latency_ms": latency_ms,
        "citations": [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "modality": c.modality,
            }
            for c in citations
        ],
    }
    if audit_dir is not None:
        record["audit_dir"] = audit_dir

    with _QUERIES_LOG_LOCK:
        with _QUERIES_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    *,
    model: str | None = None,
) -> tuple[str, int, Usage]:
    """Generate a raw answer for *query* given *retrieved* chunks.

    Returns the LLM's raw answer text (no sources-block postprocessing),
    latency in milliseconds, and token usage. Does NOT write to queries.jsonl.

    This is the bare synthesis primitive — use ``generate()`` for the full
    pipeline (with sources block, citations, and audit logging).
    """
    effective_model = model or config.GEN_MODEL
    provider = _provider_for(effective_model)

    if provider == "gemini":
        import httpx

        config.validate_api_key()
        client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options={"client_args": {"timeout": httpx.Timeout(180.0, connect=10.0)}},
        )
        contents = _build_contents(query, retrieved)
        t0 = time.perf_counter()
        response = _generate_content_with_retry(
            client, contents, _SYSTEM_INSTRUCTION, effective_model
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        answer: str = response.text or ""
        usage = _usage_from_gemini(response)
    else:
        answer, latency_ms, usage = _generate_openai_compat(query, retrieved, effective_model)

    return answer, latency_ms, usage


def generate(
    query: str,
    retrieved: list[RetrievedChunk],
    *,
    model: str | None = None,
    audit_dir: str | None = None,
) -> RAGResponse:
    """Generate a grounded, cited answer for *query* given *retrieved* chunks.

    Args:
        query:     The user's natural-language question.
        retrieved: Reranked chunks from ``src.retrieve.hybrid_search``.
        model:     Override config.GEN_MODEL for this call (e.g. "deepseek-v4-flash").
        audit_dir: If set, written into ``queries.jsonl`` so eval can locate
                   the per-query audit directory (e.g. for stage recall).

    Returns:
        A ``RAGResponse`` with the answer text, parsed citations, all retrieved
        chunks, and the end-to-end latency in milliseconds.
    """
    answer, latency_ms, usage = synthesize_answer(query, retrieved, model=model)

    # Server-rendered Sources block: strip whatever the LLM may have emitted
    # (the prompt forbids it but models are inconsistent), then append a
    # deterministic block built from meta.json keyed by which [N]s actually
    # appeared in prose. The user-facing answer is body + rendered block.
    body = _strip_llm_sources_block(answer)
    cited = _cited_indices(body)
    final_answer = f"{body}\n\n{_render_sources_block(cited, retrieved)}" if cited else body
    citations = _parse_citations(final_answer, retrieved)

    _append_log(
        query=query,
        retrieved=retrieved,
        answer=final_answer,
        citations=citations,
        latency_ms=latency_ms,
        audit_dir=audit_dir,
    )

    return RAGResponse(
        query=query,
        answer=final_answer,
        citations=citations,
        retrieved=retrieved,
        latency_ms=latency_ms,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def format_raw_chunks(retrieved: list[RetrievedChunk]) -> str:
    """Render reranked parent chunks with [file:page] headers — no LLM in the loop."""
    out: list[str] = []
    for i, rc in enumerate(retrieved, 1):
        p = rc.parent
        page_label = "image" if p.modality == "image" else str(p.page_number)
        out.append(
            f"\n--- [{i}] {p.source_file}:{page_label} "
            f"(modality={p.modality}, score={rc.rerank_score:.4f}) ---\n"
            f"{p.text}"
        )
    return "\n".join(out)


def _cli() -> None:
    """Retrieve → (optionally synthesize) → print.

    Default is RAW mode: print reranked parent chunks with [file:page] headers,
    no LLM call. Pass --synthesize to invoke the generator (Gemini) on top.
    """
    parser = argparse.ArgumentParser(
        description="Query the RAG system. Default returns raw reranked chunks; "
        "use --synthesize to also produce an LLM-synthesized answer.",
    )
    parser.add_argument(
        "--query",
        help="Natural-language question. Mutually exclusive with --query-file.",
    )
    parser.add_argument(
        "--query-file",
        help=(
            "Path to a file containing the query. "
            "Sidesteps shell escaping for multi-line / quote-heavy queries."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Generation model override. Examples: 'deepseek-v4-flash', "
            "'inclusionai/ling-2.6-flash'. Defaults to config.GEN_MODEL."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit machine-readable JSON to stdout "
            "(answer, citations, chunks, latency). Suppresses pretty-print."
        ),
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Milvus collection name (e.g. 'notes', 'trading', 'ecology', 'system').",
    )
    parser.add_argument(
        "--notebook",
        default="",
        help="Notebook / partition tag (e.g. 'example_notebook'). "
        "Omit or pass empty string to search ALL partitions in the collection.",
    )
    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=config.RETRIEVE_TOP_K,
        help=f"Candidates to retrieve before reranking (default: {config.RETRIEVE_TOP_K}).",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=config.RERANK_TOP_K,
        help=f"Chunks to keep after reranking (default: {config.RERANK_TOP_K}).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip synthesis; only print reranked chunks. "
        "Default behavior is to run the LLM generator (Gemini 3.1 Pro) on the chunks.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="In synthesis mode, also print the raw reranked chunks before the answer.",
    )
    parser.add_argument(
        "--no-decompose",
        action="store_true",
        help="Skip query decomposition (single hybrid_search on raw query).",
    )
    parser.add_argument(
        "--stepback",
        action="store_true",
        default=None,
        help=(
            "Enable step-back prompting "
            "(1 abstract foundational sub-query). Overrides config.USE_STEPBACK."
        ),
    )
    parser.add_argument(
        "--no-stepback",
        action="store_false",
        dest="stepback",
        help="Force-disable step-back even if config.USE_STEPBACK is True.",
    )
    parser.add_argument(
        "--crag",
        action="store_true",
        default=None,
        help="Enable CRAG-lite groundedness verification + retry-once on low score.",
    )
    parser.add_argument(
        "--no-crag",
        action="store_false",
        dest="crag",
        help="Force-disable CRAG-lite even if config.USE_CRAG_LITE is True.",
    )
    args = parser.parse_args()

    # Resolve query from --query or --query-file (exactly one required).
    if bool(args.query) == bool(args.query_file):
        raise SystemExit("Pass exactly one of --query or --query-file.")
    if args.query_file:
        args.query = Path(args.query_file).read_text(encoding="utf-8").strip()
    if not args.query:
        raise SystemExit("Query is empty.")

    # In --json mode, suppress incidental stdout chatter from intermediate steps.
    # We collect what we need into trace_record and emit one JSON object at the end.
    json_mode: bool = args.json

    effective_model = args.model or config.GEN_MODEL
    use_stepback = config.USE_STEPBACK if args.stepback is None else args.stepback
    use_crag = config.USE_CRAG_LITE if args.crag is None else args.crag

    # Delegate the entire pipeline (decompose via route_query, stepback,
    # retrieve, rerank, MMR, synthesis, CRAG, audit, finalize) to
    # run_query_pipeline. This is the canonical entry point — the same one
    # the eval harness (src.evaluate) calls. CLI/MCP/eval now share one path.
    # Lazy import to avoid circular: query_pipeline imports generate().
    from src.query.query_pipeline import run_query_pipeline  # noqa: PLC0415

    rag_response, trace = run_query_pipeline(
        query=args.query,
        collection=args.collection,
        notebook=args.notebook,
        decompose=not args.no_decompose,
        use_router=True,
        use_stepback=use_stepback,
        use_crag=use_crag,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        synthesize=not args.raw,
        gen_model=args.model,
    )

    retrieved: list[RetrievedChunk] = trace.get("retrieved_chunks", []) or []
    t_retrieve_ms = trace.get("retrieval", {}).get("total_latency_ms", 0) or 0
    total_latency_ms = trace.get("total_latency_ms", 0) or 0

    trace_latency: dict[str, Any] = {
        "retrieval_total_latency_ms": t_retrieve_ms,
        "synthesis_latency_ms": rag_response.latency_ms if rag_response else None,
        "total_latency_ms": total_latency_ms,
    }

    # Output: raw chunks (always for --raw, optional for --show-chunks)
    if (args.raw or args.show_chunks) and not json_mode:
        print("\n=== Retrieved Chunks (raw) ===")
        print(format_raw_chunks(retrieved))
        print(f"\n=== Retrieval latency ===\n{t_retrieve_ms} ms ({len(retrieved)} chunks)")

    if json_mode:
        _emit_json(args.query, rag_response, retrieved, trace_latency, model=effective_model)
        return

    if args.raw:
        return

    if not retrieved or rag_response is None:
        print("\nNo chunks retrieved — skipping synthesis.")
        return

    print("\n=== Synthesized Answer ===")
    print(rag_response.answer)

    print("\n=== Parsed Citations ===")
    if rag_response.citations:
        for i, cit in enumerate(rag_response.citations, 1):
            mod_suffix = f" ({cit.modality})" if cit.modality != "text" else ""
            page_suffix = f" p{cit.page_number}" if cit.page_number > 0 else ""
            print(f"[{i}] {cit.source_file}{page_suffix}{mod_suffix}")
    else:
        print("(no citations parsed)")

    print(
        f"\n=== Latency ===\nretrieval: {t_retrieve_ms} ms"
        f"  |  synthesis: {rag_response.latency_ms} ms"
        f"  |  chunks: {len(retrieved)}"
    )


def _emit_json(
    query: str,
    rag_response: RAGResponse | None,
    retrieved: list[RetrievedChunk],
    trace_record: dict[str, Any],
    model: str | None = None,
) -> None:
    """Emit a single JSON object on stdout summarizing the run.

    When synthesis ran (``rag_response`` is not None), each chunk in
    ``chunks`` keeps its metadata + rerank_score for audit trails but
    drops ``text`` — the LLM has already integrated those texts into
    ``answer``, and re-shipping them is bandwidth waste plus distraction
    for downstream callers (Claude reads `answer`, not the chunks).

    When raw retrieval (``rag_response`` is None), ``text`` is kept —
    that IS the response.

    ``model`` should be the *effective* model used for synthesis (i.e. the
    --model override if provided, else config.GEN_MODEL). Defaults to
    config.GEN_MODEL only as a fallback for callers that don't pass it.
    """
    synthesized = rag_response is not None
    out = {
        "query": query,
        "answer": rag_response.answer if rag_response else None,
        "citations": (
            [
                {
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "modality": c.modality,
                }
                for c in rag_response.citations
            ]
            if rag_response
            else []
        ),
        "chunks": [
            {
                "source_file": rc.parent.source_file,
                "page_number": rc.parent.page_number,
                "modality": rc.parent.modality,
                "rerank_score": rc.rerank_score,
                # text omitted when synthesized — answer already contains it.
                **({} if synthesized else {"text": rc.parent.text}),
            }
            for rc in retrieved
        ],
        "latency_ms": {
            "retrieval": trace_record.get("retrieval_total_latency_ms"),
            "synthesis": rag_response.latency_ms if rag_response else None,
            "total": trace_record.get("total_latency_ms"),
        },
        "usage": (
            {
                "input": rag_response.usage.input,
                "output": rag_response.usage.output,
                "reasoning": rag_response.usage.reasoning,
                "total": rag_response.usage.total,
            }
            if rag_response
            else None
        ),
        "model": model or config.GEN_MODEL,
        "n_chunks": len(retrieved),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
