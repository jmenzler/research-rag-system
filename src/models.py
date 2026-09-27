"""Frozen dataclass contracts for the RAG system.

All types are immutable (frozen=True). No business logic lives here — pure data shapes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def chunk_id(source_file: str, page_number: int, chunk_index: int, kind: str) -> str:
    """Return a stable, hash-based ID for a chunk.

    Args:
        source_file: Absolute or relative path to the source document.
        page_number:  0-based or 1-based page number (caller must be consistent).
        chunk_index:  Position of this chunk within its page/document.
        kind:         One of ``'parent'`` or ``'child'``.

    Returns:
        A 16-character hex string that is deterministic for the same inputs.
    """
    if kind not in {"parent", "child"}:
        raise ValueError(f"kind must be 'parent' or 'child', got {kind!r}")
    raw = f"{source_file}|{page_number}|{chunk_index}|{kind}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    """A parsed source document before chunking."""

    source_path: str
    notebook: str
    modality: str  # "pdf" | "image" | "audio" | "text"
    raw_text: str | None  # None for image modality
    metadata: dict[str, object]


@dataclass(frozen=True)
class ParentChunk:
    """Large context chunk (~1000 tokens) stored in SQLite, looked up at generation time."""

    id: str
    text: str
    source_file: str
    notebook: str
    modality: str
    page_number: int
    image_path: str | None = None  # set for image modality


@dataclass(frozen=True)
class ChildChunk:
    """Small precision chunk (~256 tokens) stored in Milvus for retrieval."""

    id: str
    parent_id: str
    text: str
    source_file: str
    notebook: str
    modality: str
    page_number: int


@dataclass(frozen=True)
class Citation:
    """A single inline citation attached to a RAG answer."""

    source_file: str
    page_number: int
    modality: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A child chunk that survived retrieval + reranking, with its parent context."""

    child: ChildChunk
    parent: ParentChunk
    dense_score: float
    sparse_score: float
    rerank_score: float


@dataclass(frozen=True)
class Usage:
    """Token accounting for an LLM synthesis call.

    ``reasoning`` is set only by reasoning-capable providers (DeepSeek-R1,
    OpenAI o-series, Gemini "thoughts"). For chat-style models it stays 0
    and ``output`` is the full visible-completion budget.
    ``total`` is the provider-reported sum if available, else
    ``input + output + reasoning``.

    ``input_cache_hit`` / ``input_cache_miss`` are populated when the provider
    exposes per-call cache accounting (DeepSeek: ``prompt_cache_hit_tokens`` /
    ``prompt_cache_miss_tokens``; Gemini: ``cached_content_token_count``).
    Both default to 0 — cost computation falls back to treating all input as
    cache-miss (correct upper bound) when the split is unavailable.
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    total: int = 0
    input_cache_hit: int = 0
    input_cache_miss: int = 0


@dataclass(frozen=True)
class RAGResponse:
    """Final response returned by the generate module."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    latency_ms: int
    usage: Usage = Usage()
