"""Ingest pipeline — document parsing, chunking, contextualization, and Milvus insertion."""

from src.ingest.chunker import (
    _build_hierarchical_chunks,
    _count_tokens,
    _get_encoder,
    _split_into_child_chunks,
    _split_into_parent_chunks,
    _strip_tail_sections,
)
from src.ingest.contextualize import (
    CONTEXTUALIZE_MODEL,
    contextualize_chunk,
    contextualize_document,
)
from src.ingest.contextualize_corpus import contextualize_file
from src.ingest.ingest import ingest_file

__all__ = [
    "_build_hierarchical_chunks",
    "_count_tokens",
    "_get_encoder",
    "_split_into_child_chunks",
    "_split_into_parent_chunks",
    "_strip_tail_sections",
    "contextualize_chunk",
    "contextualize_document",
    "contextualize_file",
    "CONTEXTUALIZE_MODEL",
    "ingest_file",
]
