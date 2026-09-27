# long-ok-file
"""Milvus client singleton and collection/partition management.

Provides:
- ``get_client()``             — module-level singleton ``MilvusClient``
- ``ensure_collection(name)``  — idempotent collection creation with full schema + indexes
- ``ensure_partition(collection, notebook)`` — idempotent partition creation

Schema per CLAUDE.md:
  id               VARCHAR  primary key
  dense_embedding  FLOAT_VECTOR(EMBED_DIM)  HNSW index, COSINE metric (3072 gemini / 4096 qwen)
  sparse_embedding SPARSE_FLOAT_VECTOR   auto-populated by BM25 built-in function over `text`
  text             VARCHAR(4096)         source field for BM25 + child chunk text
  parent_chunk_id  VARCHAR
  source_file      VARCHAR
  notebook         VARCHAR  partition key
  modality         VARCHAR
  page_number      INT32
  created_at       INT64    unix UTC
"""

from __future__ import annotations

import threading

from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.config import EMBED_DIM, MILVUS_TEXT_MAX_CHARS, MILVUS_URI

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_client: MilvusClient | None = None

# Module-level lock — serializes concurrent hybrid_search / search / query calls.
# pymilvus 2.x thread-safety is not documented; the conservative approach is to
# serialize at the call site. The lock is acquired in retrieve.py before each
# call. Milvus searches are fast (~1-2s), so the lock is not a throughput cost.
_MILVUS_LOCK = threading.Lock()


def get_client() -> MilvusClient:
    """Return the module-level ``MilvusClient`` singleton, creating it on first call."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(name: str) -> None:
    """Create collection ``name`` with the RAG schema + indexes if it does not exist.

    Idempotent: safe to call multiple times.

    Args:
        name: Collection name, e.g. ``"notes"``, ``"trading"``, ``"ecology"``, ``"system"``.
    """
    client = get_client()

    if client.has_collection(name):
        return

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description=f"RAG collection: {name}",
    )

    # Primary key
    schema.add_field(
        "id",
        DataType.VARCHAR,
        max_length=64,
        is_primary=True,
    )

    # Dense vector — dim=EMBED_DIM, baked into the index (3072 gemini / 4096 qwen)
    schema.add_field(
        "dense_embedding",
        DataType.FLOAT_VECTOR,
        dim=EMBED_DIM,
    )

    # Text field — source for BM25 sparse vector; enable_analyzer=True required.
    # Max length sourced from src.config.MILVUS_TEXT_MAX_CHARS (env-overridable);
    # the chunker enforces a slightly smaller budget to leave headroom for
    # breadcrumb / ctx prefixes prepended at embed time.
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=MILVUS_TEXT_MAX_CHARS,
        enable_analyzer=True,
    )

    # Sparse vector — auto-populated by the BM25 built-in function
    schema.add_field(
        "sparse_embedding",
        DataType.SPARSE_FLOAT_VECTOR,
    )

    # Metadata fields
    schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=64)
    schema.add_field("source_file", DataType.VARCHAR, max_length=1024)
    schema.add_field(
        "notebook",
        DataType.VARCHAR,
        max_length=128,
        is_partition_key=True,
    )
    schema.add_field("modality", DataType.VARCHAR, max_length=16)
    schema.add_field("page_number", DataType.INT32)
    schema.add_field("created_at", DataType.INT64)

    # ------------------------------------------------------------------
    # BM25 built-in function (server-side sparse vector generation)
    # ------------------------------------------------------------------
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_embedding"],
    )
    schema.add_function(bm25_function)

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    index_params = client.prepare_index_params()

    # Dense: HNSW, COSINE, M=16, ef_construction=200
    index_params.add_index(
        field_name="dense_embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    # Sparse: SPARSE_INVERTED_INDEX, BM25 metric
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )


# ---------------------------------------------------------------------------
# Partition management
# ---------------------------------------------------------------------------

def ensure_partition(collection: str, notebook: str) -> None:
    """Validate that ``collection`` exists and that ``notebook`` is a known partition key value.

    When ``notebook`` is declared with ``is_partition_key=True`` (our schema), Milvus
    manages physical partitions internally via key routing. Manual ``create_partition``
    calls are blocked by the server in this mode. This function therefore only ensures
    the collection is created and that ``notebook`` is a non-empty string — the actual
    partition routing happens automatically when data is inserted.

    Calling this before any ingest serves as a pre-flight check (collection reachable,
    schema present) and documents the intended partition key value.

    Idempotent: safe to call multiple times.

    Args:
        collection: Collection name, e.g. ``"notes"``.
        notebook:   Partition key value / notebook tag, e.g. ``"example_topic"``.
    """
    if not notebook:
        raise ValueError("notebook must be a non-empty string")

    ensure_collection(collection)
    # Partition key mode: Milvus routes by `notebook` field value automatically.
    # Manual partition creation is disabled by the server when is_partition_key=True.
