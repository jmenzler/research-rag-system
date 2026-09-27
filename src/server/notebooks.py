"""List collections and their notebooks with chunk counts.

Lifted from the original ``src.mcp_server.list_notebooks`` so both MCP (stdio)
and the HTTP API can share one implementation.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from pymilvus import MilvusClient

from src import config


def list_notebooks(collection: str | None = None) -> dict[str, Any]:
    """List collections and their notebooks with chunk counts.

    Pass *collection* to scope to one; omit for all (trading/ecology/notes/system).
    """
    client = MilvusClient(uri=config.MILVUS_URI)

    if collection:
        targets = [collection]
    else:
        all_cols = client.list_collections()
        targets = [c for c in all_cols if c in config.COLLECTIONS]

    out: list[dict[str, Any]] = []
    for col in targets:
        cnt: Counter[str] = Counter()
        it = client.query_iterator(
            collection_name=col,
            filter='notebook != ""',
            output_fields=["notebook"],
            batch_size=10000,
        )
        while True:
            batch = it.next()
            if not batch:
                break
            cnt.update(r["notebook"] for r in batch)
        it.close()
        notebooks = [
            {"name": nb, "chunks": n}
            for nb, n in sorted(cnt.items(), key=lambda x: -x[1])
        ]
        out.append(
            {
                "name": col,
                "total_chunks": sum(cnt.values()),
                "notebooks": notebooks,
            }
        )

    return {"collections": out}
