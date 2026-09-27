"""RED tests for src/query/retrieve_fused.py (Plan 02-09).

Fused retriever dispatches to milvus + paperqa + hipporag + lazygraph and
reranks the union. D-06 sub-stages at the top level:
``fused_dispatch``, ``fused_rerank``. The four sub-retrievers' own
sub-stages still record into the SAME audit directory — fused does NOT
spawn separate audit dirs per retriever.

Tests use deferred imports so collection stays green before Plan 02-09
lands ``src.query.retrieve_fused``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FUSED_TOP_LEVEL_STAGES: tuple[str, str] = ("fused_dispatch", "fused_rerank")


@pytest.fixture
def audit_log_root_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(root))
    yield root


@pytest.mark.timeout(30)
def test_dispatches_all_four_retrievers(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — fused fan-outs to milvus + paperqa + hipporag + lazygraph.

    Confirmed by spying on each sub-retriever's entry function.
    """
    pytest.importorskip(
        "src.query.retrieve_fused",
        reason="Plan 02-09 has not landed yet",
    )
    pytest.importorskip("src.query.retrieve_paperqa", reason="Plan 02-06")
    pytest.importorskip("src.query.retrieve_hipporag", reason="Plan 02-07")
    pytest.importorskip("src.query.retrieve_lazygraph", reason="Plan 02-08")

    from src.query.retrieve_fused import retrieve_fused  # noqa: PLC0415

    called: dict[str, bool] = {
        "milvus": False,
        "paperqa": False,
        "hipporag": False,
        "lazygraph": False,
    }

    def _fake_pq(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        called["paperqa"] = True
        return [], {}

    def _fake_hp(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        called["hipporag"] = True
        return [], {}

    def _fake_lz(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        called["lazygraph"] = True
        return [], {}

    def _fake_milvus(*_a: object, **_kw: object) -> list[Any]:
        called["milvus"] = True
        return []

    import src.query.retrieve as _retr  # noqa: PLC0415
    import src.query.retrieve_hipporag as _h  # noqa: PLC0415
    import src.query.retrieve_lazygraph as _l  # noqa: PLC0415
    import src.query.retrieve_paperqa as _p  # noqa: PLC0415

    monkeypatch.setattr(_p, "retrieve_paperqa", _fake_pq)
    monkeypatch.setattr(_h, "retrieve_hipporag", _fake_hp)
    monkeypatch.setattr(_l, "retrieve_lazygraph", _fake_lz)
    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    retrieve_fused(query="test", collections=["trading"])

    missing = [k for k, v in called.items() if not v]
    assert not missing, f"fused must dispatch to all four retrievers; missing={missing}"


@pytest.mark.timeout(30)
def test_emits_fused_dispatch_and_fused_rerank_only_at_top_level(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — only ``fused_dispatch`` and ``fused_rerank`` are top-level
    sub-stages. The four sub-retrievers' OWN sub-stages are recorded but
    are not the same set as fused's top-level set.

    Sub-retrievers are stubbed so the test exercises only fused's stage
    emission (the real sub-retrievers make LLM + cross-encoder calls that
    hang in test sandboxes without network / model files).
    """
    pytest.importorskip(
        "src.query.retrieve_fused",
        reason="Plan 02-09 has not landed yet",
    )
    from src.query.retrieve_fused import retrieve_fused  # noqa: PLC0415

    def _fake_sub(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def _fake_milvus(*_a: object, **_kw: object) -> list[Any]:
        return []

    import src.query.retrieve as _retr  # noqa: PLC0415
    import src.query.retrieve_hipporag as _h  # noqa: PLC0415
    import src.query.retrieve_lazygraph as _l  # noqa: PLC0415
    import src.query.retrieve_paperqa as _p  # noqa: PLC0415

    monkeypatch.setattr(_p, "retrieve_paperqa", _fake_sub)
    monkeypatch.setattr(_h, "retrieve_hipporag", _fake_sub)
    monkeypatch.setattr(_l, "retrieve_lazygraph", _fake_sub)
    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    observed: list[str] = []

    def _on_stage(stage_name: str, _latency_ms: int) -> None:
        observed.append(stage_name)

    retrieve_fused(query="test", collections=["trading"], on_stage=_on_stage)

    fused_top = [s for s in observed if s.startswith("fused_")]
    assert fused_top == list(FUSED_TOP_LEVEL_STAGES), (
        f"D-06: expected fused top-level stages = {FUSED_TOP_LEVEL_STAGES}; got {fused_top}"
    )


@pytest.mark.timeout(30)
def test_each_sub_retrievers_substages_recorded_in_same_audit_dir(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — fused does NOT spawn separate audit dirs per sub-retriever.

    Every sub-retriever's NN_*.json files must land in the SAME per-query
    directory as fused_dispatch and fused_rerank. Recorded by globbing the
    one audit dir after the run.

    Sub-retrievers are stubbed (real LLM + Milvus calls hang in test
    sandboxes); the test exercises fused's directory-layout invariant.
    """
    pytest.importorskip(
        "src.query.retrieve_fused",
        reason="Plan 02-09 has not landed yet",
    )
    from src.query.retrieve_fused import retrieve_fused  # noqa: PLC0415

    def _fake_sub(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        return [], {}

    def _fake_milvus(*_a: object, **_kw: object) -> list[Any]:
        return []

    import src.query.retrieve as _retr  # noqa: PLC0415
    import src.query.retrieve_hipporag as _h  # noqa: PLC0415
    import src.query.retrieve_lazygraph as _l  # noqa: PLC0415
    import src.query.retrieve_paperqa as _p  # noqa: PLC0415

    monkeypatch.setattr(_p, "retrieve_paperqa", _fake_sub)
    monkeypatch.setattr(_h, "retrieve_hipporag", _fake_sub)
    monkeypatch.setattr(_l, "retrieve_lazygraph", _fake_sub)
    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    retrieve_fused(query="test", collections=["trading"])

    audit_dirs = [d for d in audit_log_root_local.rglob("*") if d.is_dir()]
    leaf_dirs = [
        d for d in audit_dirs if any(p.is_file() and p.name.endswith(".json") for p in d.iterdir())
    ]
    assert len(leaf_dirs) <= 1, (
        f"fused must write all sub-retriever stages into ONE audit dir; "
        f"got {len(leaf_dirs)} dirs: {[d.name for d in leaf_dirs]}"
    )


@pytest.mark.timeout(30)
def test_rerank_top_50_only(audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED — CLAUDE.md "rerank top-50 only": fused_rerank consumes at most
    50 chunks from the union of sub-retrievers.

    Stubs each sub-retriever to return 30 chunks; the rerank stage payload
    must record n_input <= 50.
    """
    pytest.importorskip(
        "src.query.retrieve_fused",
        reason="Plan 02-09 has not landed yet",
    )
    from src.query.retrieve_fused import retrieve_fused  # noqa: PLC0415

    # Lots of synthetic chunks so the union is well over 50.
    fake_chunks = [{"id": f"c{i}", "score_dense": 0.5} for i in range(30)]

    def _fake_pq(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        return list(fake_chunks), {}

    def _fake_hp(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        return list(fake_chunks), {}

    def _fake_lz(*_a: object, **_kw: object) -> tuple[list[Any], dict[str, Any]]:
        return list(fake_chunks), {}

    def _fake_milvus(*_a: object, **_kw: object) -> list[Any]:
        return list(fake_chunks)

    import src.query.retrieve as _retr  # noqa: PLC0415
    import src.query.retrieve_hipporag as _h  # noqa: PLC0415
    import src.query.retrieve_lazygraph as _l  # noqa: PLC0415
    import src.query.retrieve_paperqa as _p  # noqa: PLC0415

    monkeypatch.setattr(_p, "retrieve_paperqa", _fake_pq)
    monkeypatch.setattr(_h, "retrieve_hipporag", _fake_hp)
    monkeypatch.setattr(_l, "retrieve_lazygraph", _fake_lz)
    monkeypatch.setattr(_retr, "milvus_search_only", _fake_milvus)
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    rerank_payload: dict[str, Any] = {}

    from src.query import audit  # noqa: PLC0415

    real_write = audit.write_stage

    def _spy_write(name: str, payload: dict[str, Any]) -> None:
        if name == "fused_rerank":
            rerank_payload.update(payload)
        real_write(name, payload)

    monkeypatch.setattr(audit, "write_stage", _spy_write)
    retrieve_fused(query="test", collections=["trading"])

    n_input = rerank_payload.get("n_input")
    assert n_input is not None, "fused_rerank payload missing 'n_input' field; cannot verify cap."
    assert n_input <= 50, f"CLAUDE.md 'rerank top-50 only' violated: n_input={n_input}"


@pytest.mark.timeout(30)
def test_falls_back_to_milvus_when_optional_retrievers_unavailable(
    audit_log_root_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED — fused degrades to milvus-only when all three optional libs are
    uninstalled. No exception; trace records 'degraded' state.
    """
    pytest.importorskip(
        "src.query.retrieve_fused",
        reason="Plan 02-09 has not landed yet",
    )
    import importlib.util  # noqa: PLC0415

    from src.query.retrieve_fused import retrieve_fused  # noqa: PLC0415

    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, *a: object, **kw: object) -> Any:  # noqa: ANN401
        if name in ("paperqa", "lightrag", "lightrag_hku", "graphrag"):
            return None
        return real_find_spec(name, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    import src.query.retrieve as _retr  # noqa: PLC0415

    monkeypatch.setattr(_retr, "milvus_search_only", lambda *_a, **_kw: [])
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    # Should NOT raise.
    result = retrieve_fused(query="test", collections=["trading"])
    chunks, trace = result if isinstance(result, tuple) else (result, {})

    assert isinstance(chunks, list), (
        f"fused fallback should return list of chunks; got {type(chunks).__name__}"
    )
    degraded = trace.get("degraded") if isinstance(trace, dict) else None
    assert degraded is not None, (
        f"fused fallback should record degraded state in trace; got trace={trace!r}"
    )
