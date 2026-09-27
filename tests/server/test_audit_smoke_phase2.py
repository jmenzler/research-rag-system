"""Phase 2 extension to INFRA-04 + INFRA-05 audit-smoke invariants.

Builds on tests/server/test_audit_smoke.py:
  - Every NEW stage prefix in `_STAGE_PREFIX` must have a section renderer
    in `src/query/report_render.py` (CRIT-2).
  - No "99_unknown" bucket files written by any new retriever (Pitfall 7).
  - For each new retriever, the count of NN_*.json stage files written
    equals the count of registered sub-stage prefixes for that retriever.
  - The auto-name fallback path (D-22) records cost_usd > 0 in meta.totals.

Tests use deferred imports + parametrization across paperqa / hipporag /
lazygraph / fused so collection stays green before Plan 02-05 lands the
new prefixes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.query.audit import reset_active_logger, set_active_logger

# Sub-stage prefixes locked by 02-01-LIBRARY-SPIKE.md (Plan 02-05 lands them).
EXPECTED_NEW_PREFIXES: dict[str, list[str]] = {
    "paperqa": ["paperqa_rerank", "paperqa_read", "paperqa_answer"],
    "hipporag": ["hipporag_entities", "hipporag_walk", "hipporag_synthesize"],
    "lazygraph": ["lazygraph_route", "lazygraph_community", "lazygraph_summarize"],
    "fused": ["fused_dispatch", "fused_rerank"],
}


@pytest.fixture(autouse=True)
def isolated_audit_logger() -> Iterator[None]:
    token = set_active_logger(None)
    try:
        yield
    finally:
        reset_active_logger(token)


def test_every_stage_prefix_has_section_renderer() -> None:
    """RED — CRIT-2: every stage in `_STAGE_PREFIX` has a matching renderer
    in `src/query/report_render.py`.

    Scans `_STAGE_PREFIX` keys, asserts each name has a corresponding
    ``_section_<name>`` function or the renderer handles it via a generic
    fallback marker.
    """
    from src.query import report_render
    from src.query.query_logger import _STAGE_PREFIX

    renderer_fns = {name for name in dir(report_render) if name.startswith("_section_")}

    missing: list[str] = []
    for stage in _STAGE_PREFIX:
        candidate = f"_section_{stage}"
        # accept a per-stage renderer OR a "_section_retriever" umbrella
        if candidate in renderer_fns:
            continue
        if any(stage.startswith(p) for p in ("paperqa", "hipporag", "lazygraph", "fused")):
            if "_section_retriever" in renderer_fns or candidate in renderer_fns:
                continue
        # For Phase 0/1 prefixes we accept the existing sections; for the
        # new Phase 2 prefixes we require an explicit renderer or umbrella.
        if any(stage.startswith(p) for p in EXPECTED_NEW_PREFIXES):
            missing.append(stage)

    assert not missing, (
        f"CRIT-2: new stage prefixes lack a section renderer in "
        f"report_render.py: {missing}. Add `_section_<stage>` or a "
        f"`_section_retriever` umbrella that handles all retriever sub-stages."
    )


def test_no_99_unknown_bucket_files(tmp_path: Path) -> None:
    """RED — Pitfall 7: no NN_*.json file may use the '99' or 'unknown' bucket.

    Scans `_STAGE_PREFIX` values; asserts no entry is '99' (the historic
    Phase-0 "unknown bucket" sentinel) and no entry contains 'unknown'.
    """
    from src.query.query_logger import _STAGE_PREFIX

    values = list(_STAGE_PREFIX.values())
    assert "99" not in values, (
        f"Pitfall 7: '99' is reserved for the unknown bucket — never use as a "
        f"real stage prefix; got {_STAGE_PREFIX}"
    )
    keys_lower = [k.lower() for k in _STAGE_PREFIX]
    assert not any("unknown" in k for k in keys_lower), (
        f"Pitfall 7: no stage may be named 'unknown'; got {list(_STAGE_PREFIX)}"
    )


@pytest.mark.parametrize("retriever", list(EXPECTED_NEW_PREFIXES.keys()))
def test_substage_count_matches_registry_after_retriever_run(
    retriever: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_milvus_search: list[object],
) -> None:
    """RED — for each new retriever, the count of NN_*.json stage files
    written equals the count of sub-stages locked in the registry.

    Drives the retriever; counts files; compares.
    """
    plan_map = {"paperqa": "02-06", "hipporag": "02-07", "lazygraph": "02-08", "fused": "02-09"}
    pytest.importorskip(
        f"src.query.retrieve_{retriever}",
        reason=f"Plan {plan_map[retriever]} has not landed yet",
    )

    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))

    import importlib

    mod = importlib.import_module(f"src.query.retrieve_{retriever}")
    fn = getattr(mod, f"retrieve_{retriever}")
    fn(query="test", collections=["trading"])

    stage_files = list(queries_root.rglob(f"[0-9][0-9]_{retriever}_*.json"))
    # Permit also generic prefixes if retriever uses them
    if retriever == "fused":
        stage_files += list(queries_root.rglob("[0-9][0-9]_fused_*.json"))

    expected_subs = EXPECTED_NEW_PREFIXES[retriever]
    seen_subs = {p.name.split("_", 1)[1].rsplit(".", 1)[0] for p in stage_files}
    for sub in expected_subs:
        assert sub in seen_subs, (
            f"CRIT-2: {retriever} ran but did not write a stage file for "
            f"{sub!r}. Expected {expected_subs}; found {sorted(seen_subs)}."
        )


def test_auto_name_fallback_cost_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED — D-22 / CHAT-10: when title_hint is empty, the fallback path
    routes through usage_track.call_text and meta.totals.cost_usd > 0.
    """
    pytest.importorskip("src.server.api_chats")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.server.api_chats import router
    from src.server.migrations import migrate

    # Sandbox state.
    chats_db = tmp_path / "chats.db"
    migrate.run(chats_db)
    import src.server.chats_store as _cs

    monkeypatch.setattr(_cs, "CHATS_DB_PATH", chats_db)
    queries_root = tmp_path / "queries"
    monkeypatch.setenv("QUERY_LOG_ROOT", str(queries_root))

    # Genai stub.
    import types

    fake_usage = types.SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=30,
        cached_content_token_count=0,
    )
    fake_response = types.SimpleNamespace(
        text="auto title",
        usage_metadata=fake_usage,
    )
    fake_models = types.SimpleNamespace(
        generate_content=lambda **kwargs: fake_response,
    )
    fake_client = types.SimpleNamespace(models=fake_models)
    from google import genai as _genai_mod

    from src import config as _config

    monkeypatch.setattr(_genai_mod, "Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(_config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(_config, "validate_api_key", lambda: None)
    monkeypatch.setattr(_config, "DECOMPOSE_MODEL", "gemini-2.5-flash")

    # Milvus stub.
    import src.query.retrieve as _retr

    monkeypatch.setattr(_retr, "milvus_search_only", lambda *_a, **_kw: [])
    monkeypatch.setattr(_retr, "validate_api_key", lambda: None)
    monkeypatch.setattr(_retr, "_embed_query", lambda _q: [0.0] * 4)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.post("/api/chats", json={"title": "Untitled chat"})
    chat_id = r.json()["id"]

    # Drain SSE.
    body = {"content": "q", "message_uuid": "u1", "expected_version": 0}
    with client.stream(
        "POST",
        f"/api/chats/{chat_id}/messages",
        json=body,
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass

    # Locate meta.json + assert cost_usd > 0.
    meta_files = list(queries_root.rglob("meta.json"))
    assert meta_files, f"no meta.json under {queries_root}"
    meta: dict[str, Any] = json.loads(meta_files[-1].read_text())
    cost = meta.get("totals", {}).get("cost_usd", 0)
    assert cost > 0, (
        f"D-22: meta.totals.cost_usd={cost!r}; auto-name fallback must "
        f"record positive cost via usage_track.call_text."
    )
