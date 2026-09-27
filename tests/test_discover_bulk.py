"""Tests for the bulk-query mode of scripts.discover_via_deep_research.

Goal: one process invocation can run N Deep Research jobs in parallel, then a
SINGLE `ragctl run --notebooks nb1 nb2 ...` covers all of them — eliminating
the per-notebook MinerU daemon + GPU warmup overhead and the lock contention
that motivated this whole change.

All NotebookLM CLI calls and the ragctl subprocess are stubbed; no network or
real ragctl invocation. We assert on (a) the aggregated result.json schema
and (b) the exact ragctl command that would have fired.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import discover_via_deep_research as drv

# ---------------------------------------------------------------------------
# Stub installer
# ---------------------------------------------------------------------------


class _DiscoverStubs:
    """Bundle of stubs the tests inject onto the script module."""

    def __init__(self) -> None:
        self.created_notebooks: list[str] = []
        self.deleted_notebooks: list[str] = []
        self.add_research_calls: list[tuple[str, str, str]] = []  # (nb_id, query, mode)
        self.research_waits: list[str] = []
        # Per-query plan: nb_title -> {n_imported, n_deduped, n_kept, raises?, hold_s}
        self.per_query_plan: dict[str, dict[str, Any]] = {}
        # Parallel tracker
        self.parallel_now = 0
        self.parallel_max = 0
        self._plock = threading.Lock()
        # ragctl invocation log: list of cmd lists
        self.subprocess_calls: list[list[str]] = []

    # ---- nlm_* stubs ----

    def nlm_create(self, title: str) -> str:
        nb_id = f"nb-{len(self.created_notebooks):03d}"
        self.created_notebooks.append(nb_id)
        # Map back to the query via the title prefix "discover: <query[:80]>".
        plan = self.per_query_plan.get(title, {})
        self.per_query_plan[nb_id] = plan
        return nb_id

    def nlm_add_research(self, nb_id: str, query: str, *, mode: str) -> None:
        self.add_research_calls.append((nb_id, query, mode))
        plan = self.per_query_plan.get(nb_id, {})
        plan["query"] = query
        if plan.get("raises_at") == "add_research":
            raise RuntimeError(plan.get("error_msg", "stubbed DR failure"))

    def nlm_research_wait(self, nb_id: str, *, timeout: int) -> None:
        del timeout
        with self._plock:
            self.parallel_now += 1
            if self.parallel_now > self.parallel_max:
                self.parallel_max = self.parallel_now
        plan = self.per_query_plan.get(nb_id, {})
        time.sleep(plan.get("hold_s", 0.01))
        with self._plock:
            self.parallel_now -= 1
        self.research_waits.append(nb_id)
        if plan.get("raises_at") == "research_wait":
            raise RuntimeError(plan.get("error_msg", "stubbed wait failure"))

    def nlm_source_list(self, nb_id: str) -> list[dict[str, Any]]:
        plan = self.per_query_plan.get(nb_id, {})
        n = int(plan.get("n_imported", 0))
        return [
            {"id": f"src-{nb_id}-{i}", "source_metadata": {"url": f"https://x/{nb_id}/{i}"}}
            for i in range(n)
        ]

    def nlm_source_delete(self, nb_id: str, src_id: str) -> bool:
        del nb_id, src_id
        return True

    def nlm_delete(self, nb_id: str) -> bool:
        self.deleted_notebooks.append(nb_id)
        return True

    # ---- cross-corpus dedup stub ----

    def existingcanonical_urls(self) -> set[str]:
        return set()  # tests don't exercise cross-corpus dedup directly

    def filter_dups(
        self,
        nb_id: str,
        sources: list[dict[str, Any]],
        existing: set[str],
    ) -> tuple[int, int]:
        del existing
        plan = self.per_query_plan.get(nb_id, {})
        n_imp = len(sources)
        n_dedup = int(plan.get("n_deduped", 0))
        n_keep = n_imp - n_dedup
        return n_dedup, n_keep

    # ---- subprocess (ragctl run, ingest) stub ----

    def subprocess_run(self, cmd: list[str], **kwargs: object) -> _DummyCompleted:
        del kwargs
        self.subprocess_calls.append(list(cmd))
        return _DummyCompleted(returncode=0)


class _DummyCompleted:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


@pytest.fixture()
def stubs(monkeypatch: pytest.MonkeyPatch) -> _DiscoverStubs:
    s = _DiscoverStubs()
    monkeypatch.setattr(drv, "nlm_create", s.nlm_create)
    monkeypatch.setattr(drv, "nlm_add_research", s.nlm_add_research)
    monkeypatch.setattr(drv, "nlm_research_wait", s.nlm_research_wait)
    monkeypatch.setattr(drv, "nlm_source_list", s.nlm_source_list)
    monkeypatch.setattr(drv, "nlm_source_delete", s.nlm_source_delete)
    monkeypatch.setattr(drv, "nlm_delete", s.nlm_delete)
    monkeypatch.setattr(drv, "existingcanonical_urls", s.existingcanonical_urls)
    monkeypatch.setattr(drv, "filter_dups_via_nlm_delete", s.filter_dups)
    monkeypatch.setattr(drv.subprocess, "run", s.subprocess_run)
    monkeypatch.delenv("MAX_PARALLEL_DR", raising=False)
    return s


def _plan_for_title(stubs: _DiscoverStubs, query: str, **kwargs: object) -> None:
    """Pre-register an outcome plan keyed by the notebook title the script will use."""
    title = f"discover: {query[:80]}"
    stubs.per_query_plan[title] = kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_query_uses_bulk_path(stubs: _DiscoverStubs) -> None:
    _plan_for_title(stubs, "what is GLFT", n_imported=3, n_deduped=1)
    result = drv.run_discovery_bulk(
        queries=["what is GLFT"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    assert result["status"] == "ok"
    assert result["queries"] == ["what is GLFT"]
    assert len(result["per_query"]) == 1
    pq = result["per_query"][0]
    assert pq["query"] == "what is GLFT"
    assert pq["notebook_id"] == "nb-000"
    assert pq["status"] == "ok"
    assert pq["n_imported"] == 3
    assert pq["n_deduped"] == 1
    assert pq["n_kept"] == 2

    # ragctl run + ingest both fire once.
    ragctl_calls = [c for c in stubs.subprocess_calls if any("ragctl" in tok for tok in c)]
    assert len(ragctl_calls) == 1
    assert "--notebooks" in ragctl_calls[0]
    nb_args_idx = ragctl_calls[0].index("--notebooks") + 1
    # `--notebooks` is followed by exactly one nb_id (then a flag like --stages).
    assert ragctl_calls[0][nb_args_idx] == "nb-000"


def test_three_queries_batch_single_ragctl(stubs: _DiscoverStubs) -> None:
    for q, n in [("a", 2), ("b", 1), ("c", 3)]:
        _plan_for_title(stubs, q, n_imported=n)
    result = drv.run_discovery_bulk(
        queries=["a", "b", "c"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    assert result["status"] == "ok"
    assert len(result["per_query"]) == 3
    nb_ids = [pq["notebook_id"] for pq in result["per_query"]]
    assert sorted(nb_ids) == ["nb-000", "nb-001", "nb-002"]

    # Sum aggregation.
    assert result["n_imported"] == 6
    assert result["n_kept"] == 6  # zero deduped

    # Exactly one ragctl run invocation; it lists ALL three notebooks.
    ragctl_calls = [c for c in stubs.subprocess_calls if any("ragctl" in p for p in c)]
    assert len(ragctl_calls) == 1
    args = ragctl_calls[0]
    nb_idx = args.index("--notebooks") + 1
    # `--notebooks` consumes 1+ args until the next flag (--stages).
    nb_args: list[str] = []
    for tok in args[nb_idx:]:
        if tok.startswith("--"):
            break
        nb_args.append(tok)
    assert sorted(nb_args) == ["nb-000", "nb-001", "nb-002"]


def test_one_query_errors_others_still_ingested(stubs: _DiscoverStubs) -> None:
    _plan_for_title(stubs, "ok-1", n_imported=2)
    _plan_for_title(stubs, "broken", raises_at="research_wait", error_msg="boom")
    _plan_for_title(stubs, "ok-2", n_imported=1)

    result = drv.run_discovery_bulk(
        queries=["ok-1", "broken", "ok-2"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    # Top-level still ok — at least one query succeeded.
    assert result["status"] == "ok"
    statuses = {pq["query"]: pq["status"] for pq in result["per_query"]}
    assert statuses == {"ok-1": "ok", "broken": "error", "ok-2": "ok"}
    error_pq = next(pq for pq in result["per_query"] if pq["query"] == "broken")
    assert "boom" in (error_pq["error"] or "")

    # ragctl batched the two successful notebooks only.
    ragctl_calls = [c for c in stubs.subprocess_calls if any("ragctl" in p for p in c)]
    assert len(ragctl_calls) == 1
    args = ragctl_calls[0]
    nb_idx = args.index("--notebooks") + 1
    nb_args: list[str] = []
    for tok in args[nb_idx:]:
        if tok.startswith("--"):
            break
        nb_args.append(tok)
    assert len(nb_args) == 2


def test_all_no_new_sources_skips_ragctl(stubs: _DiscoverStubs) -> None:
    for q in ("a", "b"):
        _plan_for_title(stubs, q, n_imported=2, n_deduped=2)  # all deduped → 0 kept

    result = drv.run_discovery_bulk(
        queries=["a", "b"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    assert result["status"] == "no_new_sources"
    assert all(pq["status"] == "no_new_sources" for pq in result["per_query"])

    ragctl_calls = [c for c in stubs.subprocess_calls if any("ragctl" in p for p in c)]
    assert ragctl_calls == []


def test_result_schema_keys(stubs: _DiscoverStubs) -> None:
    _plan_for_title(stubs, "schema-q", n_imported=1)
    result = drv.run_discovery_bulk(
        queries=["schema-q"],
        collection="trading",
        partition="research_briefs",
        mode="fast",
        timeout=60,
        keep=False,
    )
    top_keys = {
        "status",
        "elapsed_s",
        "collection",
        "partition",
        "mode",
        "queries",
        "per_query",
        "n_imported",
        "n_deduped",
        "n_kept",
    }
    assert top_keys.issubset(set(result.keys()))
    assert isinstance(result["queries"], list)
    assert isinstance(result["per_query"], list)
    pq = result["per_query"][0]
    pq_keys = {
        "query",
        "notebook_id",
        "notebook_kept",
        "status",
        "n_imported",
        "n_deduped",
        "n_kept",
    }
    assert pq_keys.issubset(set(pq.keys()))
    assert "query" not in result  # hard-cut from old schema


def test_max_parallel_dr_respected(monkeypatch: pytest.MonkeyPatch, stubs: _DiscoverStubs) -> None:
    monkeypatch.setenv("MAX_PARALLEL_DR", "2")
    for q in ("a", "b", "c", "d"):
        _plan_for_title(stubs, q, n_imported=1, hold_s=0.15)

    drv.run_discovery_bulk(
        queries=["a", "b", "c", "d"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    assert stubs.parallel_max <= 2
    # And concurrency was actually exercised (not all serial).
    assert stubs.parallel_max >= 2


def test_result_json_written_atomically(
    tmp_path: Path, stubs: _DiscoverStubs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI main writes the aggregated result to --result-json."""
    _plan_for_title(stubs, "q1", n_imported=1)
    result_path = tmp_path / "out.result.json"
    monkeypatch.setattr(
        os.sys,
        "argv",
        [
            "prog",
            "--query",
            "q1",
            "--collection",
            "trading",
            "--partition",
            "research_briefs",
            "--mode",
            "deep",
            "--timeout",
            "60",
            "--result-json",
            str(result_path),
        ],
    )
    rc = drv.main()
    assert rc == 0
    assert result_path.is_file()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["queries"] == ["q1"]


def test_cli_accepts_multiple_queries(
    tmp_path: Path, stubs: _DiscoverStubs, monkeypatch: pytest.MonkeyPatch
) -> None:
    for q in ("alpha", "beta"):
        _plan_for_title(stubs, q, n_imported=1)
    result_path = tmp_path / "bulk.result.json"
    monkeypatch.setattr(
        os.sys,
        "argv",
        [
            "prog",
            "--query",
            "alpha",
            "beta",
            "--collection",
            "trading",
            "--partition",
            "research_briefs",
            "--mode",
            "deep",
            "--timeout",
            "60",
            "--result-json",
            str(result_path),
        ],
    )
    rc = drv.main()
    assert rc == 0
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["queries"] == ["alpha", "beta"]
    assert len(data["per_query"]) == 2


# ---------------------------------------------------------------------------
# #36: stuck query must not block the batch beyond `bulk_deadline_s`.
# ---------------------------------------------------------------------------


def test_bulk_abandons_query_exceeding_bulk_deadline(stubs: _DiscoverStubs) -> None:
    """A query slower than ``bulk_deadline_s`` is marked abandoned, the rest
    proceed to fetch+parse. Regression for #36 (successful Deep Research
    sources blocked by single stuck IMPORT_RESEARCH retry)."""
    _plan_for_title(stubs, "fast1", n_imported=1, hold_s=0.05)
    _plan_for_title(stubs, "fast2", n_imported=2, hold_s=0.05)
    _plan_for_title(stubs, "slow", n_imported=99, hold_s=8.0)

    result = drv.run_discovery_bulk(
        queries=["fast1", "fast2", "slow"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
        bulk_deadline_s=1.0,
    )

    statuses = sorted(pq["status"] for pq in result["per_query"])
    assert statuses == ["abandoned", "ok", "ok"], statuses

    abandoned = [pq for pq in result["per_query"] if pq["status"] == "abandoned"][0]
    assert abandoned["query"] == "slow"
    assert "bulk deadline" in (abandoned["error"] or "")
    assert abandoned["n_imported"] == 0

    ragctl_calls = [c for c in stubs.subprocess_calls if any("ragctl" in t for t in c)]
    assert len(ragctl_calls) == 1, "fetch+parse must fire for the succeeded notebooks"
    idx = ragctl_calls[0].index("--notebooks")
    nb_ids_in_cmd = []
    for tok in ragctl_calls[0][idx + 1 :]:
        if tok.startswith("--"):
            break
        nb_ids_in_cmd.append(tok)
    # nb-id assignment order is racy across the three threads; compare via
    # the per_query mapping instead of hard-coded ids.
    expected_nb_ids = {pq["notebook_id"] for pq in result["per_query"] if pq["status"] == "ok"}
    assert set(nb_ids_in_cmd) == expected_nb_ids
    assert len(expected_nb_ids) == 2


def test_bulk_no_deadline_keeps_legacy_behaviour(stubs: _DiscoverStubs) -> None:
    """With ``bulk_deadline_s=None`` (default), all queries are awaited."""
    for q in ("a", "b"):
        _plan_for_title(stubs, q, n_imported=1, hold_s=0.05)
    result = drv.run_discovery_bulk(
        queries=["a", "b"],
        collection="trading",
        mode="deep",
        timeout=60,
        keep=False,
    )
    assert result["status"] == "ok"
    statuses = {pq["status"] for pq in result["per_query"]}
    assert statuses == {"ok"}, statuses
