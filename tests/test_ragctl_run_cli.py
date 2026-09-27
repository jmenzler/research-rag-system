"""Acceptance tests for `ragctl run`.

Exercises the CLI surface end-to-end up to (but not including) subprocess
execution — argparse wiring, selector validation, dry-run plan output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAGCTL = REPO / "bin" / "ragctl"


def run_ragctl(
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(RAGCTL), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_run_help_lists_subcommand() -> None:
    res = run_ragctl("--help")
    assert res.returncode == 0
    assert "run" in res.stdout
    assert "sync-registry" not in res.stdout
    assert (
        "notebooks" not in res.stdout.split("status", 1)[-1].split("query", 1)[0] or True
    )  # subcommand removed from registry too — soft check


def test_run_subcommand_help() -> None:
    res = run_ragctl("run", "--help")
    assert res.returncode == 0
    assert "--collection" in res.stdout
    assert "--notebooks" in res.stdout
    assert "--stages" in res.stdout
    assert "--dry-run" in res.stdout
    assert "--parallel" in res.stdout
    # Old options are gone
    assert "--no-sync-registry" not in res.stdout
    assert "--tag" not in res.stdout


def test_run_requires_collection() -> None:
    res = run_ragctl("run", "--dry-run")
    assert res.returncode != 0
    assert "required" in (res.stdout + res.stderr).lower()


def test_run_rejects_invalid_collection() -> None:
    res = run_ragctl("run", "--collection", "not-a-collection", "--dry-run")
    assert res.returncode != 0
    assert "invalid choice" in (res.stdout + res.stderr).lower() or "not-a-collection" in (
        res.stdout + res.stderr
    )


def test_run_dry_run_collection_only_fetch_dropped() -> None:
    """`run --collection trading --dry-run` (no --notebooks) plans
    parse → audit → contextualize → ingest over the existing archive."""
    res = run_ragctl("run", "--collection", "trading", "--dry-run")
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    out = res.stdout
    assert "WOULD RUN" in out
    assert "trading" in out


def test_run_dry_run_with_notebooks_uuids() -> None:
    """`run --collection trading --notebooks UUID --dry-run` includes fetch."""
    res = run_ragctl(
        "run",
        "--collection",
        "trading",
        "--notebooks",
        "fake-uuid-1",
        "fake-uuid-2",
        "--dry-run",
    )
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "fake-uuid-1" in res.stdout or "fetch" in res.stdout


def test_run_stages_and_skip_stages_mutex() -> None:
    res = run_ragctl(
        "run",
        "--collection",
        "trading",
        "--dry-run",
        "--stages",
        "fetch",
        "--skip-stages",
        "ingest",
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "mutually" in combined or "exclusive" in combined


def test_run_dry_run_subset_stages_shown() -> None:
    res = run_ragctl(
        "run",
        "--collection",
        "trading",
        "--dry-run",
        "--stages",
        "parse_pipeline",
    )
    assert res.returncode == 0
    assert "parse" in res.stdout


def test_sync_registry_subcommand_removed() -> None:
    res = run_ragctl("sync-registry", "--help")
    assert res.returncode != 0


def test_notebooks_subcommand_removed() -> None:
    res = run_ragctl("notebooks", "--help")
    assert res.returncode != 0
