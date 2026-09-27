"""End-to-end test: real ragctl run parsing real PDFs through the warm daemon.

This is the test that catches the regression we're fixing. It asserts the
key invariant: across multiple PDFs in one orchestrator run, MinerU's
DocAnalysis backend initializes exactly ONCE.

Marked @pytest.mark.e2e because:
  - Requires .venv-mineru installed (~3 GB models)
  - Takes 1-3 minutes per run
  - Touches GPU on Linux/CUDA, MPS on macOS

Skipped by default; run explicitly with `pytest -m e2e`.

Strategy:
  1. Build 2 copies of tests/fixtures/tiny.pdf under sources/<collection>/<slug>/
     in a tmp working tree.
  2. Run `bin/ragctl run --collection trading --stages parse_pipeline`.
  3. Assert: pdf.txt files exist, marker files cleaned, daemon state file gone,
     and `DocAnalysis init done!` appears exactly once across the captured logs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TINY_PDF = REPO / "tests" / "fixtures" / "tiny.pdf"
RAGCTL = REPO / "bin" / "ragctl"
MINERU_API_BIN = REPO / ".venv-mineru" / "bin" / "mineru-api"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not MINERU_API_BIN.exists(), reason=".venv-mineru not installed"),
    pytest.mark.skipif(not TINY_PDF.exists(), reason="tests/fixtures/tiny.pdf missing"),
]


def _make_collection_docs(sources_dir: Path, collection: str, num_copies: int) -> list[Path]:
    """Create sources/<collection>/<slug>/source.pdf for ``num_copies`` distinct
    slugs. Returns the list of PDF paths.
    """
    coll_dir = sources_dir / collection
    coll_dir.mkdir(parents=True)
    pdfs = []
    for i in range(num_copies):
        doc_dir = coll_dir / f"fixture_doc_{i}"
        doc_dir.mkdir()
        pdf = doc_dir / "source.pdf"
        shutil.copy(TINY_PDF, pdf)
        pdfs.append(pdf)
    return pdfs


@pytest.fixture()
def staged_repo(tmp_path: Path) -> Path:
    """Make a working copy of the repo layout pointing at tmp sources/.

    Symlinks ``.venv-mineru``, ``src/``, ``bin/``, ``prompts/`` from the real
    repo; creates an empty ``sources/`` and ``logs/`` in tmp.
    """
    tmp_repo = tmp_path / "repo"
    tmp_repo.mkdir()
    for name in (".venv-mineru", ".venv", "src", "scripts", "bin", "prompts", "pyproject.toml"):
        src = REPO / name
        if src.exists():
            (tmp_repo / name).symlink_to(src)
    (tmp_repo / "logs").mkdir()
    (tmp_repo / "sources").mkdir()
    return tmp_repo


def test_warm_start_across_multiple_pdfs(staged_repo: Path) -> None:
    """The crown-jewel regression test.

    Two PDFs in one collection → DocAnalysis init done! must appear exactly
    once in the captured logs."""
    collection = "trading"
    _make_collection_docs(staged_repo / "sources", collection, num_copies=2)

    env = os.environ.copy()
    log_path = staged_repo / "ragctl_run.log"
    with log_path.open("w") as log:
        rc = subprocess.run(
            [
                sys.executable,
                str(RAGCTL),
                "run",
                "--collection",
                collection,
                "--stages",
                "parse_pipeline",
                "--parallel",
                "1",
            ],
            cwd=str(staged_repo),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        ).returncode

    log_text = log_path.read_text()
    if rc != 0:
        pytest.fail(f"ragctl run failed (rc={rc}). Log tail:\n{log_text[-3000:]}")

    init_count = log_text.count("DocAnalysis init done!")
    assert init_count == 1, (
        f"Expected DocAnalysis to init exactly once across both PDFs; "
        f"got {init_count}. Daemon is not warm. Log tail:\n{log_text[-2000:]}"
    )

    for pdf_dir in (staged_repo / "sources" / collection).iterdir():
        assert (pdf_dir / "pdf.txt").exists(), f"missing pdf.txt in {pdf_dir.name}"
        assert not (pdf_dir / "pdf.needs_vlm").exists(), f"stale marker in {pdf_dir.name}"


def test_daemon_state_file_cleaned_after_run(staged_repo: Path) -> None:
    """The state file at ~/.cache/rag-system/mineru-pipeline.json (or
    $RAG_MINERU_STATE override) must be absent once ragctl exits cleanly."""
    collection = "trading"
    _make_collection_docs(staged_repo / "sources", collection, num_copies=1)

    state_path = staged_repo / "mineru-pipeline.json"
    env = os.environ.copy()
    env["RAG_MINERU_STATE"] = str(state_path)

    rc = subprocess.run(
        [
            sys.executable,
            str(RAGCTL),
            "run",
            "--collection",
            collection,
            "--stages",
            "parse_pipeline",
            "--parallel",
            "1",
        ],
        cwd=str(staged_repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ).returncode

    assert rc == 0
    assert not state_path.exists(), f"daemon state file leaked: {state_path}"
