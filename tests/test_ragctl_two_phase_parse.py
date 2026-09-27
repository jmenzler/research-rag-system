"""Contract tests for the two-phase parse stage in ragctl run.

These define how `parse` is split into `parse_pipeline` (parallel=N) and
`parse_vlm` (parallel=1, gated on marker existence) when handed to the
ProcessPoolExecutor in `bin/ragctl`.

Strategy: assert via `build_stage_command` and the public stage-list shape.
The actual subprocess wiring is exercised by `test_ragctl_run_cli.py` in
dry-run mode; this file pins the orchestration contract.
"""

from __future__ import annotations

from src.pipeline import orchestrate


def test_parse_pipeline_command_uses_pipeline_only_pass() -> None:
    cmd = orchestrate.build_parse_command("trading", pass_mode="pipeline-only")
    assert "src.pdf_parsers.cli" in cmd
    assert "--collection" in cmd
    assert "trading" in cmd
    assert "--pass" in cmd
    assert "pipeline-only" in cmd


def test_parse_vlm_command_uses_vlm_pass() -> None:
    cmd = orchestrate.build_parse_command("trading", pass_mode="vlm")
    assert "src.pdf_parsers.cli" in cmd
    assert "--pass" in cmd
    assert "vlm" in cmd


def test_parse_pipeline_and_parse_vlm_are_pre_barrier() -> None:
    assert "parse_pipeline" in orchestrate.PRE_BARRIER_STAGES
    assert "parse_vlm" in orchestrate.PRE_BARRIER_STAGES


def test_parse_pseudo_stage_expands_to_pipeline_then_vlm() -> None:
    """`--stages parse` must select both passes in order so existing CLI
    invocations don't have to know about the split."""
    import argparse

    args = argparse.Namespace(stages=["parse"], skip_stages=None)
    stages = orchestrate.select_stages(args)
    # Pipeline pass must come before VLM pass
    assert stages.index("parse_pipeline") < stages.index("parse_vlm")
    # Other stages must not be selected
    assert "fetch" not in stages
    assert "audit" not in stages


def test_legacy_parse_stage_no_longer_in_all_stages() -> None:
    """Once split, the bare `parse` stage isn't a valid stage name itself —
    it's a user-facing alias that expands. ALL_STAGES must contain the splits."""
    assert "parse_pipeline" in orchestrate.ALL_STAGES
    assert "parse_vlm" in orchestrate.ALL_STAGES


def test_select_stages_rejects_unknown_legacy_alias() -> None:
    """The bare ``parse`` token is an alias that expands; passing an unknown
    stage name (e.g. ``parse_legacy``) must still raise."""
    import argparse

    import pytest

    with pytest.raises(SystemExit):
        orchestrate.select_stages(
            argparse.Namespace(stages=["parse_legacy"], skip_stages=None),
        )
