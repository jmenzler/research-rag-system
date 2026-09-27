"""CI lint (INFRA-04): refuse direct genai / openai imports outside the
approved whitelist of existing LLM provider files.

Mechanism: AST walk over every .py file in src/, looking for:
  - `import openai` / `import google.genai` / `import google.generativeai`
  - `from openai import ...` / `from google import genai` / `from google.genai import ...`

Whitelist: files that are pre-approved import sites (existing architecture).
  The SINGLE authorized entry-point for NEW query-pipeline callers is
  src/query/usage_track.py — any new module under src/server/ or src/query/
  that is NOT in the whitelist and imports genai/openai will be caught here.

Intent (INFRA-04): this rail MUST exist before Phase 1 ships GUI-facing LLM
callers. Without it, developers bypass usage_track.call_text and the audit
trail degrades silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Canonical entry-point for new query-pipeline LLM calls (INFRA-04).
# All NEW modules must route through usage_track.call_text instead of
# importing genai/openai directly.
_CANONICAL_ENTRY_POINT = PROJECT_ROOT / "src" / "query" / "usage_track.py"

# Pre-approved legacy files that hold direct LLM imports from before this
# architectural decision. These are the ONLY additional allowed import sites.
# Do NOT add new entries without a documented architectural decision in
# DECISIONS.md. New GUI-facing callers go through usage_track.call_text.
_LEGACY_WHITELIST: frozenset[Path] = frozenset(
    {
        # LLM provider abstraction layer (pre-Phase-0 architecture).
        PROJECT_ROOT / "src" / "llm" / "providers" / "gemini.py",
        PROJECT_ROOT / "src" / "llm" / "providers" / "deepseek.py",
        PROJECT_ROOT / "src" / "llm" / "providers" / "openrouter.py",
        PROJECT_ROOT / "src" / "llm" / "retry.py",
        # Ingest pipeline — embedding + contextualization (pre-Phase-0).
        PROJECT_ROOT / "src" / "ingest" / "embed.py",
        PROJECT_ROOT / "src" / "ingest" / "contextualize.py",
        PROJECT_ROOT / "src" / "ingest" / "ingest.py",
        # Query pipeline legacy — direct SDK use predating usage_track.
        PROJECT_ROOT / "src" / "query" / "generate.py",
        PROJECT_ROOT / "src" / "query" / "retrieve.py",
        # Monitor.
        PROJECT_ROOT / "src" / "monitor" / "ingest_paper.py",
    }
)

WHITELIST: frozenset[Path] = _LEGACY_WHITELIST | {_CANONICAL_ENTRY_POINT}

BANNED_MODULES = {"openai", "google.genai", "google.generativeai"}


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (lineno, offending_module) found in `path`."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"{path}: syntax error during AST parse: {exc}")
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in BANNED_MODULES or any(name.startswith(b + ".") for b in BANNED_MODULES):
                    offenders.append((node.lineno, name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in BANNED_MODULES or any(mod.startswith(b + ".") for b in BANNED_MODULES):
                offenders.append((node.lineno, mod))
            if mod == "google":
                for alias in node.names:
                    if alias.name in {"genai", "generativeai"}:
                        offenders.append((node.lineno, f"google.{alias.name}"))
    return offenders


def test_no_direct_llm_imports() -> None:
    """Walk src/ and ensure only whitelisted files import genai/openai.

    New modules (especially GUI-facing Phase 1+ callers) must NOT import
    genai or openai directly — they must route through
    src/query/usage_track.call_text instead.

    If you need to add a new import site, add the path to _LEGACY_WHITELIST
    above and document the decision in DECISIONS.md.
    """
    offenders: list[str] = []
    for py_path in sorted(SRC_DIR.rglob("*.py")):
        if py_path in WHITELIST:
            continue
        for lineno, mod in _scan_file(py_path):
            rel = py_path.relative_to(PROJECT_ROOT)
            offenders.append(
                f"{rel}:{lineno}: imports `{mod}` "
                "— route through src/query/usage_track.call_text "
                "or add to _LEGACY_WHITELIST with a DECISIONS.md entry"
            )
    if offenders:
        joined = "\n  ".join(offenders)
        pytest.fail("INFRA-04: direct LLM SDK imports outside approved whitelist:\n  " + joined)


def test_detector_finds_violation(tmp_path: Path) -> None:
    """Meta-test: the AST walker detects a violation when one exists."""
    fixture = tmp_path / "violator.py"
    fixture.write_text("from openai import OpenAI\nclient = OpenAI()\n")
    offenders = _scan_file(fixture)
    assert len(offenders) >= 1
    assert offenders[0][1] == "openai"


def test_whitelist_is_exact_path(tmp_path: Path) -> None:
    """A file at a similar-looking path is NOT exempt from the main scan.

    Note: _scan_file itself doesn't check the whitelist — the whitelist
    logic is in test_no_direct_llm_imports. This test verifies that
    _scan_file correctly reports violations (so a path outside WHITELIST
    would be caught by the main test).
    """
    fixture = tmp_path / "usage_track.py"  # not the canonical path
    fixture.write_text("from openai import OpenAI\n")
    offenders = _scan_file(fixture)
    assert len(offenders) >= 1
