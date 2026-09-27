"""NotebookLM fulltext download.

Single helper invoked from ``FetchPipelineSpider.on_scraped_item`` to grab the
NotebookLM-hosted fulltext for a source.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def fetch_nlm_fulltext(
    notebook_id: str, src_id: str, out_path: Path,
) -> int | None:
    """Download a NotebookLM source's fulltext via the ``notebooklm`` CLI.

    Returns the number of bytes written, or ``None`` if the CLI failed,
    timed out, or wrote an empty file (the empty file is removed in that case).
    """
    try:
        result = subprocess.run(
            ["notebooklm", "source", "fulltext", "--notebook", notebook_id,
             src_id, "-o", str(out_path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not out_path.exists():
        return None
    n = out_path.stat().st_size
    if n == 0:
        out_path.unlink()
        return None
    return n
