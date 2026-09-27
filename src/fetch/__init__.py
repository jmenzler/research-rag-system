"""Fetch package: source resolution, classification, and the Spider runner.

Submodules expose the public API directly — callers import what they need
(e.g. ``from src.fetch.spider import FetchPipelineSpider``). The package
namespace itself stays empty so importing ``src.fetch`` doesn't drag in
scrapling[fetchers] -> playwright -> browserforge.
"""

from __future__ import annotations
