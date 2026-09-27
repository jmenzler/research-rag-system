"""API keys and connection URIs loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=_HERE / ".env", override=False)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")


def validate_api_key() -> None:
    """Raise ``RuntimeError`` if ``GEMINI_API_KEY`` is not set.

    Call this at the entry point of any module that makes Gemini API calls.
    Do NOT call it at module import time — that would break milvus-only tests.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Export it in your shell or add it to the .env file at the project root."
        )
