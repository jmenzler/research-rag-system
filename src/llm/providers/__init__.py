"""Provider registry: maps provider name → generate function."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.messages import LLMResponse

# Late-import to avoid circular deps and keep cold imports fast.
# Each key maps to a module-level _<provider>_generate function.

_PROVIDERS: dict[str, Callable[..., LLMResponse]] = {}


def _ensure_registry() -> None:
    """Populate the provider registry on first access (lazy)."""
    if _PROVIDERS:
        return
    from src.llm.providers.deepseek import _deepseek_generate
    from src.llm.providers.gemini import _gemini_generate
    from src.llm.providers.openrouter import _openrouter_generate

    _PROVIDERS["gemini"] = _gemini_generate
    _PROVIDERS["deepseek"] = _deepseek_generate
    _PROVIDERS["openrouter"] = _openrouter_generate


def get_provider(name: str) -> Callable[..., LLMResponse]:
    """Return the generate callable for *name*.

    Raises ``ValueError`` for unknown provider names.
    """
    _ensure_registry()
    fn = _PROVIDERS.get(name)
    if fn is None:
        raise ValueError(f"Unknown provider {name!r}. Known: {sorted(_PROVIDERS.keys())}")
    return fn
