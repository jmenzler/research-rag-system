# long-ok-file
"""Embedding clients for the ingest pipeline.

Two text providers, dispatched on ``config.EMBEDDING_PROVIDER``:

  * ``gemini:gemini-embedding-2`` — uses the genai SDK's ``embed_content``
    with task_type=RETRIEVAL_DOCUMENT and a 50-text batch.
  * ``openrouter:*`` — uses the OpenAI-compat ``/v1/embeddings`` endpoint
    via raw httpx (skipping the openai SDK's Pydantic parsing, which
    burns ~80% of wall time on dim=4096 batch=50 responses).

One image path:

  * ``_embed_image`` — Gemini multimodal embedding (image → same vector
    space as text, native MRL truncation).

Rate-limit handling: ``_embed_batch_with_retry`` wraps the Gemini call
with tenacity exponential backoff on 429. The OpenRouter path raises
``HTTPError`` directly; OpenRouter's own retry semantics handle most
transients.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src import config

if TYPE_CHECKING:
    import httpx as _httpx_typed


# ---------------------------------------------------------------------------
# MIME-type lookup (shared with the per-file dispatchers in ingest.py)
# ---------------------------------------------------------------------------

_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


# ---------------------------------------------------------------------------
# Rate-limit retry wrapper (Gemini path only)
# ---------------------------------------------------------------------------


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if *exc* is a Gemini 429 rate-limit ClientError."""
    return isinstance(exc, ClientError) and getattr(exc, "code", None) == 429


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch_with_retry(
    client: genai.Client,
    batch: list[types.Content],
) -> list[list[float]]:
    """Embed one batch of Content objects, retrying on 429 with exponential backoff."""
    response = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=cast(types.ContentListUnion, batch),
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=config.EMBED_DIM,
        ),
    )
    embeddings = response.embeddings
    if embeddings is None:
        raise RuntimeError("Gemini embed_content returned no embeddings for batch")
    vectors: list[list[float]] = []
    for emb in embeddings:
        values = emb.values
        if values is None:
            raise RuntimeError("Gemini embedding entry has no values")
        vectors.append(list(values))
    if len(vectors) != len(batch):
        raise RuntimeError(
            f"Gemini returned {len(vectors)} embeddings for {len(batch)} inputs"
        )
    return vectors


# ---------------------------------------------------------------------------
# Public batch text embed — dispatches on EMBEDDING_PROVIDER prefix
# ---------------------------------------------------------------------------


def _embed_text_batch(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Embed a batch of text strings, dispatching on EMBEDDING_PROVIDER.

    Returns a list of `EMBED_DIM`-dimensional float vectors, one per input.
    For "gemini": uses Gemini Embedding 2 via the genai client passed in.
    For "openrouter:*": uses the OpenAI-compat embeddings endpoint at
    OpenRouter, ignoring the genai client (we keep the param for signature
    compatibility with existing call sites).
    """
    if config.EMBEDDING_PROVIDER.startswith("gemini:"):
        return _embed_text_batch_gemini(client, texts)
    if config.EMBEDDING_PROVIDER.startswith("openrouter:"):
        return _embed_text_batch_openrouter(texts)
    raise RuntimeError(f"Unhandled EMBEDDING_PROVIDER: {config.EMBEDDING_PROVIDER}")


def _embed_text_batch_gemini(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Original Gemini Embedding 2 batch path. Batches of 50."""
    all_vectors: list[list[float]] = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        # Wrap each string in Content so the SDK returns one embedding per item.
        # Passing raw strings merges them into a single Content and returns
        # only one vector for the whole batch.
        batch_contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in batch_texts
        ]
        vectors = _embed_batch_with_retry(client, batch_contents)
        all_vectors.extend(vectors)

    return all_vectors


# ---------------------------------------------------------------------------
# OpenRouter path — process-singleton httpx client + raw JSON parse
# ---------------------------------------------------------------------------

_OPENROUTER_HTTPX_CLIENT: _httpx_typed.Client | None = None


def _get_openrouter_client() -> _httpx_typed.Client:
    """Return a process-singleton httpx.Client to OpenRouter with keep-alive."""
    global _OPENROUTER_HTTPX_CLIENT
    if _OPENROUTER_HTTPX_CLIENT is None:
        import httpx  # noqa: PLC0415
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        _OPENROUTER_HTTPX_CLIENT = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
    return _OPENROUTER_HTTPX_CLIENT


def _embed_text_batch_openrouter(texts: list[str]) -> list[list[float]]:
    """Embed via OpenRouter's OpenAI-compat /v1/embeddings endpoint.

    Uses raw httpx + json.loads instead of the openai SDK. The SDK constructs
    a Pydantic model per response, which on dim=4096 + batch=50 is ~80% of
    wall time for fast docs (recursive field validation over 50 × 4096 floats
    burns CPU). Plain JSON parse is O(n) and 5-10× faster.

    Batches of 50. Returns float vectors at the model's native dim (matches
    config.EMBED_DIM by construction).
    """
    client = _get_openrouter_client()
    all_vectors: list[list[float]] = []
    batch_size = 50

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        resp = client.post(
            "/embeddings",
            json={
                "model": config.EMBED_MODEL,
                "input": batch_texts,
                "encoding_format": "float",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError(
                f"OpenRouter response missing 'data' list. Got: {body!r}"[:500]
            )

        # OpenAI-compat returns data sorted by request order with .index.
        # Sort defensively in case the provider returns out-of-order.
        items = sorted(data, key=lambda d: d.get("index", 0))
        for item in items:
            vec = item.get("embedding")
            if isinstance(vec, str):
                # Some providers return base64 even when "float" requested —
                # fail loudly so we notice rather than silently store garbage.
                raise RuntimeError(
                    f"Provider returned str embedding (likely base64) for "
                    f"model={config.EMBED_MODEL}; expected float array."
                )
            if not isinstance(vec, list) or len(vec) != config.EMBED_DIM:
                got = len(vec) if isinstance(vec, list) else type(vec).__name__
                raise RuntimeError(
                    f"Provider returned dim={got}, expected "
                    f"{config.EMBED_DIM} for model={config.EMBED_MODEL}."
                )
            all_vectors.append(vec)

    if len(all_vectors) != len(texts):
        raise RuntimeError(
            f"OpenRouter returned {len(all_vectors)} embeddings for "
            f"{len(texts)} inputs."
        )
    return all_vectors


# ---------------------------------------------------------------------------
# Image embedding (Gemini multimodal — same vector space as text)
# ---------------------------------------------------------------------------


def _embed_image(
    client: genai.Client,
    path: Path,
) -> list[float]:
    """Embed an image file natively via Gemini Embedding 2 at config.EMBED_DIM.

    The multimodal model maps the image into the same EMBED_DIM space as text
    (3072 gemini / 4096 qwen).
    """
    mime_type = _MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
    image_bytes = path.read_bytes()
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    response = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=cast(types.ContentListUnion, [part]),
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=config.EMBED_DIM,
        ),
    )
    embeddings = response.embeddings
    if not embeddings:
        raise RuntimeError(f"Gemini embed_content returned no embeddings for image: {path}")
    values = embeddings[0].values
    if values is None:
        raise RuntimeError(f"Gemini embedding entry has no values for image: {path}")
    return list(values)
