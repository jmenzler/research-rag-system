# long-ok-file
"""Hierarchical and structural chunking for the ingest pipeline.

Two ways to produce ``(parents, children)`` from a document:

1. ``_build_hierarchical_chunks`` — text-only path. Splits flat text into
   ~1000-tok parents, then ~512-tok children with overlap. Used for HTML,
   plain text, and PDFs without a ``content_list.json`` sidecar.
2. ``_proto_chunks_to_models`` — structural path. Consumes ``ProtoChunk``s
   from the structural chunker (``src.chunking``) which preserves headings,
   tables, equations, and figure captions from MinerU's ``content_list.json``.

Both emit ``ParentChunk`` (sqlite-backed) and ``ChildChunk`` (Milvus-backed)
sharing ``chunk_id`` collisions only by design (deterministic content hash).

Tail-section stripping (``_strip_tail_sections``) removes References /
Acknowledgments / Appendix tails before chunking so retrieval doesn't
surface citation lists as "answer" content.

Context sidecars (``_load_context_sidecar``) read pre-computed Anthropic
Contextual Retrieval summaries written by ``contextualize_corpus`` and
prepend them to the embed text of each child.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken

from src import config
from src.chunking import ProtoChunk
from src.models import ChildChunk, ParentChunk, chunk_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _get_encoder() -> tiktoken.Encoding:
    """Return the cl100k_base tiktoken encoder (gpt-4 compatible)."""
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Return the number of tokens in *text* using the given encoder."""
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Hierarchical chunker (flat text → parents → children)
# ---------------------------------------------------------------------------


def _split_into_parent_chunks(
    text: str,
    enc: tiktoken.Encoding,
    max_tokens: int,
) -> list[str]:
    """Split *text* into parent-size segments (~*max_tokens* tokens each).

    Strategy:
    1. Try paragraph splits first (double-newline boundaries).
    2. Fall back to sentence splitting when a paragraph exceeds *max_tokens*.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para, enc)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        if para_tokens > max_tokens:
            sentences = [s.strip() for s in para.split(". ") if s.strip()]
            for sent in sentences:
                sent_tokens = _count_tokens(sent, enc)
                if current_tokens + sent_tokens > max_tokens and current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0
                current.append(sent)
                current_tokens += sent_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current) if len(current) > 1 else current[0])

    return chunks if chunks else [text]


def _split_into_child_chunks(
    text: str,
    enc: tiktoken.Encoding,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Split *text* into child-size segments (~*max_tokens* tokens each).

    Uses word-level splits to respect boundaries. When *overlap_tokens* > 0,
    each subsequent chunk begins with the last ~overlap_tokens worth of words
    from the previous chunk, preserving cross-boundary context for retrieval.
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    word_token_counts: list[int] = []

    def _carryover() -> tuple[list[str], int]:
        if overlap_tokens <= 0 or not current:
            return [], 0
        carry: list[str] = []
        carry_tokens = 0
        for w, t in zip(reversed(current), reversed(word_token_counts)):
            if carry_tokens + t > overlap_tokens:
                break
            carry.insert(0, w)
            carry_tokens += t
        return carry, carry_tokens

    for word in words:
        word_tokens = _count_tokens(word, enc)
        if current_tokens + word_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            carry, carry_tokens = _carryover()
            current = list(carry)
            word_token_counts = [_count_tokens(w, enc) for w in carry]
            current_tokens = carry_tokens
        current.append(word)
        word_token_counts.append(word_tokens)
        current_tokens += word_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Tail-section stripping
# ---------------------------------------------------------------------------

# Section names that mark the END of substantive paper content.
_TAIL_SECTION_NAMES = (
    r"(?:references|bibliography|works\s+cited|citations?|cited\s+by|"
    r"acknowledg(?:e?ment)?s?|funding|"
    r"author\s+contributions?|conflicts?\s+of\s+interest|"
    r"competing\s+interests?|data\s+availability|"
    r"supplementary(?:\s+(?:material|information))?|appendix\s*[A-Z0-9]?)"
)

# HARD form: markdown-style heading (## References) — trust regardless of position.
_TAIL_HARD_RE = re.compile(
    rf"(?im)^[ \t]*#{{1,4}}[ \t]+{_TAIL_SECTION_NAMES}[ \t]*:?\s*$"
)

# SOFT form: bare line "References:" or "Cited by:" — require late position
# to avoid clipping body prose that ends a paragraph with "...references."
_TAIL_SOFT_RE = re.compile(
    rf"(?im)^[ \t]*{_TAIL_SECTION_NAMES}[ \t]*:?\s*$"
)


def _strip_tail_sections(text: str) -> tuple[str, str | None]:
    """Truncate text at the first reference/acknowledgments/etc heading.

    Returns (kept_text, dropped_reason). Two-tier matching:
    - HARD: markdown heading (## References) — trust at any position.
    - SOFT: bare-text heading (References:) — require >=20% into doc to avoid
      false positives on top-of-doc TOCs / nav.
    """
    m = _TAIL_HARD_RE.search(text)
    if m:
        return text[: m.start()].rstrip(), m.group(0).strip()

    soft_floor = int(len(text) * 0.2)
    for cand in _TAIL_SOFT_RE.finditer(text):
        if cand.start() >= soft_floor:
            return text[: cand.start()].rstrip(), cand.group(0).strip()
    return text, None


# ---------------------------------------------------------------------------
# Context sidecar (Anthropic Contextual Retrieval)
# ---------------------------------------------------------------------------


def _load_context_sidecar(source_file: str) -> dict[tuple[int, int], str]:
    """Load ``<source>.ctx.json`` sidecar if present, indexed by (parent_idx, child_idx).

    Sidecar format:
        {"chunks": [{"parent_idx": 0, "child_idx": 0, "context": "..."}, ...]}
    Absent / empty / malformed → empty dict (caller falls back to plain chunks).
    """
    sidecar = Path(source_file).with_suffix(Path(source_file).suffix + ".ctx.json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return {(c["parent_idx"], c["child_idx"]): c["context"] for c in data.get("chunks", [])}
    except Exception as e:  # noqa: BLE001
        logger.warning("ctx sidecar unreadable for %s: %s", source_file, e)
        return {}


# ---------------------------------------------------------------------------
# Source-level metadata
# ---------------------------------------------------------------------------


def _load_source_meta(source_dir: Path) -> dict[str, Any] | None:
    """Load ``meta.json`` for a source directory; ``None`` on missing/corrupt.

    Defensive: any read or parse error is logged and returns ``None`` so the
    structural chunker falls back to breadcrumb-only chunks. We intentionally
    do NOT raise — meta.json is best-effort signal, not load-bearing.
    """
    meta_path = source_dir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("meta.json unreadable for %s: %s", source_dir, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Chunk assembly (parents + children)
# ---------------------------------------------------------------------------


def _build_hierarchical_chunks(
    text: str,
    source_file: str,
    notebook: str,
    modality: str,
    page_number: int,
    enc: tiktoken.Encoding,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Produce (parents, children) from a block of text.

    Each parent (~1000 tok) is subdivided into children (~512 tok with 50-tok overlap).
    Children reference their parent via *parent_id*. If a ``<source>.ctx.json``
    sidecar exists (produced by ``scripts/contextualize_corpus.py``), the
    pre-computed context summary is prepended to each child chunk before
    embedding (Anthropic Contextual Retrieval pattern).
    """
    text, dropped = _strip_tail_sections(text)
    if dropped:
        logger.info("stripped tail section '%s' from %s", dropped, source_file)
    parent_texts = _split_into_parent_chunks(text, enc, config.PARENT_TOKENS)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    contexts = _load_context_sidecar(source_file)
    n_with_ctx = 0

    for p_idx, p_text in enumerate(parent_texts):
        p_id = chunk_id(source_file, page_number, p_idx, "parent")
        parent = ParentChunk(
            id=p_id,
            text=p_text,
            source_file=source_file,
            notebook=notebook,
            modality=modality,
            page_number=page_number,
        )
        parents.append(parent)

        child_texts = _split_into_child_chunks(
            p_text, enc, config.CHILD_TOKENS, overlap_tokens=config.CHILD_OVERLAP
        )
        for c_idx, c_text in enumerate(child_texts):
            if _count_tokens(c_text, enc) < config.CHILD_MIN_TOKENS:
                ntok = _count_tokens(c_text, enc)
                logger.warning("skip short child (%d tok) in %s", ntok, source_file)
                continue
            ctx = contexts.get((p_idx, c_idx))
            doc_id = Path(source_file).stem
            if ctx:
                embed_text = f"[{doc_id}] {ctx}\n\n{c_text}"
                n_with_ctx += 1
            else:
                embed_text = f"[{doc_id}]\n\n{c_text}"
            c_id = chunk_id(source_file, page_number, p_idx * 10000 + c_idx, "child")
            child = ChildChunk(
                id=c_id,
                parent_id=p_id,
                text=embed_text,
                source_file=source_file,
                notebook=notebook,
                modality=modality,
                page_number=page_number,
            )
            children.append(child)

    if contexts:
        logger.info("ctx sidecar applied: %d/%d children contextualized for %s",
                    n_with_ctx, len(children), source_file)
    return parents, children


# Milvus VARCHAR(4096) cap on child text. Parent text is uncapped (sqlite).
_CHILD_TEXT_MAX_CHARS = 4096
_CHILD_TEXT_TRUNCATE_AT = 4090
_CHILD_TEXT_ELLIPSIS = " …"


def _cap_child_text(text: str, child_id: str) -> str:
    """Truncate child text to fit Milvus VARCHAR(4096). Parent text is uncapped."""
    if len(text) <= _CHILD_TEXT_MAX_CHARS:
        return text
    logger.warning(
        "truncated child %s: %d chars -> %d (Milvus VARCHAR limit)",
        child_id, len(text), _CHILD_TEXT_MAX_CHARS,
    )
    return text[:_CHILD_TEXT_TRUNCATE_AT] + _CHILD_TEXT_ELLIPSIS


def _apply_ctx(text: str, ctx: str | None) -> str:
    """Prepend an Anthropic Contextual-Retrieval summary ahead of chunk text.

    The structural chunk text already opens with ``Title (Cite) → Breadcrumb``;
    the context sentence goes in front of that so retrieval sees both the
    document anchor and the chunk-specific summary. No-op when ``ctx`` is empty.
    """
    if not ctx:
        return text
    return f"{ctx}\n\n{text}"


def iter_child_keys(
    proto_chunks: list[ProtoChunk],
) -> list[tuple[int, int, str]]:
    """Yield ``(parent_idx, child_idx, raw_text)`` for every child a structural
    ingest would emit, in the SAME order/keys as ``_proto_chunks_to_models``.

    ``contextualize_corpus`` calls this so its sidecar keys line up exactly with
    the children that ingest produces — including the synthetic child_idx that
    atomic (table) groups get. ``raw_text`` is the chunk body (no breadcrumb),
    used as the ``<chunk>`` anchor sent to the contextualizer.
    """
    groups: dict[int, list[ProtoChunk]] = {}
    for proto in proto_chunks:
        groups.setdefault(proto.parent_idx, []).append(proto)

    out: list[tuple[int, int, str]] = []
    for parent_idx, group in groups.items():
        child_members = [p for p in group if p.child_idx is not None]
        none_members = [p for p in group if p.child_idx is None]
        if child_members:
            for proto in child_members:
                assert proto.child_idx is not None
                out.append((parent_idx, proto.child_idx, proto.raw_text))
        else:
            for synth_idx, proto in enumerate(none_members):
                out.append((parent_idx, synth_idx, proto.raw_text))
    return out


def _proto_chunks_to_models(
    proto_chunks: list[ProtoChunk],
    source_file: str,
    notebook: str,
    modality: str,
    *,
    ctx_sidecar: str | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Convert structural-chunker output to ParentChunk + ChildChunk.

    Per ProtoChunk group (keyed by parent_idx):
      * One parent: text from the child_idx=None member if present (table),
        otherwise '\\n\\n'-joined raw_text of all members (text section).
      * Children: every member with child_idx is not None becomes a ChildChunk.
        If the group is purely atomic (no child_idx), a synthetic child is
        emitted whose text equals the proto's embed text — Milvus only
        indexes children, so without one the table is invisible to retrieval.

    When ``ctx_sidecar`` is given (the ``content_list.json`` whose
    ``.ctx.json`` sidecar holds Anthropic Contextual-Retrieval summaries),
    each child's embed text is prefixed with its ``(parent_idx, child_idx)``
    context. Keys align because the sidecar was produced by the SAME structural
    chunker (``contextualize_corpus`` runs ``chunk_sidecar`` too).

    Child text is capped at Milvus's VARCHAR(4096) limit (truncation +
    ``" …"`` marker on overflow). Parent text is uncapped because parents
    live in sqlite. The ``modality`` argument is the document-level label
    (pdf/text); chunk modality follows the proto (text/table/equation).
    """
    contexts = _load_context_sidecar(ctx_sidecar) if ctx_sidecar else {}
    n_with_ctx = 0

    groups: dict[int, list[ProtoChunk]] = {}
    for proto in proto_chunks:
        groups.setdefault(proto.parent_idx, []).append(proto)

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for parent_idx, group in groups.items():
        none_members = [p for p in group if p.child_idx is None]
        child_members = [p for p in group if p.child_idx is not None]

        if none_members:
            head = none_members[0]
            parent_text = head.raw_text
        else:
            head = group[0]
            parent_text = "\n\n".join(p.raw_text for p in group)

        parent_modality = head.modality
        page_number = head.page_idx

        p_id = chunk_id(source_file, page_number, parent_idx, "parent")
        parents.append(
            ParentChunk(
                id=p_id,
                text=parent_text,
                source_file=source_file,
                notebook=notebook,
                modality=parent_modality,
                page_number=page_number,
            )
        )

        if child_members:
            for proto in child_members:
                assert proto.child_idx is not None
                c_id = chunk_id(
                    source_file,
                    proto.page_idx,
                    parent_idx * 10000 + proto.child_idx,
                    "child",
                )
                ctx = contexts.get((parent_idx, proto.child_idx))
                if ctx:
                    n_with_ctx += 1
                children.append(
                    ChildChunk(
                        id=c_id,
                        parent_id=p_id,
                        text=_cap_child_text(_apply_ctx(proto.text, ctx), c_id),
                        source_file=source_file,
                        notebook=notebook,
                        modality=proto.modality,
                        page_number=proto.page_idx,
                    )
                )
        else:
            for synth_idx, proto in enumerate(none_members):
                c_id = chunk_id(
                    source_file,
                    proto.page_idx,
                    parent_idx * 10000 + synth_idx,
                    "child",
                )
                ctx = contexts.get((parent_idx, synth_idx))
                if ctx:
                    n_with_ctx += 1
                children.append(
                    ChildChunk(
                        id=c_id,
                        parent_id=p_id,
                        text=_cap_child_text(_apply_ctx(proto.text, ctx), c_id),
                        source_file=source_file,
                        notebook=notebook,
                        modality=proto.modality,
                        page_number=proto.page_idx,
                    )
                )

    if contexts:
        logger.info(
            "ctx sidecar applied: %d/%d children contextualized for %s",
            n_with_ctx, len(children), source_file,
        )
    return parents, children
