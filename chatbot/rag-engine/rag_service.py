"""
Shared RAG pipeline: rewrite → retrieve → optional rerank → gate → answer → ground.

Used by the FastAPI /ask endpoint and CLI demos.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from chunker import chunk_file
from vector_store import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MODEL_NAME,
    DEFAULT_TOP_K,
    add_chunks,
    chunk_preview,
    create_collection,
    format_retrieved_chunks,
    is_relevant,
    load_embedding_model,
    retrieve,
)

RAG_ENGINE_DIR = Path(__file__).resolve().parent
DATA_FILE = RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt"
HISTORY_TURN_CAP = 4
REFUSAL_MESSAGE = "I don't have enough information to answer that"
GROQ_MODEL = "llama-3.3-70b-versatile"
MIN_TOP_K = 1
MAX_TOP_K = 8
RERANK_CANDIDATE_COUNT = 10

_GROUNDED_RE = re.compile(r"GROUNDED\s*:\s*(true|false)", re.IGNORECASE)
_CHUNK_ID_RE = re.compile(r"chunk_\d+", re.IGNORECASE)


@dataclass
class SourceInfo:
    id: str
    distance: float | None
    preview: str


@dataclass
class AskResult:
    answer: str
    refused: bool
    top_k: int
    sources: list[SourceInfo] = field(default_factory=list)
    rewritten_question: str = ""
    grounded: bool | None = None
    source_ids: list[str] = field(default_factory=list)


@dataclass
class RagEngine:
    """Warmed embedding model + Chroma collection ready for queries."""

    collection: Any
    embedding_model: Any
    chunks_indexed: int
    embedding_model_name: str = DEFAULT_MODEL_NAME
    default_top_k: int = DEFAULT_TOP_K
    max_distance: float = DEFAULT_MAX_DISTANCE
    _groq: Groq | None = field(default=None, repr=False)


def load_env() -> None:
    """Load .env from repo root, chatbot/, and cwd."""
    repo_root = RAG_ENGINE_DIR.parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(RAG_ENGINE_DIR.parent / ".env")
    load_dotenv(RAG_ENGINE_DIR / ".env")
    load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def clamp_top_k(top_k: int | None, default: int = DEFAULT_TOP_K) -> int:
    value = default if top_k is None else int(top_k)
    return max(MIN_TOP_K, min(MAX_TOP_K, value))


def recent_history(history: list[dict], max_turns: int = HISTORY_TURN_CAP) -> list[dict]:
    """Return the last max_turns user/assistant pairs."""
    max_messages = max_turns * 2
    return history[-max_messages:]


def _history_nonempty(history: list[dict]) -> bool:
    return any((m.get("content") or "").strip() for m in history)


def rewrite_question(client: Groq | None, history: list[dict], question: str) -> str:
    """
    Turn a follow-up into a standalone search query using conversation history.

    Skips the LLM when history is empty (returns the original question).
    """
    if not _history_nonempty(history):
        return question

    if client is None:
        return question

    hist_lines = []
    for msg in recent_history(history):
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if content:
            hist_lines.append(f"{role}: {content}")
    history_block = "\n".join(hist_lines) if hist_lines else "(none)"

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest question into ONE standalone search query "
                "that includes all needed context from the conversation. "
                "Do not answer the question. Output only the rewritten query as plain text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history_block}\n\n"
                f"Latest question: {question}\n\n"
                "Standalone search query:"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    # Take first non-empty line; strip wrapping quotes
    for line in rewritten.splitlines():
        cleaned = line.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return question


def parse_grounding_response(text: str) -> bool | None:
    """Parse GROUNDED: true|false from a judge response. None if unparseable."""
    if not text:
        return None
    match = _GROUNDED_RE.search(text)
    if not match:
        return None
    return match.group(1).lower() == "true"


def parse_rerank_ids(text: str, candidate_ids: list[str], top_k: int) -> list[str]:
    """
    Extract an ordered list of chunk ids from an LLM rerank response.

    Falls back to the original candidate order when parsing yields nothing useful.
    """
    found = _CHUNK_ID_RE.findall(text or "")
    # Normalize case to match candidates
    id_map = {cid.lower(): cid for cid in candidate_ids}
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in found:
        key = raw.lower()
        if key in id_map and key not in seen:
            ordered.append(id_map[key])
            seen.add(key)
        if len(ordered) >= top_k:
            break

    if len(ordered) < top_k:
        for cid in candidate_ids:
            if cid.lower() not in seen:
                ordered.append(cid)
                seen.add(cid.lower())
            if len(ordered) >= top_k:
                break

    return ordered[:top_k] if ordered else candidate_ids[:top_k]


def select_results_by_ids(results: dict[str, Any], ordered_ids: list[str]) -> dict[str, Any]:
    """Reorder a retrieve() result dict to match ordered_ids."""
    documents = results.get("documents") or []
    distances = results.get("distances")
    ids = results.get("ids") or []
    metadatas = results.get("metadatas")

    by_id: dict[str, int] = {}
    for i, cid in enumerate(ids):
        by_id[cid] = i

    new_docs: list[str] = []
    new_dists: list[float] | None = [] if distances is not None else None
    new_ids: list[str] = []
    new_meta: list[dict] | None = [] if metadatas is not None else None

    for cid in ordered_ids:
        if cid not in by_id:
            continue
        i = by_id[cid]
        new_docs.append(documents[i])
        new_ids.append(cid)
        if new_dists is not None and distances is not None:
            new_dists.append(distances[i])
        if new_meta is not None and metadatas is not None:
            new_meta.append(metadatas[i])

    return {
        "documents": new_docs,
        "distances": new_dists,
        "ids": new_ids,
        "metadatas": new_meta,
    }


def rerank_chunks(
    client: Groq,
    question: str,
    results: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Use the LLM to pick the best top_k chunk ids from a wider candidate set."""
    ids = results.get("ids") or []
    documents = results.get("documents") or []
    if len(ids) <= top_k:
        return select_results_by_ids(results, ids[:top_k])

    listing_parts = []
    for cid, doc in zip(ids, documents):
        listing_parts.append(f"[{cid}]\n{doc}")
    listing = "\n\n".join(listing_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You re-rank retrieved document chunks for a search query. "
                f"Return ONLY a comma-separated list of the best {top_k} chunk ids "
                "in order of relevance (most relevant first). "
                "Use the exact ids shown in brackets. No other text."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {question}\n\nCandidates:\n{listing}\n\nBest chunk ids:",
        },
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
    )
    text = response.choices[0].message.content or ""
    ordered = parse_rerank_ids(text, list(ids), top_k)
    return select_results_by_ids(results, ordered)


def build_messages(
    history: list[dict],
    retrieved_text: str,
    question: str,
    *,
    strict: bool = False,
) -> list[dict]:
    """System + prior turns + latest user message with retrieved chunk(s)."""
    if strict:
        system = (
            "You are a careful study assistant. Answer using ONLY facts that appear "
            "in the provided reference chunk(s). Do not use outside knowledge. "
            "If the chunks do not contain enough information, say you don't know. "
            "Do not invent dates, names, or details not written in the chunks."
        )
    else:
        system = (
            "You are a helpful study assistant. Answer using ONLY the provided "
            "reference chunk(s) and the conversation context. If a follow-up "
            "refers to something mentioned earlier (for example 'the second "
            "stage'), use prior turns to resolve it. If the chunk(s) and "
            "history together still lack the answer, say so."
        )

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(recent_history(history))
    messages.append(
        {
            "role": "user",
            "content": (
                f"Reference chunk(s):\n{retrieved_text}\n\n"
                f"Current question: {question}"
            ),
        }
    )
    return messages


def check_grounding(client: Groq, retrieved_text: str, answer: str) -> bool:
    """
    Second LLM call: is the answer supported by the chunks?

    Returns True if grounded, False otherwise (unparseable treated as not grounded).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict factuality judge. Decide whether the ASSISTANT "
                "ANSWER is fully supported by the REFERENCE CHUNKS. "
                "Reply with exactly one line: GROUNDED: true  or  GROUNDED: false. "
                "Use false if the answer adds facts, dates, or claims not present "
                "in the chunks, or if it invents details."
            ),
        },
        {
            "role": "user",
            "content": (
                f"REFERENCE CHUNKS:\n{retrieved_text}\n\n"
                f"ASSISTANT ANSWER:\n{answer}\n\n"
                "Verdict:"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.0,
    )
    text = response.choices[0].message.content or ""
    parsed = parse_grounding_response(text)
    return True if parsed is True else False


def _build_sources(results: dict[str, Any]) -> list[SourceInfo]:
    documents = results.get("documents") or []
    distances = results.get("distances")
    ids = results.get("ids")
    sources: list[SourceInfo] = []
    for i, doc in enumerate(documents):
        chunk_id = ids[i] if ids else f"result_{i}"
        distance = distances[i] if distances is not None else None
        sources.append(
            SourceInfo(
                id=chunk_id,
                distance=distance,
                preview=chunk_preview(doc),
            )
        )
    return sources


def create_engine(
    collection_name: str = "study_chunks",
    data_file: Path | None = None,
) -> RagEngine:
    """Chunk the source doc, embed, and return a ready RagEngine."""
    load_env()
    path = data_file or DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    chunks = chunk_file(path)
    embedding_model = load_embedding_model()
    collection = create_collection(name=collection_name)
    add_chunks(collection, embedding_model, chunks)

    return RagEngine(
        collection=collection,
        embedding_model=embedding_model,
        chunks_indexed=len(chunks),
        embedding_model_name=DEFAULT_MODEL_NAME,
        default_top_k=clamp_top_k(_env_int("DEFAULT_TOP_K", DEFAULT_TOP_K)),
        max_distance=_env_float("MAX_DISTANCE", DEFAULT_MAX_DISTANCE),
    )


def _get_groq(engine: RagEngine) -> Groq:
    if engine._groq is not None:
        return engine._groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file before asking."
        )
    engine._groq = Groq(api_key=api_key)
    return engine._groq


def _append_history(history: list[dict], question: str, answer: str) -> None:
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    capped = recent_history(history)
    history.clear()
    history.extend(capped)


def _generate_answer(
    client: Groq,
    history: list[dict],
    retrieved_text: str,
    question: str,
    *,
    strict: bool = False,
) -> str:
    messages = build_messages(history, retrieved_text, question, strict=strict)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2 if strict else 0.3,
    )
    return response.choices[0].message.content or ""


def ask(
    engine: RagEngine,
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
    include_sources: bool = True,
    update_history: bool = False,
    rerank: bool | None = None,
) -> AskResult:
    """
    Full Phase 5 pipeline:
      rewrite (if history) → retrieve → optional LLM rerank → relevance gate
      → answer with ORIGINAL question → grounding check → one stricter regen.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    k = clamp_top_k(top_k, default=engine.default_top_k)
    hist = list(history) if history is not None else []

    env_rerank = _env_bool("ENABLE_RERANK", True)
    do_rerank = env_rerank if rerank is None else (bool(rerank) and env_rerank)

    # Part A: rewrite for retrieval only (LLM only when history exists)
    client: Groq | None = None
    if _history_nonempty(hist):
        client = _get_groq(engine)
        rewritten = rewrite_question(client, hist, cleaned)
    else:
        rewritten = cleaned

    n_retrieve = RERANK_CANDIDATE_COUNT if do_rerank else k
    # Never ask Chroma for more than indexed chunks
    n_retrieve = max(1, min(n_retrieve, max(engine.chunks_indexed, 1)))

    results = retrieve(
        engine.collection,
        engine.embedding_model,
        rewritten,
        n_results=n_retrieve,
    )

    if do_rerank and len(results.get("ids") or []) > k:
        if client is None:
            client = _get_groq(engine)
        results = rerank_chunks(client, rewritten, results, top_k=k)
    else:
        # Trim to top_k when not reranking (or too few candidates)
        ids = (results.get("ids") or [])[:k]
        results = select_results_by_ids(results, ids)

    distances = results.get("distances")

    if not is_relevant(distances, max_distance=engine.max_distance):
        answer = REFUSAL_MESSAGE
        if update_history and history is not None:
            _append_history(history, cleaned, answer)
        return AskResult(
            answer=answer,
            refused=True,
            top_k=k,
            sources=[],
            rewritten_question=rewritten,
            grounded=None,
            source_ids=[],
        )

    documents = results["documents"]
    retrieved_text = format_retrieved_chunks(documents) if documents else "(none)"

    if client is None:
        client = _get_groq(engine)

    # Answer with ORIGINAL question (natural response)
    answer = _generate_answer(client, hist, retrieved_text, cleaned, strict=False)

    # Part C: grounding check + one stricter regenerate
    grounded = check_grounding(client, retrieved_text, answer)
    if not grounded:
        answer = _generate_answer(client, hist, retrieved_text, cleaned, strict=True)
        grounded = check_grounding(client, retrieved_text, answer)

    if update_history and history is not None:
        _append_history(history, cleaned, answer)

    final_sources = _build_sources(results) if include_sources else []
    final_ids = [s.id for s in final_sources] if include_sources else list(
        results.get("ids") or []
    )

    return AskResult(
        answer=answer,
        refused=False,
        top_k=k,
        sources=final_sources,
        rewritten_question=rewritten,
        grounded=grounded,
        source_ids=final_ids,
    )
