"""
Shared RAG pipeline: warm store once, retrieve top-k, relevance gate, then Groq.

Used by the FastAPI /ask endpoint and CLI demos.
"""

from __future__ import annotations

import os
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


def clamp_top_k(top_k: int | None, default: int = DEFAULT_TOP_K) -> int:
    value = default if top_k is None else int(top_k)
    return max(MIN_TOP_K, min(MAX_TOP_K, value))


def recent_history(history: list[dict], max_turns: int = HISTORY_TURN_CAP) -> list[dict]:
    """Return the last max_turns user/assistant pairs."""
    max_messages = max_turns * 2
    return history[-max_messages:]


def build_messages(
    history: list[dict],
    retrieved_text: str,
    question: str,
) -> list[dict]:
    """System + prior turns + latest user message with retrieved chunk(s)."""
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a helpful study assistant. Answer using ONLY the provided "
                "reference chunk(s) and the conversation context. If a follow-up "
                "refers to something mentioned earlier (for example 'the second "
                "stage'), use prior turns to resolve it. If the chunk(s) and "
                "history together still lack the answer, say so."
            ),
        }
    ]
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


def ask(
    engine: RagEngine,
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
    include_sources: bool = True,
    update_history: bool = False,
) -> AskResult:
    """
    Retrieve top-k chunks, refuse if none are relevant enough, else call Groq.

    If update_history is True and history is a mutable list, append this turn
    and cap to HISTORY_TURN_CAP pairs (CLI conversational mode).
    """
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("question must be a non-empty string")

    k = clamp_top_k(top_k, default=engine.default_top_k)
    hist = history if history is not None else []

    results = retrieve(
        engine.collection,
        engine.embedding_model,
        cleaned,
        n_results=k,
    )
    sources = _build_sources(results) if include_sources else []
    distances = results.get("distances")

    if not is_relevant(distances, max_distance=engine.max_distance):
        answer = REFUSAL_MESSAGE
        if update_history and history is not None:
            history.append({"role": "user", "content": cleaned})
            history.append({"role": "assistant", "content": answer})
            capped = recent_history(history)
            history.clear()
            history.extend(capped)
        return AskResult(
            answer=answer,
            refused=True,
            top_k=k,
            sources=sources if include_sources else [],
        )

    documents = results["documents"]
    retrieved_text = format_retrieved_chunks(documents) if documents else "(none)"
    client = _get_groq(engine)
    messages = build_messages(hist, retrieved_text, cleaned)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
    )
    answer = response.choices[0].message.content or ""

    if update_history and history is not None:
        history.append({"role": "user", "content": cleaned})
        history.append({"role": "assistant", "content": answer})
        capped = recent_history(history)
        history.clear()
        history.extend(capped)

    return AskResult(
        answer=answer,
        refused=False,
        top_k=k,
        sources=sources if include_sources else [],
    )
