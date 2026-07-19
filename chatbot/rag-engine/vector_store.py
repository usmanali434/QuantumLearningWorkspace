"""Embed chunks with SentenceTransformer and store/query them in ChromaDB."""

from __future__ import annotations

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "study_chunks"
DEFAULT_TOP_K = 4
# Chroma default space is L2; lower distance = more similar.
# Tuned so on-topic photosynthesis questions pass and off-topic ones refuse.
DEFAULT_MAX_DISTANCE = 1.2


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load the local embedding model used for documents and queries."""
    return SentenceTransformer(model_name)


def create_collection(name: str = DEFAULT_COLLECTION_NAME):
    """Create an in-memory Chroma collection (ephemeral for demo scripts)."""
    client = chromadb.Client()
    return client.create_collection(name=name)


def add_chunks(collection, embedding_model: SentenceTransformer, chunks: list[dict]) -> None:
    """Embed each chunk and add it to the collection with metadata."""
    if not chunks:
        return

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = embedding_model.encode(documents).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def retrieve(
    collection,
    embedding_model: SentenceTransformer,
    question: str,
    n_results: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Query the collection for the top-n chunks matching the question.

    Returns a dict with:
      - documents: list[str]
      - distances: list[float] | None
      - metadatas: list[dict] | None
      - ids: list[str] | None
    """
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
    )

    return {
        "documents": results["documents"][0] if results.get("documents") else [],
        "distances": results["distances"][0] if results.get("distances") else None,
        "metadatas": results["metadatas"][0] if results.get("metadatas") else None,
        "ids": results["ids"][0] if results.get("ids") else None,
    }


def is_relevant(
    distances: list[float] | None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> bool:
    """
    Return True if at least one retrieved chunk is close enough.

    Chroma returns L2 distances (lower = better). Refuse when there are no
    distances, or when the best (minimum) distance exceeds max_distance.
    """
    if not distances:
        return False
    return min(distances) <= max_distance


def format_retrieved_chunks(documents: list[str]) -> str:
    """Join one or more retrieved chunks for the LLM prompt."""
    if not documents:
        return ""
    if len(documents) == 1:
        return documents[0]
    parts = []
    for i, doc in enumerate(documents, start=1):
        parts.append(f"[Chunk {i}]\n{doc}")
    return "\n\n".join(parts)


def chunk_preview(text: str, max_words: int = 20) -> str:
    """Short preview of a chunk for API source metadata."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."
