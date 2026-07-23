"""Unit tests for query rewrite skip, grounding parse, and rerank id selection."""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import (  # noqa: E402
    parse_grounding_response,
    parse_rerank_ids,
    rewrite_question,
    select_results_by_ids,
)


def test_rewrite_skips_llm_when_history_empty():
    # client=None would blow up if called; empty history must short-circuit
    assert rewrite_question(None, [], "Where does it happen?") == "Where does it happen?"
    assert (
        rewrite_question(None, [{"role": "user", "content": "   "}], "Q?") == "Q?"
    )


def test_parse_grounding_true_false():
    assert parse_grounding_response("GROUNDED: true\nLooks fine.") is True
    assert parse_grounding_response("Verdict\nGROUNDED: false") is False
    assert parse_grounding_response("grounded: TRUE") is True
    assert parse_grounding_response("no verdict here") is None


def test_parse_rerank_ids_orders_and_pads():
    candidates = ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]
    text = "Best: chunk_2, chunk_0, chunk_9, chunk_1"
    assert parse_rerank_ids(text, candidates, top_k=3) == [
        "chunk_2",
        "chunk_0",
        "chunk_1",
    ]


def test_parse_rerank_ids_fallback_to_candidates():
    candidates = ["chunk_0", "chunk_1", "chunk_2"]
    assert parse_rerank_ids("none", candidates, top_k=2) == ["chunk_0", "chunk_1"]


def test_select_results_by_ids():
    results = {
        "documents": ["a", "b", "c"],
        "distances": [0.9, 0.5, 0.7],
        "ids": ["chunk_0", "chunk_1", "chunk_2"],
        "metadatas": [{"i": 0}, {"i": 1}, {"i": 2}],
    }
    picked = select_results_by_ids(results, ["chunk_2", "chunk_0"])
    assert picked["ids"] == ["chunk_2", "chunk_0"]
    assert picked["documents"] == ["c", "a"]
    assert picked["distances"] == [0.7, 0.9]
