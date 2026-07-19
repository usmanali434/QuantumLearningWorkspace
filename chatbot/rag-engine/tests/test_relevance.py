"""Unit tests for relevance gating and chunk formatting (no Groq / no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from vector_store import (  # noqa: E402
    DEFAULT_MAX_DISTANCE,
    format_retrieved_chunks,
    is_relevant,
)


def test_is_relevant_true_when_best_within_threshold():
    assert is_relevant([0.9, 1.5, 1.8], max_distance=1.2) is True


def test_is_relevant_false_when_all_too_far():
    assert is_relevant([1.5, 1.7, 2.0], max_distance=1.2) is False


def test_is_relevant_false_when_empty_or_none():
    assert is_relevant(None) is False
    assert is_relevant([]) is False


def test_is_relevant_uses_default_max_distance():
    assert is_relevant([DEFAULT_MAX_DISTANCE]) is True
    assert is_relevant([DEFAULT_MAX_DISTANCE + 0.01]) is False


def test_format_retrieved_chunks_single():
    assert format_retrieved_chunks(["only one"]) == "only one"


def test_format_retrieved_chunks_multi_has_separators():
    text = format_retrieved_chunks(["alpha chunk", "beta chunk"])
    assert "[Chunk 1]\nalpha chunk" in text
    assert "[Chunk 2]\nbeta chunk" in text
    assert "\n\n" in text


def test_format_retrieved_chunks_empty():
    assert format_retrieved_chunks([]) == ""


def test_clamp_top_k():
    from rag_service import clamp_top_k

    assert clamp_top_k(None, default=4) == 4
    assert clamp_top_k(0, default=4) == 1
    assert clamp_top_k(99, default=4) == 8
    assert clamp_top_k(3) == 3
