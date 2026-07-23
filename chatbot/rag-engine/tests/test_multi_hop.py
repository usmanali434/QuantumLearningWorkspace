"""Unit tests for multi-hop parsing, delimiters, and multi-file chunk ids."""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from chunker import chunk_data_directory, chunk_file, source_slug_for_file  # noqa: E402
from rag_service import (  # noqa: E402
    merge_results,
    parse_enough_decision,
)
from vector_store import format_untrusted_chunks  # noqa: E402


def test_parse_enough_true():
    enough, nxt = parse_enough_decision("ENOUGH: true\nLooks complete.")
    assert enough is True
    assert nxt is None


def test_parse_enough_false_with_next_query():
    enough, nxt = parse_enough_decision(
        "ENOUGH: false\nNEXT_QUERY: YouTube lecture ATP synthase proton gradient"
    )
    assert enough is False
    assert nxt is not None
    assert "ATP" in nxt


def test_parse_enough_unparseable_defaults_to_enough():
    enough, nxt = parse_enough_decision("I need more info somehow")
    assert enough is True
    assert nxt is None


def test_format_untrusted_chunks_delimiters():
    text = format_untrusted_chunks(
        ["hello world"],
        ids=["pdf_chunk_0"],
        metadatas=[{"source": "photosynthesis_overview.txt"}],
    )
    assert '<<<UNTRUSTED_DOCUMENT id="pdf_chunk_0"' in text
    assert 'source="photosynthesis_overview.txt"' in text
    assert "<<<END_UNTRUSTED_DOCUMENT>>>" in text
    assert "hello world" in text


def test_source_slug_mapping():
    assert source_slug_for_file("photosynthesis_overview.txt") == "pdf"
    assert source_slug_for_file("youtube_lecture_energy.txt") == "yt"
    assert source_slug_for_file("conflict_notes.txt") == "conflict"
    assert source_slug_for_file("injection_sample.txt") == "injection"


def test_chunk_file_prefixed_ids():
    path = RAG_ENGINE_DIR / "data" / "injection_sample.txt"
    chunks = chunk_file(path, min_words=40)
    assert chunks
    assert all(c["id"].startswith("injection_chunk_") for c in chunks)
    assert any("hacked" in c["text"] for c in chunks)


def test_chunk_data_directory_multi_source():
    chunks = chunk_data_directory(RAG_ENGINE_DIR / "data")
    ids = {c["id"] for c in chunks}
    sources = {c["metadata"]["source"] for c in chunks}
    assert any(i.startswith("pdf_chunk_") for i in ids)
    assert any(i.startswith("yt_chunk_") for i in ids)
    assert any(i.startswith("conflict_chunk_") for i in ids)
    assert "photosynthesis_overview.txt" in sources
    assert "youtube_lecture_energy.txt" in sources


def test_merge_results_dedupes_by_id():
    a = {
        "documents": ["a"],
        "distances": [0.1],
        "ids": ["pdf_chunk_0"],
        "metadatas": [{"source": "a.txt"}],
    }
    b = {
        "documents": ["a-dup", "b"],
        "distances": [0.2, 0.3],
        "ids": ["pdf_chunk_0", "yt_chunk_0"],
        "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
    }
    merged = merge_results(a, b)
    assert merged["ids"] == ["pdf_chunk_0", "yt_chunk_0"]
    assert merged["documents"] == ["a", "b"]
