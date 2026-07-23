"""API contract tests (no Groq, mocked RAG pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))


@pytest.fixture()
def client():
    mock_engine = MagicMock()
    mock_engine.chunks_indexed = 7
    mock_engine.embedding_model_name = "all-MiniLM-L6-v2"
    mock_engine.default_top_k = 4
    mock_engine.max_distance = 1.2

    with patch("main.create_engine", return_value=mock_engine):
        import main

        main._engine = mock_engine
        main._engine_ready = True
        with TestClient(main.app) as test_client:
            yield test_client
        main._engine = None
        main._engine_ready = False


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["chunks_indexed"] == 7
    assert "cache_backend" in data
    assert "rate_limit_backend" in data


def test_ask_empty_question_400(client):
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


@patch("main.prepare_ask")
def test_ask_rate_limit_429(mock_prepare, client):
    from rate_limiter import RateLimiter
    from rag_service import PreparedAsk

    mock_prepare.return_value = PreparedAsk(
        question="What is photosynthesis?",
        history=[],
        top_k=4,
        rewritten_question="What is photosynthesis?",
        hop_queries=["What is photosynthesis?"],
        retrieved_text="",
        accumulated={"documents": [], "ids": [], "distances": [], "metadatas": []},
        refused=True,
        refusal_answer="I don't have enough information to answer that",
        include_sources=True,
    )

    limiter = RateLimiter(max_requests=2, window_seconds=60, redis_url="")
    with patch("rate_limiter.rate_limiter", limiter):
        headers = {"X-User-Id": "rate-test-user"}
        body = {"question": "What is photosynthesis?", "skip_cache": True}
        assert client.post("/ask", json=body, headers=headers).status_code != 429
        assert client.post("/ask", json=body, headers=headers).status_code != 429
        resp = client.post("/ask", json=body, headers=headers)
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After")


@patch("main.prepare_ask")
@patch("main.finalize_ask")
@patch("main.generate_answer_sync")
def test_ask_success_mocked(mock_gen, mock_finalize, mock_prepare, client):
    from rag_service import AskResult, PreparedAsk, SourceInfo

    prepared = PreparedAsk(
        question="Where is the Calvin cycle?",
        history=[],
        top_k=4,
        rewritten_question="Where is the Calvin cycle?",
        hop_queries=["Where is the Calvin cycle?"],
        retrieved_text="<<<UNTRUSTED_DOCUMENT>>>...",
        accumulated={"documents": ["x"], "ids": ["pdf_chunk_1"], "distances": [0.5], "metadatas": [{}]},
        refused=False,
        include_sources=True,
    )
    mock_prepare.return_value = prepared
    mock_gen.return_value = "The Calvin cycle occurs in the stroma."
    mock_finalize.return_value = AskResult(
        answer="The Calvin cycle occurs in the stroma.",
        refused=False,
        top_k=4,
        sources=[SourceInfo(id="pdf_chunk_1", distance=0.5, preview="stroma", source="pdf.txt")],
        source_ids=["pdf_chunk_1"],
        rewritten_question="Where is the Calvin cycle?",
        grounded=True,
        retrieval_rounds=1,
        hop_queries=["Where is the Calvin cycle?"],
    )

    resp = client.post(
        "/ask",
        json={"question": "Where is the Calvin cycle?", "skip_cache": True},
        headers={"X-User-Id": "mock-user"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "stroma" in data["answer"]
    assert data["grounded"] is True
    assert data["cached"] is False
    assert data["timing"] is not None
    assert resp.headers.get("X-Cache-Hit") == "0"


@patch("main.prepare_ask")
def test_ask_stream_refusal_metadata_done(mock_prepare, client):
    from rag_service import PreparedAsk

    prepared = PreparedAsk(
        question="off topic",
        history=[],
        top_k=4,
        rewritten_question="off topic",
        hop_queries=["off topic"],
        retrieved_text="",
        accumulated={"documents": [], "ids": [], "distances": [], "metadatas": []},
        refused=True,
        refusal_answer="I don't have enough information to answer that",
        include_sources=True,
    )
    mock_prepare.return_value = prepared

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "off topic", "skip_cache": True},
        headers={"X-User-Id": "stream-user"},
    ) as resp:
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.iter_lines() if line]

    types = [e["type"] for e in lines]
    assert types[0] == "metadata"
    assert lines[0]["refused"] is True
    assert "token" not in types
    assert types[-1] == "done"


@patch("main.prepare_ask")
@patch("main.finalize_ask")
@patch("main.stream_answer_tokens")
def test_ask_stream_tokens(mock_stream, mock_finalize, mock_prepare, client):
    from rag_service import AskResult, PreparedAsk

    prepared = PreparedAsk(
        question="What is ATP?",
        history=[],
        top_k=4,
        rewritten_question="What is ATP?",
        hop_queries=["What is ATP?"],
        retrieved_text="energy",
        accumulated={"documents": ["x"], "ids": ["yt_chunk_0"], "distances": [0.4], "metadatas": [{}]},
        refused=False,
        include_sources=True,
        client=MagicMock(),
    )
    mock_prepare.return_value = prepared
    mock_stream.return_value = iter(["ATP ", "is energy."])
    mock_finalize.return_value = AskResult(
        answer="ATP is energy.",
        refused=False,
        top_k=4,
        source_ids=["yt_chunk_0"],
        grounded=True,
        retrieval_rounds=1,
        hop_queries=["What is ATP?"],
    )

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "What is ATP?", "skip_cache": True},
        headers={"X-User-Id": "stream-user-2"},
    ) as resp:
        lines = [json.loads(line) for line in resp.iter_lines() if line]

    assert any(e["type"] == "token" for e in lines)
    assert lines[-1]["type"] == "done"
    assert lines[-1]["grounded"] is True
