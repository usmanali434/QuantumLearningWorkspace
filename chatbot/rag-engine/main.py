"""
Team Mu RAG Chat API — FastAPI service (separate from web/backend).

Run from chatbot/rag-engine/:
  uvicorn main:app --reload --host 127.0.0.1 --port 8001

Or from repo root:
  uvicorn main:app --reload --app-dir chatbot/rag-engine --host 127.0.0.1 --port 8001

Interactive docs: http://127.0.0.1:8001/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_service import RagEngine, ask, create_engine, load_env
from schemas import AskRequest, AskResponse, HealthResponse, SourceItem

# Module-level engine set during lifespan
_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="RAG engine is not ready yet")
    return _engine


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    load_env()
    _engine = create_engine(collection_name="study_chunks_api")
    yield
    _engine = None


app = FastAPI(
    title="StudyMind Chatbot — RAG API",
    description=(
        "Team Mu chat service: query rewriting, multi-chunk retrieval, "
        "optional LLM re-ranking, source attribution, and grounding checks. "
        "See docs/api-contracts.md for the Web team contract."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(
        status="ok",
        chunks_indexed=engine.chunks_indexed,
        embedding_model=engine.embedding_model_name,
        default_top_k=engine.default_top_k,
        max_distance=engine.max_distance,
    )


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    engine = get_engine()
    history = (
        [{"role": m.role, "content": m.content} for m in body.history]
        if body.history
        else None
    )

    try:
        result = ask(
            engine,
            body.question,
            history=history,
            top_k=body.top_k,
            include_sources=body.include_sources,
            update_history=False,
            rerank=body.rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Missing GROQ_API_KEY when an LLM call is required
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = None
    source_ids: list[str] = []
    if body.include_sources:
        if result.refused:
            sources = []
            source_ids = []
        else:
            sources = [
                SourceItem(id=s.id, distance=s.distance, preview=s.preview)
                for s in result.sources
            ]
            source_ids = list(result.source_ids) or [s.id for s in result.sources]

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        top_k=result.top_k,
        sources=sources,
        source_ids=source_ids,
        rewritten_question=result.rewritten_question,
        grounded=result.grounded,
    )
