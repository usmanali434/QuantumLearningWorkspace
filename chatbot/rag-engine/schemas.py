"""Pydantic models for the Team Mu RAG chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from vector_store import DEFAULT_TOP_K


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer")
    history: list[HistoryMessage] | None = Field(
        default=None,
        description="Optional prior conversation turns (user/assistant pairs)",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description=f"Number of chunks to keep per retrieval round (default {DEFAULT_TOP_K}, max 8)",
    )
    include_sources: bool = Field(
        default=True,
        description="Include retrieved chunk ids/distances/previews in the response",
    )
    rerank: bool = Field(
        default=True,
        description=(
            "When true, retrieve a wider candidate pool and LLM-rerank to top_k. "
            "Ignored if ENABLE_RERANK=false in the environment."
        ),
    )
    multi_hop: bool = Field(
        default=True,
        description=(
            "When true, allow up to MAX_RETRIEVAL_ROUNDS agentic retrieval hops. "
            "Ignored if ENABLE_MULTI_HOP=false."
        ),
    )
    skip_cache: bool = Field(
        default=False,
        description="Bypass the in-memory answer cache for this request",
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")
        return cleaned


class SourceItem(BaseModel):
    id: str
    distance: float | None = None
    preview: str
    source: str = ""


class TimingInfo(BaseModel):
    retrieval_ms: float | None = None
    llm_ms: float | None = None
    grounding_ms: float | None = None
    total_ms: float | None = None


class AskResponse(BaseModel):
    answer: str
    refused: bool = False
    top_k: int = DEFAULT_TOP_K
    sources: list[SourceItem] | None = None
    source_ids: list[str] = Field(default_factory=list)
    rewritten_question: str = ""
    grounded: bool | None = None
    retrieval_rounds: int = 0
    hop_queries: list[str] = Field(default_factory=list)
    conflict_hint: bool = False
    cached: bool = False
    timing: TimingInfo | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool = True
    chunks_indexed: int
    embedding_model: str
    default_top_k: int
    max_distance: float
    cache_entries: int = 0
    cache_hits: int = 0
    cache_backend: str = "memory"
    rate_limit_backend: str = "memory"
    groq_configured: bool = False
