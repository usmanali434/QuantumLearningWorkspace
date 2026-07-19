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
        description=f"Number of chunks to retrieve (default {DEFAULT_TOP_K}, max 8)",
    )
    include_sources: bool = Field(
        default=True,
        description="Include retrieved chunk ids/distances/previews in the response",
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


class AskResponse(BaseModel):
    answer: str
    refused: bool = False
    top_k: int = DEFAULT_TOP_K
    sources: list[SourceItem] | None = None


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embedding_model: str
    default_top_k: int
    max_distance: float
