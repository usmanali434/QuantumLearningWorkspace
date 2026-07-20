"""Per-request timing for retrieval, LLM, and grounding phases."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rag.timing")


@dataclass
class TimingRecord:
    retrieval_ms: float | None = None
    llm_ms: float | None = None
    grounding_ms: float | None = None
    total_ms: float | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)
    _retrieval_start: float | None = field(default=None, repr=False)
    _llm_start: float | None = field(default=None, repr=False)
    _grounding_start: float | None = field(default=None, repr=False)

    def start_retrieval(self) -> None:
        self._retrieval_start = time.perf_counter()

    def end_retrieval(self) -> None:
        if self._retrieval_start is not None:
            self.retrieval_ms = (time.perf_counter() - self._retrieval_start) * 1000

    def start_llm(self) -> None:
        self._llm_start = time.perf_counter()

    def end_llm(self) -> None:
        if self._llm_start is not None:
            self.llm_ms = (time.perf_counter() - self._llm_start) * 1000

    def start_grounding(self) -> None:
        self._grounding_start = time.perf_counter()

    def end_grounding(self) -> None:
        if self._grounding_start is not None:
            self.grounding_ms = (time.perf_counter() - self._grounding_start) * 1000

    def finish(self) -> None:
        self.total_ms = (time.perf_counter() - self._start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1) if self.retrieval_ms is not None else None,
            "llm_ms": round(self.llm_ms, 1) if self.llm_ms is not None else None,
            "grounding_ms": round(self.grounding_ms, 1) if self.grounding_ms is not None else None,
            "total_ms": round(self.total_ms, 1) if self.total_ms is not None else None,
        }

    def log(self, *, user: str = "", cached: bool = False) -> None:
        self.finish()
        logger.info(
            "rag.timing user=%s cached=%s retrieval_ms=%s llm_ms=%s grounding_ms=%s total_ms=%s",
            user or "-",
            cached,
            self.retrieval_ms,
            self.llm_ms,
            self.grounding_ms,
            self.total_ms,
        )
