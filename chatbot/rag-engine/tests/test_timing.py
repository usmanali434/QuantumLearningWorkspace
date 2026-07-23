"""Unit tests for timing logger."""

from __future__ import annotations

import sys
import time
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from timing_logger import TimingRecord  # noqa: E402


def test_timing_record_phases():
    t = TimingRecord()
    t.start_retrieval()
    time.sleep(0.01)
    t.end_retrieval()
    t.start_llm()
    time.sleep(0.01)
    t.end_llm()
    t.start_grounding()
    time.sleep(0.01)
    t.end_grounding()
    t.finish()
    d = t.to_dict()
    assert d["retrieval_ms"] is not None and d["retrieval_ms"] >= 5
    assert d["llm_ms"] is not None and d["llm_ms"] >= 5
    assert d["grounding_ms"] is not None and d["grounding_ms"] >= 5
    assert d["total_ms"] is not None and d["total_ms"] >= 15
