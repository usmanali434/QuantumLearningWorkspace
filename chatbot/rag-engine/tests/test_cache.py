"""Unit tests for answer cache."""

from __future__ import annotations

import sys
import time
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from cache import AnswerCache, CacheEntry  # noqa: E402


def test_cache_key_differs_with_include_sources():
    k1 = AnswerCache.make_key("Q?", None, 4, True, True, True)
    k2 = AnswerCache.make_key("Q?", None, 4, True, True, False)
    assert k1 != k2


def test_cache_key_differs_with_history():
    k1 = AnswerCache.make_key("Q?", None, 4, True, True)
    k2 = AnswerCache.make_key(
        "Q?",
        [{"role": "user", "content": "hi"}],
        4,
        True,
        True,
    )
    assert k1 != k2


def test_cache_get_set_and_hit():
    cache = AnswerCache(max_entries=10, ttl_seconds=60, enabled=True)
    key = "abc"
    entry = CacheEntry(
        answer="hello",
        refused=False,
        top_k=4,
        grounded=True,
        source_ids=["pdf_chunk_0"],
    )
    cache.set(key, entry)
    got = cache.get(key)
    assert got is not None
    assert got.answer == "hello"
    assert cache.hits == 1


def test_cache_skips_refused_and_ungrounded():
    cache = AnswerCache(enabled=True)
    cache.set("r", CacheEntry(answer="no", refused=True, top_k=4))
    cache.set("u", CacheEntry(answer="bad", refused=False, top_k=4, grounded=False))
    assert cache.get("r") is None
    assert cache.get("u") is None


def test_cache_ttl_expiry():
    cache = AnswerCache(max_entries=10, ttl_seconds=1, enabled=True)
    cache.set("x", CacheEntry(answer="old", refused=False, top_k=4, grounded=True))
    time.sleep(1.1)
    assert cache.get("x") is None


def test_cache_lru_eviction():
    cache = AnswerCache(max_entries=2, ttl_seconds=60, enabled=True)
    cache.set("a", CacheEntry(answer="a", refused=False, top_k=4, grounded=True))
    cache.set("b", CacheEntry(answer="b", refused=False, top_k=4, grounded=True))
    cache.get("a")
    cache.set("c", CacheEntry(answer="c", refused=False, top_k=4, grounded=True))
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
