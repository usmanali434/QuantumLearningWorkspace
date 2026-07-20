"""Unit tests for rate limiter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rate_limiter import RateLimiter  # noqa: E402


def _mock_request(user_id: str = "", host: str = "127.0.0.1"):
    req = MagicMock()
    req.headers = {"X-User-Id": user_id} if user_id else {}
    req.client = MagicMock()
    req.client.host = host
    return req


def test_rate_limit_allows_under_max():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    req = _mock_request(user_id="alice")
    for _ in range(3):
        limiter.check(req)


def test_rate_limit_blocks_over_max():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    req = _mock_request(user_id="bob")
    limiter.check(req)
    limiter.check(req)
    with pytest.raises(HTTPException) as exc:
        limiter.check(req)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_rate_limit_separate_users():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check(_mock_request(user_id="u1"))
    limiter.check(_mock_request(user_id="u2"))
