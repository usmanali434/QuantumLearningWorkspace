"""Sliding-window rate limiter (memory or Redis)."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


DEFAULT_MAX_REQUESTS = 10
DEFAULT_WINDOW_SECONDS = 60
RATE_KEY_PREFIX = "studymind:ratelimit:"


class RateLimiter:
    """Per-user sliding window rate limiter."""

    def __init__(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
        redis_url: str | None = None,
    ) -> None:
        self.max_requests = max_requests or _env_int(
            "RATE_LIMIT_MAX", DEFAULT_MAX_REQUESTS
        )
        self.window_seconds = window_seconds or _env_int(
            "RATE_LIMIT_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS
        )
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._redis = None
        self.backend = "memory"
        url = redis_url if redis_url is not None else os.environ.get("REDIS_URL", "").strip()
        if url:
            try:
                import redis

                self._redis = redis.Redis.from_url(url, decode_responses=True)
                self._redis.ping()
                self.backend = "redis"
            except Exception:
                self._redis = None
                self.backend = "memory"

    def _client_key(self, request: Request) -> str:
        user_id = request.headers.get("X-User-Id", "").strip()
        if user_id:
            return f"user:{user_id}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def check(self, request: Request) -> None:
        key = self._client_key(request)
        if self._redis is not None:
            self._check_redis(key)
            return
        now = time.time()
        window_start = now - self.window_seconds
        hits = self._hits[key]
        while hits and hits[0] <= window_start:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)

    def _check_redis(self, key: str) -> None:
        assert self._redis is not None
        redis_key = f"{RATE_KEY_PREFIX}{key}"
        now = time.time()
        window_start = now - self.window_seconds
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(redis_key, 0, window_start)
        pipe.zadd(redis_key, {str(now): now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, self.window_seconds)
        _, _, count, _ = pipe.execute()
        if int(count) > self.max_requests:
            oldest = self._redis.zrange(redis_key, 0, 0, withscores=True)
            retry_after = self.window_seconds
            if oldest:
                retry_after = max(1, int(self.window_seconds - (now - oldest[0][1])))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )


rate_limiter = RateLimiter()


def check_rate_limit(request: Request) -> None:
    """FastAPI dependency: enforce per-user rate limits."""
    rate_limiter.check(request)
