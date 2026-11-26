"""Rate limiting for POST /ask, which spends the owner's LLM API credit.

A public demo with a paid key needs a cap: per client (sliding one-hour window)
and a global daily budget. In memory, which fits the single-worker deployment;
counters reset on restart, which is acceptable for a demo.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from app.config import get_settings

HOUR_S = 3600.0
DAY_S = 86400.0


class RateLimiter:
    """Sliding-window limits: `per_client_per_hour` and `global_per_day` (0 = unlimited)."""

    def __init__(self, per_client_per_hour: int, global_per_day: int):
        self.per_client = per_client_per_hour
        self.global_per_day = global_per_day
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> str | None:
        """Record a request; return a reason string if it must be refused."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._clients[client]
            while hits and now - hits[0] >= HOUR_S:
                hits.popleft()
            while self._all and now - self._all[0] >= DAY_S:
                self._all.popleft()
            if self.global_per_day and len(self._all) >= self.global_per_day:
                return "The demo's daily question budget is used up. Please try again tomorrow."
            if self.per_client and len(hits) >= self.per_client:
                return "Too many questions from you in the last hour. Please try again later."
            hits.append(now)
            self._all.append(now)
            return None


_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    """Process-wide limiter built from settings."""
    global _limiter
    if _limiter is None:
        s = get_settings()
        _limiter = RateLimiter(s.ask_limit_per_client_per_hour, s.ask_limit_per_day)
    return _limiter


def client_key(request: Request) -> str:
    """Client identity: first X-Forwarded-For hop (set by the host's proxy), else peer IP."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def limit_ask(request: Request) -> None:
    """FastAPI dependency: raise 429 when a limit is exceeded."""
    reason = get_limiter().check(client_key(request))
    if reason:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, reason)
