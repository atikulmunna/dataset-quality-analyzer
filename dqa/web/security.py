from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol


class RateLimitCounter(Protocol):
    def consume(self, key: str, *, window_start: int, limit: int) -> bool: ...


@dataclass(frozen=True)
class RateLimitPolicy:
    requests: int
    window_seconds: int


class FixedWindowRateLimiter:
    """Per-owner route limits backed by an atomic external counter."""

    def __init__(
        self,
        counter: RateLimitCounter,
        *,
        policies: dict[str, RateLimitPolicy] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._counter = counter
        self._policies = policies or {
            "POST jobs": RateLimitPolicy(requests=5, window_seconds=60),
            "POST uploads": RateLimitPolicy(requests=5, window_seconds=60),
            "GET jobs": RateLimitPolicy(requests=60, window_seconds=60),
            "DELETE jobs": RateLimitPolicy(requests=5, window_seconds=60),
        }
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def allow(self, owner_id: str, action: str) -> bool:
        policy = self._policies.get(action)
        if policy is None:
            return False
        now = int(self._clock().timestamp())
        window_start = now - (now % policy.window_seconds)
        return self._counter.consume(
            f"{owner_id}:{action}",
            window_start=window_start,
            limit=policy.requests,
        )
