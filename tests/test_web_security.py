from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dqa.web.security import FixedWindowRateLimiter


class MemoryCounter:
    def __init__(self) -> None:
        self.values: dict[tuple[str, int], int] = {}

    def consume(self, key: str, *, window_start: int, limit: int) -> bool:
        counter_key = (key, window_start)
        value = self.values.get(counter_key, 0)
        if value >= limit:
            return False
        self.values[counter_key] = value + 1
        return True


def test_default_submission_and_status_limits_are_per_owner() -> None:
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    limiter = FixedWindowRateLimiter(MemoryCounter(), clock=lambda: now)

    assert [limiter.allow("user-1", "POST jobs") for _ in range(6)] == [True] * 5 + [False]
    assert limiter.allow("user-2", "POST jobs") is True
    assert [limiter.allow("user-1", "GET jobs") for _ in range(61)] == [True] * 60 + [False]


def test_rate_limit_resets_in_next_window() -> None:
    current = [datetime(2026, 7, 12, tzinfo=timezone.utc)]
    limiter = FixedWindowRateLimiter(MemoryCounter(), clock=lambda: current[0])
    for _ in range(5):
        assert limiter.allow("user-1", "POST jobs")
    assert not limiter.allow("user-1", "POST jobs")

    current[0] += timedelta(seconds=60)

    assert limiter.allow("user-1", "POST jobs")


def test_unknown_action_fails_closed() -> None:
    limiter = FixedWindowRateLimiter(MemoryCounter())

    assert not limiter.allow("user-1", "DELETE admin")
