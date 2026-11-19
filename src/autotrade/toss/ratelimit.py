from __future__ import annotations

import time


class RateLimiter:
    """Per-group minimum spacing (1/rate seconds) between calls."""

    def __init__(self, rates: dict[str, float]):
        self._interval = {g: 1.0 / r for g, r in rates.items()}
        self._next_ok: dict[str, float] = {}

    def acquire(self, group: str) -> None:
        interval = self._interval[group]  # KeyError on unknown group is intentional
        now = time.monotonic()
        wait = self._next_ok.get(group, now) - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        self._next_ok[group] = now + interval
