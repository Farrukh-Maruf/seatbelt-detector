
from __future__ import annotations

from collections import deque


class TemporalSmoother:
    def __init__(self, maxlen: int = 10, vote_threshold: float = 0.6):
        self.window: deque[int] = deque(maxlen=maxlen)
        self.vote_threshold = vote_threshold

    def update(self, is_violation: bool) -> bool:
        """Push this frame's raw decision (True = kamar_yoq) and return the smoothed,
        stable decision. Returns True (violation) only when the majority of the recent
        window agrees — so a single bad frame cannot cause a fine."""
        self.window.append(1 if is_violation else 0)
        return (sum(self.window) / len(self.window)) > self.vote_threshold

    def reset(self):
        self.window.clear()
