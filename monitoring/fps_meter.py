from __future__ import annotations

import collections
import threading
import time
from typing import Deque, Dict, List, Tuple

import numpy as np


class FPSMeter:
    """
    Sliding window FPS/latency estimator with percentiles.
    """

    def __init__(self, window: int = 200):
        self.window = window
        self.intervals: Deque[float] = collections.deque(maxlen=window)  # seconds between ticks
        self.latencies_ms: Deque[float] = collections.deque(maxlen=window)
        self.last_tick: float | None = None
        self._lock = threading.Lock()

    def _snapshot(self) -> Tuple[List[float], List[float], float | None]:
        with self._lock:
            return list(self.intervals), list(self.latencies_ms), self.last_tick

    def tick(self) -> None:
        """Call once per frame to update FPS intervals."""
        now = time.perf_counter()
        with self._lock:
            if self.last_tick is not None:
                self.intervals.append(now - self.last_tick)
            self.last_tick = now

    def observe(self, latency_ms: float) -> None:
        """Record an inference latency (ms)."""
        with self._lock:
            self.latencies_ms.append(latency_ms)

    def fps(self) -> float:
        intervals, _, _ = self._snapshot()
        if not intervals:
            return 0.0
        avg = sum(intervals) / len(intervals)
        return 1.0 / avg if avg > 0 else 0.0

    def avg_latency_ms(self) -> float:
        _, latencies_ms, _ = self._snapshot()
        if not latencies_ms:
            return 0.0
        return float(np.mean(latencies_ms))

    def percentiles(self) -> Dict[str, float]:
        _, latencies_ms, _ = self._snapshot()
        if not latencies_ms:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
        arr = np.array(latencies_ms, dtype=np.float32)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    def to_dict(self) -> Dict[str, float]:
        intervals, latencies_ms, _ = self._snapshot()

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            fps = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            fps = 0.0

        if latencies_ms:
            arr = np.array(latencies_ms, dtype=np.float32)
            avg_latency_ms = float(np.mean(arr))
            pct = {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
            }
        else:
            avg_latency_ms = 0.0
            pct = {"p50": 0.0, "p90": 0.0, "p95": 0.0}

        return {"fps": fps, "avg_latency_ms": avg_latency_ms, **pct}



