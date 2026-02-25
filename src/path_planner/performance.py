from __future__ import annotations

import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class StageTimer:
    enabled: bool = True
    _stage_starts_ns: dict[str, int] = field(default_factory=dict)
    stage_ms: dict[str, float] = field(default_factory=dict)
    _total_start_ns: int = 0
    total_ms: float = 0.0

    def start_total(self) -> None:
        if not self.enabled:
            return
        self._total_start_ns = time.perf_counter_ns()

    def stop_total(self) -> float:
        if not self.enabled:
            return 0.0
        if self._total_start_ns <= 0:
            return self.total_ms
        elapsed = (time.perf_counter_ns() - self._total_start_ns) / 1_000_000.0
        self.total_ms = float(elapsed)
        return self.total_ms

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        self._stage_starts_ns[name] = time.perf_counter_ns()

    def stop(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        start = self._stage_starts_ns.pop(name, None)
        if start is None:
            return float(self.stage_ms.get(name, 0.0))
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        self.stage_ms[name] = float(self.stage_ms.get(name, 0.0) + elapsed)
        return float(self.stage_ms[name])

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def as_dict(self) -> dict[str, float]:
        out = dict(self.stage_ms)
        out["totalRuntimeMs"] = float(self.total_ms)
        return out


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if len(latencies_ms) == 0:
        return {
            "count": 0.0,
            "avgMs": 0.0,
            "medianMs": 0.0,
            "p95Ms": 0.0,
            "p99Ms": 0.0,
            "worstMs": 0.0,
            "throughputHz": 0.0,
        }

    arr = np.asarray(latencies_ms, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0.0,
            "avgMs": 0.0,
            "medianMs": 0.0,
            "p95Ms": 0.0,
            "p99Ms": 0.0,
            "worstMs": 0.0,
            "throughputHz": 0.0,
        }

    total_s = float(np.sum(arr) / 1000.0)
    throughput = float(arr.size / total_s) if total_s > 1e-12 else 0.0
    return {
        "count": float(arr.size),
        "avgMs": float(np.mean(arr)),
        "medianMs": float(np.median(arr)),
        "p95Ms": float(np.percentile(arr, 95)),
        "p99Ms": float(np.percentile(arr, 99)),
        "worstMs": float(np.max(arr)),
        "stdMs": float(statistics.pstdev(arr.tolist())) if arr.size > 1 else 0.0,
        "throughputHz": throughput,
    }


def aggregate_stage_stats(stage_rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    if len(stage_rows) == 0:
        return {}
    keys: set[str] = set()
    for row in stage_rows:
        keys.update(row.keys())
    out: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        vals = [float(row[key]) for row in stage_rows if key in row and math.isfinite(float(row[key]))]
        out[key] = latency_stats(vals)
    return out

