from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FmmWorkspace:
    t_field: np.ndarray | None = None
    accepted: np.ndarray | None = None

    def ensure_shape(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self.t_field is None or self.t_field.shape != shape:
            self.t_field = np.empty(shape, dtype=float)
        if self.accepted is None or self.accepted.shape != shape:
            self.accepted = np.empty(shape, dtype=bool)
        return self.t_field, self.accepted


def _axis_neighbor_min(
    t_field: np.ndarray,
    accepted: np.ndarray,
    blocked: np.ndarray,
    r: int,
    c: int,
    axis: int,
) -> float:
    h, w = t_field.shape
    best = float("inf")
    if axis == 0:
        candidates = ((r - 1, c), (r + 1, c))
    else:
        candidates = ((r, c - 1), (r, c + 1))
    for rr, cc in candidates:
        if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and accepted[rr, cc]:
            val = float(t_field[rr, cc])
            if val < best:
                best = val
    return best


def _upwind_update(
    t_field: np.ndarray,
    accepted: np.ndarray,
    blocked: np.ndarray,
    w_cost: np.ndarray,
    r: int,
    c: int,
    resolution_m: float,
) -> float:
    if blocked[r, c]:
        return float("inf")

    a = _axis_neighbor_min(t_field, accepted, blocked, r, c, axis=1)
    b = _axis_neighbor_min(t_field, accepted, blocked, r, c, axis=0)
    f = float(w_cost[r, c]) * resolution_m

    if not math.isfinite(a) and not math.isfinite(b):
        return float("inf")
    if not math.isfinite(a):
        return b + f
    if not math.isfinite(b):
        return a + f

    if abs(a - b) >= f:
        return min(a, b) + f

    disc = 2.0 * (f ** 2) - (a - b) ** 2
    if disc < 0:
        return min(a, b) + f
    return 0.5 * (a + b + math.sqrt(disc))


def compute_cost_to_go_fmm(
    cost_density: np.ndarray,
    goal_rc: tuple[int, int],
    blocked: np.ndarray,
    resolution_m: float,
    workspace: FmmWorkspace | None = None,
) -> np.ndarray:
    h, w = cost_density.shape
    gr, gc = goal_rc
    if blocked[gr, gc]:
        raise ValueError("Goal is blocked.")

    if workspace is None:
        t_field = np.full((h, w), np.inf, dtype=float)
        accepted = np.zeros((h, w), dtype=bool)
    else:
        t_field, accepted = workspace.ensure_shape((h, w))
        t_field.fill(np.inf)
        accepted.fill(False)
    tie = itertools.count()
    heap: list[tuple[float, int, int, int]] = []

    t_field[gr, gc] = 0.0
    accepted[gr, gc] = True

    for rr, cc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
        if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc]:
            val = _upwind_update(
                t_field=t_field,
                accepted=accepted,
                blocked=blocked,
                w_cost=cost_density,
                r=rr,
                c=cc,
                resolution_m=resolution_m,
            )
            if val < t_field[rr, cc]:
                t_field[rr, cc] = val
                heapq.heappush(heap, (val, next(tie), rr, cc))

    while heap:
        _, _, r, c = heapq.heappop(heap)
        if accepted[r, c]:
            continue

        accepted[r, c] = True
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not accepted[rr, cc]:
                val = _upwind_update(
                    t_field=t_field,
                    accepted=accepted,
                    blocked=blocked,
                    w_cost=cost_density,
                    r=rr,
                    c=cc,
                    resolution_m=resolution_m,
                )
                if val < t_field[rr, cc]:
                    t_field[rr, cc] = val
                    heapq.heappush(heap, (val, next(tie), rr, cc))

    return t_field
