from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DijkstraWorkspace:
    dist: np.ndarray | None = None

    def ensure_shape(self, shape: tuple[int, int]) -> np.ndarray:
        if self.dist is None or self.dist.shape != shape:
            self.dist = np.empty(shape, dtype=float)
        return self.dist


def compute_cost_to_go_dijkstra(
    cost_density: np.ndarray,
    goal_rc: tuple[int, int],
    blocked: np.ndarray,
    resolution_m: float,
    workspace: DijkstraWorkspace | None = None,
) -> np.ndarray:
    h, w = cost_density.shape
    gr, gc = goal_rc
    if blocked[gr, gc]:
        raise ValueError("Goal is blocked.")

    if workspace is None:
        dist = np.full((h, w), np.inf, dtype=float)
    else:
        dist = workspace.ensure_shape((h, w))
        dist.fill(np.inf)
    dist[gr, gc] = 0.0

    tie = itertools.count()
    heap: list[tuple[float, int, int, int]] = [(0.0, next(tie), gr, gc)]

    neighbors: list[tuple[int, int, float]] = [
        (-1, 0, resolution_m),
        (1, 0, resolution_m),
        (0, -1, resolution_m),
        (0, 1, resolution_m),
        (-1, -1, resolution_m * math.sqrt(2.0)),
        (-1, 1, resolution_m * math.sqrt(2.0)),
        (1, -1, resolution_m * math.sqrt(2.0)),
        (1, 1, resolution_m * math.sqrt(2.0)),
    ]

    while heap:
        cur, _, r, c = heapq.heappop(heap)
        if cur > dist[r, c]:
            continue

        for dr, dc, edge_len in neighbors:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            edge_cost = 0.5 * (float(cost_density[r, c]) + float(cost_density[rr, cc])) * edge_len
            candidate = cur + edge_cost
            if candidate < dist[rr, cc]:
                dist[rr, cc] = candidate
                heapq.heappush(heap, (candidate, next(tie), rr, cc))

    return dist
