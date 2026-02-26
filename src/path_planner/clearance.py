from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import PlannerConfig
from .geometry import bilinear_sample_grid_vectorized


@dataclass(frozen=True)
class ClearanceLayers:
    required_clearance_m: float
    wall_clearance_m: np.ndarray
    heat_region_clearance_m: np.ndarray
    heat_region_mask: np.ndarray
    heat_region_threshold: float | None
    wall_penalty: np.ndarray
    heat_region_penalty: np.ndarray
    combined_penalty: np.ndarray
    planning_blocked: np.ndarray
    hard_clearance_feasible: bool


@dataclass(frozen=True)
class ClearanceBase:
    required_clearance_m: float
    wall_clearance_m: np.ndarray
    heat_region_clearance_m: np.ndarray
    heat_region_mask: np.ndarray
    heat_region_threshold: float | None
    wall_penalty: np.ndarray
    heat_region_penalty: np.ndarray
    traversable_mask: np.ndarray
    feasible_mask: np.ndarray


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected(
    traversable: np.ndarray,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> bool:
    h, w = traversable.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return False
    if not traversable[sr, sc] or not traversable[gr, gc]:
        return False
    if start_rc == goal_rc:
        return True

    seen = np.zeros_like(traversable, dtype=bool)
    q: deque[tuple[int, int]] = deque([start_rc])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if rr < 0 or cc < 0 or rr >= h or cc >= w:
                continue
            if seen[rr, cc] or not traversable[rr, cc]:
                continue
            if (rr, cc) == goal_rc:
                return True
            seen[rr, cc] = True
            q.append((rr, cc))
    return False


def _geometry_blocked_mask(blocked: np.ndarray) -> np.ndarray:
    geo = np.asarray(blocked, dtype=bool).copy()
    if geo.size == 0:
        return geo
    geo[0, :] = True
    geo[-1, :] = True
    geo[:, 0] = True
    geo[:, -1] = True
    return geo


def compute_wall_clearance_field_m(
    blocked: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    geo_blocked = _geometry_blocked_mask(blocked)
    traversable = ~geo_blocked
    dist_cells = distance_transform_edt(traversable)
    return np.asarray(dist_cells * resolution_m, dtype=float)


def derive_heat_region_mask(
    heat: np.ndarray,
    blocked: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[np.ndarray, float | None]:
    if not cfg.heat_region_clearance_enabled:
        return np.zeros_like(blocked, dtype=bool), None

    traversable = ~blocked
    vals = np.asarray(heat[traversable], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(blocked, dtype=bool), None

    if cfg.heat_region_threshold is not None:
        threshold = float(cfg.heat_region_threshold)
    else:
        q = float(np.clip(cfg.heat_region_quantile, 0.50, 0.995))
        threshold = float(np.quantile(vals, q))

    mask = np.zeros_like(blocked, dtype=bool)
    mask[traversable] = np.asarray(heat[traversable] >= threshold, dtype=bool)
    return mask, threshold


def compute_heat_region_clearance_field_m(
    heat_region_mask: np.ndarray,
    blocked: np.ndarray,
    resolution_m: float,
    required_clearance_m: float = 0.0,
) -> np.ndarray:
    if not np.any(heat_region_mask):
        return np.full_like(heat_region_mask, np.inf, dtype=float)

    dist_cells = distance_transform_edt(~heat_region_mask)
    out = np.asarray(dist_cells * resolution_m, dtype=float)
    if required_clearance_m > 0.0:
        np.maximum(0.0, out - float(required_clearance_m), out=out)
    out[blocked] = 0.0
    return out


def build_clearance_layers(
    heat: np.ndarray,
    blocked: np.ndarray,
    cfg: PlannerConfig,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> ClearanceLayers:
    base = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg)
    return clearance_layers_from_base(
        base=base,
        blocked=blocked,
        cfg=cfg,
        start_rc=start_rc,
        goal_rc=goal_rc,
    )


def precompute_clearance_base(
    heat: np.ndarray,
    blocked: np.ndarray,
    cfg: PlannerConfig,
) -> ClearanceBase:
    required = max(0.0, float(cfg.required_clearance_m))
    wall_clearance = compute_wall_clearance_field_m(blocked, cfg.resolution_m_per_cell)
    traversable = ~blocked

    soft_target = max(required * cfg.wall_clearance_soft_ratio, required + 1e-6)
    wall_deficit = np.maximum(0.0, soft_target - wall_clearance)
    wall_penalty = cfg.wall_clearance_weight * np.power(
        wall_deficit / max(soft_target, 1e-6),
        max(1.0, float(cfg.wall_clearance_power)),
    )
    wall_penalty[blocked] = 0.0

    heat_region_mask, heat_thresh = derive_heat_region_mask(heat, blocked, cfg)
    heat_region_clearance = compute_heat_region_clearance_field_m(
        heat_region_mask,
        blocked,
        cfg.resolution_m_per_cell,
        required_clearance_m=required,
    )
    heat_decay = max(1e-6, float(cfg.heat_region_clearance_decay_m))
    heat_region_penalty = np.zeros_like(heat, dtype=float)
    if cfg.heat_region_clearance_enabled and cfg.heat_region_clearance_weight > 0.0:
        heat_region_penalty[traversable] = cfg.heat_region_clearance_weight * np.exp(
            -heat_region_clearance[traversable] / heat_decay
        )

    feasible_mask = traversable & (wall_clearance >= required - 1e-9)
    return ClearanceBase(
        required_clearance_m=required,
        wall_clearance_m=wall_clearance,
        heat_region_clearance_m=heat_region_clearance,
        heat_region_mask=heat_region_mask,
        heat_region_threshold=heat_thresh,
        wall_penalty=wall_penalty,
        heat_region_penalty=heat_region_penalty,
        traversable_mask=traversable,
        feasible_mask=feasible_mask,
    )


def clearance_layers_from_base(
    base: ClearanceBase,
    blocked: np.ndarray,
    cfg: PlannerConfig,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> ClearanceLayers:
    feasible_mask = base.feasible_mask
    hard_feasible = bool(cfg.enable_clearance_constraints) and _connected(feasible_mask, start_rc, goal_rc)
    planning_blocked = blocked.copy()
    # Wall clearance is always a hard constraint; heat remains a soft penalty.
    planning_blocked |= base.wall_clearance_m < base.required_clearance_m
    planning_blocked[start_rc] = False
    planning_blocked[goal_rc] = False
    combined = base.wall_penalty + base.heat_region_penalty
    combined[planning_blocked] = 0.0

    return ClearanceLayers(
        required_clearance_m=base.required_clearance_m,
        wall_clearance_m=base.wall_clearance_m,
        heat_region_clearance_m=base.heat_region_clearance_m,
        heat_region_mask=base.heat_region_mask,
        heat_region_threshold=base.heat_region_threshold,
        wall_penalty=base.wall_penalty,
        heat_region_penalty=base.heat_region_penalty,
        combined_penalty=combined,
        planning_blocked=planning_blocked,
        hard_clearance_feasible=hard_feasible,
    )


def sample_field_along_path(
    points_world: np.ndarray,
    field: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    if len(points_world) == 0:
        return np.empty((0,), dtype=float)
    points = np.asarray(points_world, dtype=float)
    x_idx = points[:, 0] / float(resolution_m)
    y_idx = points[:, 1] / float(resolution_m)
    return bilinear_sample_grid_vectorized(field, x_idx, y_idx)
