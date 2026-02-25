from __future__ import annotations

import math

import numpy as np

from .config import PlannerConfig


def blocked_from_inputs(
    heat: np.ndarray,
    blocked_mask: np.ndarray | None,
    blocked_sentinel: float | None,
) -> np.ndarray:
    blocked = np.zeros_like(heat, dtype=bool)
    if blocked_mask is not None:
        if blocked_mask.shape != heat.shape:
            raise ValueError("blocked_mask shape must match heat shape.")
        blocked |= blocked_mask.astype(bool)

    if blocked_sentinel is not None:
        if math.isnan(blocked_sentinel):
            blocked |= np.isnan(heat)
        else:
            blocked |= np.isclose(heat, blocked_sentinel, atol=0.0, rtol=0.0)

    return blocked


def build_cost_density(
    heat: np.ndarray,
    cfg: PlannerConfig,
    blocked_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if heat.ndim != 2:
        raise ValueError("heat must be a 2D array.")

    heat_f = np.asarray(heat, dtype=float)
    blocked = blocked_from_inputs(heat_f, blocked_mask, cfg.blocked_sentinel)

    traversable = ~blocked
    if not np.any(traversable):
        raise ValueError("No traversable cells available.")

    trav_vals = heat_f[traversable]
    if not np.all(np.isfinite(trav_vals)):
        raise ValueError(
            "All non-blocked heat cells must be finite. Use blocked mask/sentinel for obstacles."
        )
    if np.any(trav_vals <= 0.0):
        raise ValueError("Heat values must be positive for all traversable cells.")

    if cfg.cost_mode == "density":
        w = cfg.base_cost + cfg.alpha * heat_f
    elif cfg.cost_mode == "inverse_speed":
        speed = 1.0 / (cfg.epsilon + heat_f)
        w = cfg.base_cost + cfg.alpha * (1.0 / speed)
    else:
        raise ValueError(f"Unsupported cost_mode: {cfg.cost_mode}")

    w = np.asarray(w, dtype=float)
    w[traversable] = np.maximum(w[traversable], 1e-6)
    w[blocked] = np.inf
    return w, blocked


def world_to_cell_index(
    x_m: float,
    y_m: float,
    resolution_m: float,
    shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    c = int(round(x_m / resolution_m))
    r = int(round(y_m / resolution_m))
    if r < 0 or c < 0 or r >= h or c >= w:
        raise ValueError(
            f"Point ({x_m:.3f}, {y_m:.3f}) is outside map bounds "
            f"[0,{w * resolution_m:.3f}] x [0,{h * resolution_m:.3f}]."
        )
    return r, c

