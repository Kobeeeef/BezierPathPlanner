from __future__ import annotations

import math

import numpy as np

from .config import PlannerConfig


def wrap_to_pi(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_delta(start_rad: float, goal_rad: float) -> float:
    return wrap_to_pi(goal_rad - start_rad)


def smoothstep01(t: float) -> float:
    tt = max(0.0, min(1.0, float(t)))
    return tt * tt * (3.0 - 2.0 * tt)


def progress_from_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.zeros(len(points), dtype=float)
    s[1:] = np.cumsum(ds)
    total = float(s[-1])
    if total <= 1e-9:
        return np.linspace(0.0, 1.0, len(points))
    return s / total


def _resolve_heading_rad(heading_deg: float | None, fallback_rad: float) -> float:
    if heading_deg is None:
        return float(fallback_rad)
    return float(math.radians(heading_deg))


def compute_holonomic_rotation_profile(
    path_tangent_headings_rad: np.ndarray,
    progress_u: np.ndarray,
    cfg: PlannerConfig,
) -> np.ndarray:
    if len(progress_u) == 0:
        return np.empty((0,), dtype=float)

    tangents = np.asarray(path_tangent_headings_rad, dtype=float)
    u = np.asarray(progress_u, dtype=float)

    if cfg.holonomic_rotation_mode == "tangent_follow":
        if len(tangents) == len(u):
            return tangents.copy()
        if len(tangents) == 0:
            return np.zeros_like(u, dtype=float)
        return np.full_like(u, float(tangents[-1]), dtype=float)

    tangent_start = float(tangents[0]) if len(tangents) else 0.0
    tangent_goal = float(tangents[-1]) if len(tangents) else tangent_start
    start_rad = _resolve_heading_rad(cfg.start_heading_deg, tangent_start)
    goal_rad = _resolve_heading_rad(cfg.end_heading_deg, tangent_goal)
    delta = shortest_angle_delta(start_rad, goal_rad)
    finish = max(1e-3, min(1.0, float(cfg.rotation_finish_progress)))

    clamped = np.clip(u / finish, 0.0, 1.0)
    eased = clamped * clamped * (3.0 - 2.0 * clamped)
    out = start_rad + delta * eased
    out = (out + math.pi) % (2.0 * math.pi) - math.pi
    out = np.asarray(out, dtype=float)
    out[u >= finish] = goal_rad
    return out
