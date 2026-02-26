from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

from .clearance import sample_field_along_path
from .config import PlannerConfig
from .geometry import (
    angle_rad,
    bezier_point,
    bilinear_sample_grid_vectorized,
    cumulative_arc_length,
    dedupe_consecutive,
    heading_to_unit,
    max_curvature,
    normalize,
    polyline_length,
    resample_polyline,
    sample_bezier_chain,
)
from .models import BezierSegment


@dataclass
class SmoothingResult:
    segments: list[BezierSegment]
    anchors: np.ndarray
    tangent_vectors: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SmoothingContext:
    start_xy: np.ndarray
    goal_xy: np.ndarray
    wall_clearance_field: np.ndarray | None = None
    heat_region_clearance_field: np.ndarray | None = None
    cost_density_field: np.ndarray | None = None
    required_clearance_m: float = 0.0
    hard_clearance_feasible: bool = False


def _wrap_to_pi(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _angle_error_deg(a_rad: float, b_rad: float) -> float:
    return abs(math.degrees(_wrap_to_pi(a_rad - b_rad)))


def _angle_error_to_heading_deg(vec: np.ndarray, heading_deg: float | None) -> float:
    if heading_deg is None or np.linalg.norm(vec) <= 1e-12:
        return 0.0
    return _angle_error_deg(angle_rad(vec), math.radians(float(heading_deg)))


def _resolved_start_approach_heading_deg(cfg: PlannerConfig) -> float | None:
    return cfg.resolved_start_approach_heading_deg


def _resolved_goal_approach_heading_deg(cfg: PlannerConfig) -> float | None:
    return cfg.resolved_goal_approach_heading_deg


def _resolved_start_dir(cfg: PlannerConfig, fallback: np.ndarray) -> np.ndarray:
    heading = _resolved_start_approach_heading_deg(cfg)
    if heading is None:
        d = normalize(fallback)
    else:
        d = heading_to_unit(heading)
    if np.linalg.norm(d) <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return d


def _resolved_goal_dir(cfg: PlannerConfig, fallback: np.ndarray) -> np.ndarray:
    heading = _resolved_goal_approach_heading_deg(cfg)
    if heading is None:
        d = normalize(fallback)
    else:
        d = heading_to_unit(heading)
    if np.linalg.norm(d) <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return d


def _segment_intersection(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    eps: float = 1e-9,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_seg(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    if abs(o1) <= eps and on_seg(a0, b0, a1):
        return True
    if abs(o2) <= eps and on_seg(a0, b1, a1):
        return True
    if abs(o3) <= eps and on_seg(b0, a0, b1):
        return True
    if abs(o4) <= eps and on_seg(b0, a1, b1):
        return True
    return False


def _self_intersection_counts(points: np.ndarray, endpoint_zone_m: float) -> tuple[int, int]:
    if len(points) < 4:
        return 0, 0
    s = cumulative_arc_length(points)
    total_len = float(s[-1]) if len(s) else 0.0
    endpoint_zone = max(0.0, min(float(endpoint_zone_m), total_len))
    total = 0
    endpoint = 0
    seg_count = len(points) - 1
    for i in range(seg_count):
        a0 = points[i]
        a1 = points[i + 1]
        for j in range(i + 2, seg_count):
            if j == i + 1:
                continue
            if i == 0 and j == seg_count - 1:
                continue
            b0 = points[j]
            b1 = points[j + 1]
            if not _segment_intersection(a0, a1, b0, b1):
                continue
            total += 1
            si = float(s[i])
            sj = float(s[j])
            if (
                si <= endpoint_zone
                or sj <= endpoint_zone
                or si >= total_len - endpoint_zone
                or sj >= total_len - endpoint_zone
            ):
                endpoint += 1
    return total, endpoint
def _angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    d = float(np.dot(a, b) / (na * nb))
    d = max(-1.0, min(1.0, d))
    return float(math.degrees(math.acos(d)))


def _curvature_samples_for_polyline(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    s = cumulative_arc_length(points)
    out_k: list[float] = []
    out_s: list[float] = []
    for i in range(1, len(points) - 1):
        a = points[i - 1]
        b = points[i]
        c = points[i + 1]
        ab = b - a
        bc = c - b
        ac = c - a
        lab = float(np.linalg.norm(ab))
        lbc = float(np.linalg.norm(bc))
        lac = float(np.linalg.norm(ac))
        denom = lab * lbc * lac
        if denom <= 1e-12:
            continue
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        k = 2.0 * abs(cross) / denom
        if math.isfinite(k):
            out_k.append(float(k))
            out_s.append(float(s[i]))
    return np.asarray(out_k, dtype=float), np.asarray(out_s, dtype=float)


def _tangent_jumps_for_polyline(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    segs = np.diff(points, axis=0)
    s = cumulative_arc_length(points)
    jumps = [_angle_between_deg(segs[i - 1], segs[i]) for i in range(1, len(segs))]
    jump_s = [float(s[i]) for i in range(1, len(segs))]
    return np.asarray(jumps, dtype=float), np.asarray(jump_s, dtype=float)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _zone_max(
    values: np.ndarray,
    positions: np.ndarray,
    total_len: float,
    zone_m: float,
    from_start: bool,
) -> float:
    if values.size == 0 or positions.size == 0:
        return 0.0
    zone = max(0.0, min(float(zone_m), float(total_len)))
    if zone <= 1e-9:
        return 0.0
    if from_start:
        mask = positions <= zone
    else:
        mask = positions >= max(0.0, total_len - zone)
    if not np.any(mask):
        return 0.0
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else 0.0


def _terminal_zone_length(total_len: float, cfg: PlannerConfig, from_start: bool) -> float:
    if total_len <= 1e-9:
        return 0.0
    if from_start:
        base = max(
            float(cfg.endpoint_zone_m),
            float(cfg.start_approach_lock_distance_m),
            0.45 * float(cfg.terminal_progress_window_m),
        )
    else:
        base = max(
            float(cfg.endpoint_zone_m),
            float(cfg.goal_approach_lock_distance_m),
            float(cfg.terminal_progress_window_m),
        )
    return max(0.0, min(total_len, base))


def _terminal_zone_indices(
    s: np.ndarray,
    total_len: float,
    zone_m: float,
    from_start: bool,
) -> np.ndarray:
    n = len(s)
    if n == 0:
        return np.empty((0,), dtype=int)
    if zone_m <= 1e-9:
        if from_start:
            return np.arange(min(2, n), dtype=int)
        return np.arange(max(0, n - 2), n, dtype=int)

    if from_start:
        idx = np.flatnonzero(s <= zone_m + 1e-9)
        if idx.size < 2:
            idx = np.arange(min(3, n), dtype=int)
    else:
        idx = np.flatnonzero(s >= max(0.0, total_len - zone_m) - 1e-9)
        if idx.size < 2:
            idx = np.arange(max(0, n - 3), n, dtype=int)
    if idx.size == 0:
        if from_start:
            return np.arange(min(2, n), dtype=int)
        return np.arange(max(0, n - 2), n, dtype=int)
    return idx.astype(int, copy=False)


def _mean_inverse_clearance(clearance_samples: np.ndarray, floor: float = 0.05) -> float:
    finite = clearance_samples[np.isfinite(clearance_samples)]
    finite = finite[finite >= 0.0]
    if finite.size == 0:
        return 0.0
    return float(np.mean(1.0 / np.maximum(float(floor), finite)))


def _terminal_zone_metrics(
    points: np.ndarray,
    s: np.ndarray,
    curvatures: np.ndarray,
    curvature_s: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
    from_start: bool,
    heading_deg: float | None,
) -> dict[str, float]:
    if len(points) < 2:
        return {
            "zoneLengthM": 0.0,
            "arcLengthM": 0.0,
            "straightLengthM": 0.0,
            "directnessRatio": 1.0,
            "directnessPenalty": 0.0,
            "directnessScore": 1.0,
            "lateralDeviationMeanM": 0.0,
            "lateralDeviationMaxM": 0.0,
            "maxCurvature": 0.0,
            "projectionMonotonicViolations": 0.0,
            "distanceMonotonicViolations": 0.0,
            "overshootCount": 0.0,
            "progressRatio": 1.0,
            "hookOrOvershootFlag": 0.0,
            "alignmentErrorDeg": 0.0,
            "minWallClearanceM": 0.0,
            "minHeatRegionClearanceM": -1.0,
            "heatExposure": 0.0,
        }

    total_len = float(s[-1]) if len(s) else 0.0
    zone_m = _terminal_zone_length(total_len, cfg, from_start=from_start)
    idx = _terminal_zone_indices(s, total_len, zone_m, from_start=from_start)
    zone_pts = points[idx] if idx.size else points
    if len(zone_pts) < 2:
        zone_pts = points[:2] if from_start else points[-2:]

    zone_line = zone_pts[-1] - zone_pts[0]
    line_dir = normalize(zone_line)
    if np.linalg.norm(line_dir) <= 1e-12:
        if heading_deg is not None:
            line_dir = heading_to_unit(float(heading_deg))
        elif from_start and len(points) >= 2:
            line_dir = normalize(points[1] - points[0])
        elif not from_start and len(points) >= 2:
            line_dir = normalize(points[-1] - points[-2])
    if np.linalg.norm(line_dir) <= 1e-12:
        line_dir = np.array([1.0, 0.0], dtype=float)

    arc_len = polyline_length(zone_pts)
    straight_len = float(np.linalg.norm(zone_pts[-1] - zone_pts[0]))
    directness_ratio = arc_len / max(1e-6, straight_len)
    directness_penalty = max(0.0, directness_ratio - 1.0)

    rel_zone = zone_pts - zone_pts[0]
    lateral = np.abs(rel_zone[:, 0] * line_dir[1] - rel_zone[:, 1] * line_dir[0])
    lateral_mean = float(np.mean(lateral)) if lateral.size else 0.0
    lateral_max = float(np.max(lateral)) if lateral.size else 0.0

    tol = max(0.01, 0.15 * float(cfg.sample_ds_m))
    overshoot_tol = max(float(cfg.endpoint_overshoot_tolerance_m), 1e-6)
    if from_start:
        endpoint = zone_pts[0]
        rel_endpoint = zone_pts - endpoint
        progress = np.asarray(np.dot(rel_endpoint, line_dir), dtype=float)
        dist = np.linalg.norm(rel_endpoint, axis=1)
        proj_viols = int(np.sum(np.diff(progress) < -tol))
        dist_viols = int(np.sum(np.diff(dist) < -tol))
        overshoot = int(np.sum(progress[:-1] < -overshoot_tol)) if len(progress) > 1 else 0
        forward = float(progress[-1] - progress[0]) if len(progress) > 1 else 0.0
    else:
        endpoint = zone_pts[-1]
        rel_endpoint = endpoint[None, :] - zone_pts
        progress = np.asarray(np.dot(rel_endpoint, line_dir), dtype=float)
        dist = np.linalg.norm(rel_endpoint, axis=1)
        proj_viols = int(np.sum(np.diff(progress) > tol))
        dist_viols = int(np.sum(np.diff(dist) > tol))
        overshoot = int(np.sum(progress[:-1] < -overshoot_tol)) if len(progress) > 1 else 0
        forward = float(progress[0] - progress[-1]) if len(progress) > 1 else 0.0
    motion = float(np.sum(np.abs(np.diff(progress)))) if len(progress) > 1 else 0.0
    progress_ratio = 1.0 if motion <= 1e-9 else max(0.0, min(1.0, forward / motion))

    max_curv = _zone_max(
        curvatures,
        curvature_s,
        total_len,
        zone_m,
        from_start=from_start,
    )
    align_error = 0.0
    if heading_deg is not None and len(zone_pts) >= 2:
        vec = zone_pts[1] - zone_pts[0] if from_start else zone_pts[-1] - zone_pts[-2]
        align_error = _angle_error_to_heading_deg(vec, heading_deg)

    wall_min = 0.0
    if context is not None and context.wall_clearance_field is not None:
        wall_samples = sample_field_along_path(zone_pts, context.wall_clearance_field, cfg.resolution_m_per_cell)
        wall_min = _safe_stat_min(wall_samples, default=0.0)

    heat_min = -1.0
    heat_exposure = 0.0
    if context is not None and context.heat_region_clearance_field is not None:
        heat_samples = sample_field_along_path(
            zone_pts,
            context.heat_region_clearance_field,
            cfg.resolution_m_per_cell,
        )
        heat_min = _safe_stat_min(heat_samples, default=-1.0)
        heat_exposure = _mean_inverse_clearance(heat_samples)

    directness_score = 1.0 / (1.0 + directness_penalty + lateral_mean / max(0.2, straight_len))
    hook_flag = float(
        overshoot > 0
        or proj_viols > 0
        or dist_viols > 0
        or progress_ratio < float(cfg.min_terminal_progress_ratio)
    )
    return {
        "zoneLengthM": float(zone_m),
        "arcLengthM": float(arc_len),
        "straightLengthM": float(straight_len),
        "directnessRatio": float(directness_ratio),
        "directnessPenalty": float(directness_penalty),
        "directnessScore": float(directness_score),
        "lateralDeviationMeanM": float(lateral_mean),
        "lateralDeviationMaxM": float(lateral_max),
        "maxCurvature": float(max_curv),
        "projectionMonotonicViolations": float(proj_viols),
        "distanceMonotonicViolations": float(dist_viols),
        "overshootCount": float(overshoot),
        "progressRatio": float(progress_ratio),
        "hookOrOvershootFlag": float(hook_flag),
        "alignmentErrorDeg": float(align_error),
        "minWallClearanceM": float(wall_min),
        "minHeatRegionClearanceM": float(heat_min),
        "heatExposure": float(heat_exposure),
    }


def _prefixed_terminal_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        out[f"{prefix}{key[0].upper()}{key[1:]}"] = float(value)
    return out


def _terminal_zone_objective(
    metrics: dict[str, float],
    cfg: PlannerConfig,
    context: SmoothingContext | None,
) -> float:
    score = 0.0
    score += 18.0 * float(metrics.get("directnessPenalty", 0.0))
    score += 5.0 * float(metrics.get("lateralDeviationMeanM", 0.0))
    score += 2.2 * float(metrics.get("maxCurvature", 0.0))
    score += 7.0 * float(metrics.get("projectionMonotonicViolations", 0.0))
    score += 7.0 * float(metrics.get("distanceMonotonicViolations", 0.0))
    score += 8.0 * float(metrics.get("overshootCount", 0.0))
    score += 20.0 * float(metrics.get("hookOrOvershootFlag", 0.0))
    score += 7.5 * max(0.0, float(cfg.min_terminal_progress_ratio) - float(metrics.get("progressRatio", 1.0)))
    score += 0.035 * float(metrics.get("alignmentErrorDeg", 0.0))

    min_heat = float(metrics.get("minHeatRegionClearanceM", -1.0))
    if min_heat >= 0.0:
        score += 0.85 / max(0.05, min_heat)
        score += 0.55 * float(metrics.get("heatExposure", 0.0))

    min_wall = float(metrics.get("minWallClearanceM", 0.0))
    if min_wall > 0.0:
        score += 0.35 / max(0.05, min_wall)
    if context is not None:
        required = max(0.0, float(context.required_clearance_m))
        score += 22.0 * max(0.0, required - min_wall)
    return score


def _blend_unit_dirs(base: np.ndarray, heading: np.ndarray | None, amount: float) -> np.ndarray:
    if heading is None:
        out = normalize(base)
        return out if np.linalg.norm(out) > 1e-12 else np.array([1.0, 0.0], dtype=float)
    w = max(0.0, min(1.0, float(amount)))
    out = normalize((1.0 - w) * base + w * heading)
    if np.linalg.norm(out) <= 1e-12:
        out = normalize(base)
    if np.linalg.norm(out) <= 1e-12:
        out = np.array([1.0, 0.0], dtype=float)
    return out


def _apply_terminal_line_blend(
    points: np.ndarray,
    s: np.ndarray,
    total_len: float,
    *,
    from_start: bool,
    zone_m: float,
    direction: np.ndarray,
    strength: float,
    blend_power: float,
) -> np.ndarray:
    if len(points) < 3 or zone_m <= 1e-9 or strength <= 1e-9:
        return points.copy()
    out = points.copy()
    d = normalize(direction)
    if np.linalg.norm(d) <= 1e-12:
        return out
    for i in range(1, len(out) - 1):
        if from_start:
            dist = float(s[i])
            if dist > zone_m:
                continue
            w = float(strength) * _endpoint_blend_weight(dist, zone_m, blend_power)
            if w <= 1e-9:
                continue
            desired = out[0] + d * dist
            out[i] = (1.0 - w) * out[i] + w * desired
        else:
            dist = float(total_len - s[i])
            if dist > zone_m:
                continue
            w = float(strength) * _endpoint_blend_weight(dist, zone_m, blend_power)
            if w <= 1e-9:
                continue
            desired = out[-1] - d * dist
            out[i] = (1.0 - w) * out[i] + w * desired
    return out


def _optimize_terminal_zone_points(
    points: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
    *,
    from_start: bool,
    attempt: int,
    requested_heading_deg: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 3:
        base = normalize(points[1] - points[0]) if len(points) >= 2 else np.array([1.0, 0.0], dtype=float)
        return points.copy(), base

    s = cumulative_arc_length(points)
    total_len = float(s[-1]) if len(s) else 0.0
    if total_len <= 1e-9:
        base = np.array([1.0, 0.0], dtype=float)
        return points.copy(), base
    zone_m = (
        _start_endpoint_zone_length(total_len, cfg, attempt)
        if from_start
        else _goal_endpoint_zone_length(total_len, cfg, attempt)
    )
    if zone_m <= 1e-9:
        base = normalize(points[1] - points[0]) if from_start else normalize(points[-1] - points[-2])
        if np.linalg.norm(base) <= 1e-12:
            base = np.array([1.0, 0.0], dtype=float)
        return points.copy(), base

    base_dir = normalize(points[1] - points[0]) if from_start else normalize(points[-1] - points[-2])
    if np.linalg.norm(base_dir) <= 1e-12:
        if from_start:
            base_dir = normalize(points[min(len(points) - 1, 2)] - points[0])
        else:
            base_dir = normalize(points[-1] - points[max(0, len(points) - 3)])
    if np.linalg.norm(base_dir) <= 1e-12:
        base_dir = np.array([1.0, 0.0], dtype=float)

    heading_dir = heading_to_unit(float(requested_heading_deg)) if requested_heading_deg is not None else None
    blend_values = [0.0]
    if heading_dir is not None:
        blend_values = [0.0, 0.15, 0.30, 0.45, 0.60]

    if cfg.fast_runtime:
        strengths = [0.0, 0.32, 0.55]
    else:
        strengths = [0.0, 0.18, 0.34, 0.50, 0.64]

    base_curv, base_curv_s = _curvature_samples_for_polyline(points)
    base_metrics = _terminal_zone_metrics(
        points=points,
        s=s,
        curvatures=base_curv,
        curvature_s=base_curv_s,
        cfg=cfg,
        context=context,
        from_start=from_start,
        heading_deg=requested_heading_deg,
    )
    best_points = points.copy()
    best_dir = base_dir.copy()
    best_score = _terminal_zone_objective(base_metrics, cfg, context)

    zone_idx = _terminal_zone_indices(s, total_len, zone_m, from_start=from_start)
    blend_power = max(1.0, float(cfg.endpoint_heading_blend_power))
    for blend in blend_values:
        candidate_dir = _blend_unit_dirs(base_dir, heading_dir, blend)
        for strength in strengths:
            cand_pts = _apply_terminal_line_blend(
                points=points,
                s=s,
                total_len=total_len,
                from_start=from_start,
                zone_m=zone_m,
                direction=candidate_dir,
                strength=strength,
                blend_power=blend_power,
            )
            cand_s = cumulative_arc_length(cand_pts)
            if len(cand_s) == 0 or float(cand_s[-1]) <= 1e-9:
                continue
            cand_curv, cand_curv_s = _curvature_samples_for_polyline(cand_pts)
            metrics = _terminal_zone_metrics(
                points=cand_pts,
                s=cand_s,
                curvatures=cand_curv,
                curvature_s=cand_curv_s,
                cfg=cfg,
                context=context,
                from_start=from_start,
                heading_deg=requested_heading_deg,
            )
            score = _terminal_zone_objective(metrics, cfg, context)
            if zone_idx.size > 0:
                local_disp = np.linalg.norm(cand_pts[zone_idx] - points[zone_idx], axis=1)
                score += 3.0 * float(np.mean(local_disp))
            score += 0.45 * float(blend)

            base_heat_min = float(base_metrics.get("minHeatRegionClearanceM", -1.0))
            cand_heat_min = float(metrics.get("minHeatRegionClearanceM", -1.0))
            if base_heat_min >= 0.0 and cand_heat_min >= 0.0 and cand_heat_min < base_heat_min - 0.01:
                score += 25.0 * (base_heat_min - cand_heat_min)
            base_heat_exp = float(base_metrics.get("heatExposure", 0.0))
            cand_heat_exp = float(metrics.get("heatExposure", 0.0))
            if base_heat_exp > 1e-9 and cand_heat_exp > base_heat_exp * 1.03:
                score += 8.0 * (cand_heat_exp / base_heat_exp - 1.0)

            base_wall_min = float(base_metrics.get("minWallClearanceM", 0.0))
            cand_wall_min = float(metrics.get("minWallClearanceM", 0.0))
            if base_wall_min > 0.0 and cand_wall_min < base_wall_min - 0.01:
                score += 22.0 * (base_wall_min - cand_wall_min)

            if score < best_score:
                best_score = score
                best_points = cand_pts
                best_dir = candidate_dir

    return best_points, best_dir


def _polyline_diagnostics(
    points: np.ndarray,
    endpoint_zone_m: float,
    cfg: PlannerConfig | None = None,
    context: SmoothingContext | None = None,
) -> dict[str, float]:
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.empty((0,))
    total_len = float(np.sum(seg_lengths)) if seg_lengths.size else 0.0
    zone_05 = min(0.5, total_len)
    tangent_jumps, tangent_jump_s = _tangent_jumps_for_polyline(points)
    curvatures, curvature_s = _curvature_samples_for_polyline(points)
    pct = _percentiles(curvatures)

    out: dict[str, float] = {
        "endpointZoneM": float(endpoint_zone_m),
        "maxTangentJumpDeg": float(np.max(tangent_jumps)) if tangent_jumps.size else 0.0,
        "maxTangentJumpNearStartDeg": _zone_max(
            tangent_jumps, tangent_jump_s, total_len, endpoint_zone_m, from_start=True
        ),
        "maxTangentJumpNearEndDeg": _zone_max(
            tangent_jumps, tangent_jump_s, total_len, endpoint_zone_m, from_start=False
        ),
        "maxCurvature": float(np.max(curvatures)) if curvatures.size else 0.0,
        "maxCurvatureNearStart": _zone_max(
            curvatures, curvature_s, total_len, endpoint_zone_m, from_start=True
        ),
        "maxCurvatureNearEnd": _zone_max(
            curvatures, curvature_s, total_len, endpoint_zone_m, from_start=False
        ),
        "maxCurvatureFirst0p5m": _zone_max(curvatures, curvature_s, total_len, zone_05, from_start=True),
        "maxCurvatureLast0p5m": _zone_max(curvatures, curvature_s, total_len, zone_05, from_start=False),
        "maxTangentJumpFirst0p5mDeg": _zone_max(
            tangent_jumps, tangent_jump_s, total_len, zone_05, from_start=True
        ),
        "maxTangentJumpLast0p5mDeg": _zone_max(
            tangent_jumps, tangent_jump_s, total_len, zone_05, from_start=False
        ),
        "curvatureP50": pct["p50"],
        "curvatureP90": pct["p90"],
        "curvatureP95": pct["p95"],
        "curvatureP99": pct["p99"],
        "segmentLengthMinM": float(np.min(seg_lengths)) if seg_lengths.size else 0.0,
        "segmentLengthMeanM": float(np.mean(seg_lengths)) if seg_lengths.size else 0.0,
        "segmentLengthMaxM": float(np.max(seg_lengths)) if seg_lengths.size else 0.0,
    }

    if cfg is not None:
        s = cumulative_arc_length(points)
        start_heading = _resolved_start_approach_heading_deg(cfg)
        goal_heading = _resolved_goal_approach_heading_deg(cfg)
        start_metrics = _terminal_zone_metrics(
            points=points,
            s=s,
            curvatures=curvatures,
            curvature_s=curvature_s,
            cfg=cfg,
            context=context,
            from_start=True,
            heading_deg=start_heading,
        )
        goal_metrics = _terminal_zone_metrics(
            points=points,
            s=s,
            curvatures=curvatures,
            curvature_s=curvature_s,
            cfg=cfg,
            context=context,
            from_start=False,
            heading_deg=goal_heading,
        )
        out.update(_prefixed_terminal_metrics("startTerminal", start_metrics))
        out.update(_prefixed_terminal_metrics("goalTerminal", goal_metrics))
        out["startEndpointAlignmentErrorDeg"] = float(start_metrics["alignmentErrorDeg"])
        out["endEndpointAlignmentErrorDeg"] = float(goal_metrics["alignmentErrorDeg"])
        out["terminalOvershootCount"] = float(goal_metrics["overshootCount"])
        out["terminalGoalProjectionMonotonicViolations"] = float(goal_metrics["projectionMonotonicViolations"])
        out["terminalGoalDistanceMonotonicViolations"] = float(goal_metrics["distanceMonotonicViolations"])
        out["terminalProgressRatio"] = float(goal_metrics["progressRatio"])

    return out


def _boundary_derivative_mag(points: np.ndarray, s: np.ndarray, start: bool) -> float:
    if len(points) < 2:
        return 1.0
    if start:
        ds = max(float(s[1] - s[0]), 1e-9)
        mag = float(np.linalg.norm(points[1] - points[0]) / ds)
    else:
        ds = max(float(s[-1] - s[-2]), 1e-9)
        mag = float(np.linalg.norm(points[-1] - points[-2]) / ds)
    return max(0.15, mag)


def _fit_centerline_with_constraints(
    resampled: np.ndarray,
    cfg: PlannerConfig,
    smoothing_scale: float,
    attempt: int,
    context: SmoothingContext | None = None,
) -> tuple[CubicSpline, CubicSpline, float]:
    s_raw = cumulative_arc_length(resampled)
    total = float(s_raw[-1]) if len(s_raw) else 0.0
    if total <= 1e-9:
        raise ValueError("Resampled path has near-zero length.")

    smooth_points = resampled.copy()
    if len(resampled) >= 4:
        k = min(3, len(resampled) - 1)
        u = s_raw / total
        s_param = (
            smoothing_scale
            * len(resampled)
            * max(cfg.sample_ds_m, 1e-3)
            * max(cfg.sample_ds_m, 1e-3)
        )
        try:
            tck, _ = splprep([resampled[:, 0], resampled[:, 1]], u=u, k=k, s=max(0.0, s_param))
            eval_count = max(
                len(resampled),
                int(math.ceil(total / max(cfg.sample_ds_m * 0.75, 1e-3))) + 1,
            )
            u_eval = np.linspace(0.0, 1.0, eval_count)
            x_eval, y_eval = splev(u_eval, tck)
            smooth_points = np.column_stack([x_eval, y_eval]).astype(float, copy=False)
        except Exception:
            smooth_points = resampled.copy()

    smooth_points[0] = resampled[0]
    smooth_points[-1] = resampled[-1]
    smooth_points = dedupe_consecutive(smooth_points, tol=1e-9)
    if len(smooth_points) < 3:
        mid = 0.5 * (resampled[0] + resampled[-1])
        smooth_points = np.vstack([resampled[0], mid, resampled[-1]])

    s = cumulative_arc_length(smooth_points)
    total = float(s[-1])
    if total <= 1e-9:
        raise ValueError("Spline source path collapsed to near-zero length.")

    # Keep terminal zones anchored to the raw path shape before local endpoint optimization.
    if len(smooth_points) > 2:
        raw_total = float(s_raw[-1]) if len(s_raw) else 0.0
        if raw_total > 1e-9:
            u_smooth = np.clip(s / total, 0.0, 1.0)
            u_raw = np.clip(s_raw / raw_total, 0.0, 1.0)
            ref_points = np.empty_like(smooth_points)
            ref_points[:, 0] = np.interp(u_smooth, u_raw, resampled[:, 0])
            ref_points[:, 1] = np.interp(u_smooth, u_raw, resampled[:, 1])
            preserve_start_zone = _start_endpoint_zone_length(total, cfg, attempt)
            preserve_goal_zone = _goal_endpoint_zone_length(total, cfg, attempt)
            for i in range(1, len(smooth_points) - 1):
                dist_start = float(s[i])
                dist_goal = float(total - s[i])
                w_start = _endpoint_blend_weight(dist_start, preserve_start_zone, 1.0)
                w_goal = _endpoint_blend_weight(dist_goal, preserve_goal_zone, 1.0)
                w_ref = 0.78 * max(w_start, w_goal)
                if w_ref > 1e-9:
                    smooth_points[i] = (1.0 - w_ref) * smooth_points[i] + w_ref * ref_points[i]
            s = cumulative_arc_length(smooth_points)
            total = float(s[-1])
            if total <= 1e-9:
                raise ValueError("Spline source path collapsed after terminal raw-shape anchoring.")

    start_heading = _resolved_start_approach_heading_deg(cfg)
    goal_heading = _resolved_goal_approach_heading_deg(cfg)
    smooth_points, start_dir_candidate = _optimize_terminal_zone_points(
        points=smooth_points,
        cfg=cfg,
        context=context,
        from_start=True,
        attempt=attempt,
        requested_heading_deg=start_heading,
    )
    smooth_points, end_dir_candidate = _optimize_terminal_zone_points(
        points=smooth_points,
        cfg=cfg,
        context=context,
        from_start=False,
        attempt=attempt,
        requested_heading_deg=goal_heading,
    )
    smooth_points[0] = resampled[0]
    smooth_points[-1] = resampled[-1]
    smooth_points = dedupe_consecutive(smooth_points, tol=1e-9)
    if len(smooth_points) < 3:
        mid = 0.5 * (resampled[0] + resampled[-1])
        smooth_points = np.vstack([resampled[0], mid, resampled[-1]])

    s = cumulative_arc_length(smooth_points)
    total = float(s[-1])
    if total <= 1e-9:
        raise ValueError("Spline source path collapsed after terminal optimization.")

    local_start_dir = normalize(smooth_points[1] - smooth_points[0])
    local_end_dir = normalize(smooth_points[-1] - smooth_points[-2])
    start_dir = _blend_unit_dirs(local_start_dir, start_dir_candidate, 0.25)
    end_dir = _blend_unit_dirs(local_end_dir, end_dir_candidate, 0.25)
    if np.linalg.norm(start_dir) <= 1e-12:
        start_dir = _resolved_start_dir(cfg, smooth_points[1] - smooth_points[0])
    if np.linalg.norm(end_dir) <= 1e-12:
        end_dir = _resolved_goal_dir(cfg, smooth_points[-1] - smooth_points[-2])

    d0 = start_dir * _boundary_derivative_mag(smooth_points, s, start=True)
    d1 = end_dir * _boundary_derivative_mag(smooth_points, s, start=False)

    spline_x = CubicSpline(
        s,
        smooth_points[:, 0],
        bc_type=((1, float(d0[0])), (1, float(d1[0]))),
    )
    spline_y = CubicSpline(
        s,
        smooth_points[:, 1],
        bc_type=((1, float(d0[1])), (1, float(d1[1]))),
    )
    return spline_x, spline_y, total


def _segment_turn_angle_deg(anchors: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(anchors) - 1:
        return 0.0
    return _angle_between_deg(anchors[idx] - anchors[idx - 1], anchors[idx + 1] - anchors[idx])


def _bezier_segment_count(total_len: float, cfg: PlannerConfig, attempt: int) -> int:
    target = cfg.bezier_target_segment_length_m * (cfg.refit_segment_length_growth**attempt)
    n_seg = int(math.ceil(total_len / max(target, 1e-3)))
    n_seg = max(cfg.min_bezier_segments, n_seg)
    n_seg = min(cfg.max_bezier_segments, n_seg)
    return max(1, n_seg)


def _anchor_sample_positions(
    total_len: float,
    n_seg: int,
    cfg: PlannerConfig,
    attempt: int,
) -> np.ndarray:
    if total_len <= 1e-9:
        return np.linspace(0.0, total_len, n_seg + 1)

    u = np.linspace(0.0, 1.0, n_seg + 1)
    exp_base = max(0.35, min(1.0, float(cfg.endpoint_spacing_exponent)))
    exp_attempt = exp_base + 0.03 * attempt
    exp_used = max(0.35, min(1.0, exp_attempt))
    warped = np.empty_like(u)
    left = u <= 0.5
    warped[left] = 0.5 * np.power(2.0 * u[left], exp_used)
    warped[~left] = 1.0 - 0.5 * np.power(2.0 * (1.0 - u[~left]), exp_used)
    warped[0] = 0.0
    warped[-1] = 1.0
    s = total_len * warped
    s = np.maximum.accumulate(s)
    return s


def _start_endpoint_zone_length(total_len: float, cfg: PlannerConfig, attempt: int) -> float:
    if total_len <= 1e-9:
        return 0.0
    base = max(0.0, float(cfg.endpoint_zone_m), float(cfg.start_approach_lock_distance_m))
    zone = base * (cfg.endpoint_zone_growth**attempt)
    return max(0.0, min(zone, 0.48 * total_len))


def _goal_endpoint_zone_length(total_len: float, cfg: PlannerConfig, attempt: int) -> float:
    if total_len <= 1e-9:
        return 0.0
    base = max(0.0, float(cfg.endpoint_zone_m), float(cfg.goal_approach_lock_distance_m))
    zone = base * (cfg.endpoint_zone_growth**attempt)
    return max(0.0, min(zone, 0.48 * total_len))


def _endpoint_zone_length(total_len: float, cfg: PlannerConfig, attempt: int) -> float:
    return max(
        _start_endpoint_zone_length(total_len, cfg, attempt),
        _goal_endpoint_zone_length(total_len, cfg, attempt),
    )


def _endpoint_blend_weight(distance: float, zone: float, power: float) -> float:
    if zone <= 1e-9:
        return 0.0
    t = max(0.0, min(1.0, 1.0 - distance / zone))
    return t**power


def _compute_anchor_dirs(anchors: np.ndarray, cfg: PlannerConfig, attempt: int) -> np.ndarray:
    n = len(anchors)
    dirs = np.zeros((n, 2), dtype=float)
    if n == 0:
        return dirs

    for i in range(n):
        if i == 0:
            dirs[i] = normalize(anchors[1] - anchors[0]) if n > 1 else np.array([1.0, 0.0], dtype=float)
        elif i == n - 1:
            dirs[i] = normalize(anchors[-1] - anchors[-2]) if n > 1 else np.array([1.0, 0.0], dtype=float)
        else:
            dirs[i] = normalize(anchors[i + 1] - anchors[i - 1])

    for i in range(1, n):
        if np.linalg.norm(dirs[i]) <= 1e-12:
            dirs[i] = dirs[i - 1]
    for i in range(n - 2, -1, -1):
        if np.linalg.norm(dirs[i]) <= 1e-12:
            dirs[i] = dirs[i + 1]
    for i in range(n):
        if np.linalg.norm(dirs[i]) <= 1e-12:
            dirs[i] = np.array([1.0, 0.0], dtype=float)
    start_heading = _resolved_start_approach_heading_deg(cfg)
    goal_heading = _resolved_goal_approach_heading_deg(cfg)
    if start_heading is not None:
        dirs[0] = _blend_unit_dirs(dirs[0], heading_to_unit(start_heading), 0.12)
    if goal_heading is not None:
        dirs[-1] = _blend_unit_dirs(dirs[-1], heading_to_unit(goal_heading), 0.12)

    for i in range(n):
        if np.linalg.norm(dirs[i]) <= 1e-12:
            dirs[i] = np.array([1.0, 0.0], dtype=float)
    return dirs


def _initial_handle_lengths(anchors: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    n = len(anchors)
    chord_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            local = float(chord_lengths[0]) if chord_lengths.size else 0.0
        elif i == n - 1:
            local = float(chord_lengths[-1]) if chord_lengths.size else 0.0
        else:
            local = float(min(chord_lengths[i - 1], chord_lengths[i]))
        out[i] = max(cfg.min_handle_length_m, cfg.handle_scale * local)
    return out


def _align_dirs_to_chords(dirs: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if len(anchors) < 2:
        return dirs
    out = dirs.copy()
    chords = np.diff(anchors, axis=0)
    chord_dirs = np.zeros_like(chords)
    for i in range(len(chords)):
        chord_dirs[i] = normalize(chords[i])

    for i in range(len(out)):
        if i == 0:
            ref = chord_dirs[0]
            if np.dot(out[i], ref) < 0.0:
                out[i] = ref
            continue
        if i == len(out) - 1:
            ref = chord_dirs[-1]
            if np.dot(out[i], ref) < 0.0:
                out[i] = ref
            continue

        prev_ref = chord_dirs[i - 1]
        next_ref = chord_dirs[i]
        if np.dot(out[i], prev_ref) < -0.2 and np.dot(out[i], next_ref) < -0.2:
            bisector = normalize(prev_ref + next_ref)
            if np.linalg.norm(bisector) <= 1e-12:
                bisector = next_ref
            out[i] = bisector
    return out


def _segments_from_controls(
    anchors: np.ndarray,
    dirs: np.ndarray,
    handle_lengths: np.ndarray,
) -> tuple[list[BezierSegment], np.ndarray]:
    tangent_vectors = dirs * (3.0 * handle_lengths)[:, None]
    segments: list[BezierSegment] = []
    for i in range(len(anchors) - 1):
        p0 = anchors[i]
        p3 = anchors[i + 1]
        p1 = p0 + tangent_vectors[i] / 3.0
        p2 = p3 - tangent_vectors[i + 1] / 3.0
        segments.append(
            BezierSegment(
                p0=p0.copy(),
                p1=p1.copy(),
                p2=p2.copy(),
                p3=p3.copy(),
            )
        )
    return segments, tangent_vectors


def _compute_handle_caps(
    anchors: np.ndarray,
    s_anchor: np.ndarray,
    cfg: PlannerConfig,
    attempt: int,
) -> np.ndarray:
    n = len(anchors)
    chord_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    caps = np.zeros(n, dtype=float)
    attempt_scale = cfg.refit_handle_decay**attempt
    total_len = float(s_anchor[-1]) if len(s_anchor) else 0.0
    endpoint_zone = _endpoint_zone_length(total_len, cfg, attempt)
    endpoint_scale = max(
        cfg.min_endpoint_handle_scale,
        cfg.endpoint_handle_scale * (cfg.endpoint_handle_decay**attempt),
    )

    for i in range(n):
        if i == 0:
            local = float(chord_lengths[0]) if chord_lengths.size else 0.0
        elif i == n - 1:
            local = float(chord_lengths[-1]) if chord_lengths.size else 0.0
        else:
            local = float(min(chord_lengths[i - 1], chord_lengths[i]))

        cap = cfg.handle_clamp_ratio * local
        turn_deg = _segment_turn_angle_deg(anchors, i)
        if turn_deg > cfg.sharp_turn_deg:
            frac = (turn_deg - cfg.sharp_turn_deg) / max(1e-6, 180.0 - cfg.sharp_turn_deg)
            frac = max(0.0, min(1.0, frac))
            cap *= 1.0 - (1.0 - cfg.sharp_turn_handle_scale) * frac
        cap *= attempt_scale
        if endpoint_zone > 1e-9:
            d_start = float(s_anchor[i])
            d_end = total_len - d_start
            w_start = _endpoint_blend_weight(d_start, endpoint_zone, 1.0)
            w_end = _endpoint_blend_weight(d_end, endpoint_zone, 1.0)
            w = max(w_start, w_end)
            cap *= 1.0 - (1.0 - endpoint_scale) * w
        caps[i] = max(cfg.min_handle_length_m, cap)
    return caps


def _compute_handle_floors(anchors: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    n = len(anchors)
    chord_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    floors = np.zeros(n, dtype=float)
    endpoint_floor_ratio = max(
        cfg.min_handle_ratio,
        min(cfg.handle_clamp_ratio * 0.75, cfg.min_handle_ratio + 0.12),
    )
    near_endpoint_floor_ratio = max(
        cfg.min_handle_ratio,
        min(cfg.handle_clamp_ratio * 0.65, cfg.min_handle_ratio + 0.06),
    )
    for i in range(n):
        if i == 0:
            local = float(chord_lengths[0]) if chord_lengths.size else 0.0
        elif i == n - 1:
            local = float(chord_lengths[-1]) if chord_lengths.size else 0.0
        else:
            local = float(min(chord_lengths[i - 1], chord_lengths[i]))
        ratio = cfg.min_handle_ratio
        if i in (0, n - 1):
            ratio = endpoint_floor_ratio
        elif i in (1, n - 2):
            ratio = near_endpoint_floor_ratio
        floors[i] = max(cfg.min_handle_length_m, ratio * local)
    return floors


def _regularize_handle_lengths(
    handle_lengths: np.ndarray,
    caps: np.ndarray,
    floors: np.ndarray,
    cfg: PlannerConfig,
) -> np.ndarray:
    if len(handle_lengths) <= 2 or cfg.c2_regularization_iters <= 0:
        return handle_lengths
    w = max(0.0, min(1.0, cfg.c2_regularization_weight))
    out = handle_lengths.copy()
    for _ in range(cfg.c2_regularization_iters):
        updated = out.copy()
        for i in range(1, len(out) - 1):
            updated[i] = (1.0 - w) * out[i] + w * 0.5 * (out[i - 1] + out[i + 1])
        out = np.minimum(updated, caps)
        out = np.maximum(out, floors)
    return out


def _limit_handles_by_curvature(
    anchors: np.ndarray,
    s_anchor: np.ndarray,
    dirs: np.ndarray,
    handle_lengths: np.ndarray,
    caps: np.ndarray,
    floors: np.ndarray,
    cfg: PlannerConfig,
    attempt: int,
) -> np.ndarray:
    if len(anchors) < 2:
        return handle_lengths

    out = handle_lengths.copy()
    total_len = float(s_anchor[-1]) if len(s_anchor) else 0.0
    endpoint_zone = _endpoint_zone_length(total_len, cfg, attempt)
    for _ in range(max(1, cfg.curvature_iters)):
        segments, _ = _segments_from_controls(anchors, dirs, out)
        changed = False
        for i, seg in enumerate(segments):
            k = max_curvature(seg, samples=50)
            seg_mid = 0.5 * (float(s_anchor[i]) + float(s_anchor[i + 1]))
            target_k = cfg.max_curvature
            if endpoint_zone > 1e-9 and (
                seg_mid <= endpoint_zone or (total_len - seg_mid) <= endpoint_zone
            ):
                target_k = min(target_k, cfg.max_endpoint_curvature)
            if k <= target_k:
                continue
            factor = max(0.15, min(0.95, math.sqrt(target_k / (k + 1e-12))))
            if i == 0:
                out[i + 1] = max(floors[i + 1], out[i + 1] * factor)
            elif i == len(segments) - 1:
                out[i] = max(floors[i], out[i] * factor)
            else:
                out[i] = max(floors[i], out[i] * factor)
                out[i + 1] = max(floors[i + 1], out[i + 1] * factor)
            changed = True
        out = np.minimum(out, caps)
        out = np.maximum(out, floors)
        if not changed:
            break
    return out


def _build_bezier_chain_from_centerline(
    spline_x: CubicSpline,
    spline_y: CubicSpline,
    total_len: float,
    cfg: PlannerConfig,
    attempt: int,
) -> tuple[list[BezierSegment], np.ndarray, np.ndarray]:
    n_seg = _bezier_segment_count(total_len, cfg, attempt)
    s_anchor = _anchor_sample_positions(total_len, n_seg, cfg, attempt)
    anchors = np.column_stack([spline_x(s_anchor), spline_y(s_anchor)]).astype(float, copy=False)
    if len(anchors) > 0:
        anchors[0] = np.array([float(spline_x(0.0)), float(spline_y(0.0))], dtype=float)
        anchors[-1] = np.array([float(spline_x(total_len)), float(spline_y(total_len))], dtype=float)
    s_anchor = cumulative_arc_length(anchors)

    dirs = _compute_anchor_dirs(anchors, cfg, attempt)
    dirs = _align_dirs_to_chords(dirs, anchors)

    handle_lengths = _initial_handle_lengths(anchors, cfg)
    caps = _compute_handle_caps(anchors, s_anchor, cfg, attempt)
    floors = _compute_handle_floors(anchors, cfg)
    floors = np.minimum(floors, caps)
    handle_lengths = np.minimum(handle_lengths, caps)
    handle_lengths = np.maximum(handle_lengths, floors)
    handle_lengths = _regularize_handle_lengths(handle_lengths, caps, floors, cfg)
    handle_lengths = _limit_handles_by_curvature(
        anchors=anchors,
        s_anchor=s_anchor,
        dirs=dirs,
        handle_lengths=handle_lengths,
        caps=caps,
        floors=floors,
        cfg=cfg,
        attempt=attempt,
    )
    segments, tangent_vectors = _segments_from_controls(anchors, dirs, handle_lengths)
    return segments, anchors, tangent_vectors


def _fallback_dense_sample_ds_m(cfg: PlannerConfig) -> float:
    # Keep fallback path dense enough for runtime and visuals (2-5 cm).
    return min(0.05, max(0.02, float(cfg.sample_ds_m)))


def _dedupe_points_with_u(
    points: np.ndarray,
    u: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    uu = np.asarray(u, dtype=float)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    keep: list[int] = [0]
    for i in range(1, len(pts)):
        j = keep[-1]
        if np.linalg.norm(pts[i] - pts[j]) <= tol and abs(float(uu[i] - uu[j])) <= 1e-9:
            continue
        keep.append(i)

    idx = np.asarray(keep, dtype=int)
    out_pts = pts[idx]
    out_u = uu[idx]
    if len(out_u) > 0:
        out_u = np.clip(out_u, 0.0, 1.0)
        out_u = np.maximum.accumulate(out_u)
        out_u[0] = 0.0
        out_u[-1] = 1.0
    return out_pts, out_u


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= 1e-9:
        return float(default)
    return float(num / den)


def _integrate_cost_along_polyline(
    points_world: np.ndarray,
    cost_field: np.ndarray | None,
    resolution_m: float,
) -> float:
    if cost_field is None or len(points_world) < 2:
        return 0.0
    points = np.asarray(points_world, dtype=float)
    p0 = points[:-1]
    p1 = points[1:]
    ds = np.linalg.norm(p1 - p0, axis=1)
    valid_seg = ds > 1e-12
    if not np.any(valid_seg):
        return 0.0
    x = points[:, 0] / float(resolution_m)
    y = points[:, 1] / float(resolution_m)
    w = bilinear_sample_grid_vectorized(cost_field, x, y)
    w0 = w[:-1]
    w1 = w[1:]
    valid = valid_seg & np.isfinite(w0) & np.isfinite(w1)
    if not np.any(valid):
        return 0.0
    return float(np.sum(0.5 * (w0[valid] + w1[valid]) * ds[valid]))


def _max_cost_along_polyline(
    points_world: np.ndarray,
    cost_field: np.ndarray | None,
    resolution_m: float,
    sample_ds_m: float,
) -> float:
    if cost_field is None or len(points_world) == 0:
        return 0.0
    points = dedupe_consecutive(np.asarray(points_world, dtype=float), tol=1e-9)
    if len(points) == 0:
        return 0.0
    if len(points) >= 2:
        dense = dedupe_consecutive(
            resample_polyline(points, max(0.01, float(sample_ds_m))),
            tol=1e-9,
        )
        if len(dense) >= 2:
            points = dense
    vals = sample_field_along_path(points, cost_field, resolution_m)
    finite = vals[np.isfinite(vals)]
    return float(np.max(finite)) if finite.size else 0.0


def _interpolate_point_by_u(
    points: np.ndarray,
    u: np.ndarray,
    target_u: float,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((2,), dtype=float)
    if len(points) == 1:
        return points[0].copy()

    t = float(np.clip(target_u, 0.0, 1.0))
    uu = np.asarray(u, dtype=float)
    idx = int(np.searchsorted(uu, t, side="right"))
    if idx <= 0:
        return points[0].copy()
    if idx >= len(points):
        return points[-1].copy()
    lo = idx - 1
    hi = idx
    u0 = float(uu[lo])
    u1 = float(uu[hi])
    if u1 - u0 <= 1e-9:
        return points[hi].copy()
    a = (t - u0) / (u1 - u0)
    return ((1.0 - a) * points[lo] + a * points[hi]).astype(float, copy=False)


def _polyline_subpath_by_u(
    points: np.ndarray,
    u: np.ndarray,
    u0: float,
    u1: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)

    reverse = float(u1) < float(u0)
    lo = float(np.clip(min(u0, u1), 0.0, 1.0))
    hi = float(np.clip(max(u0, u1), 0.0, 1.0))
    if hi - lo <= 1e-9:
        p = _interpolate_point_by_u(points, u, lo)
        out_pts = np.vstack([p, p])
        out_u = np.array([lo, hi], dtype=float)
        if reverse:
            return out_pts[::-1].copy(), out_u[::-1].copy()
        return out_pts, out_u

    start = _interpolate_point_by_u(points, u, lo)
    end = _interpolate_point_by_u(points, u, hi)
    mask = (u > lo + 1e-9) & (u < hi - 1e-9)
    mids = points[mask]
    mids_u = u[mask]
    if len(mids) > 0:
        out_pts = np.vstack([start, mids, end])
        out_u = np.concatenate([np.array([lo], dtype=float), mids_u.astype(float), np.array([hi], dtype=float)])
    else:
        out_pts = np.vstack([start, end])
        out_u = np.array([lo, hi], dtype=float)

    out_pts, out_u = _dedupe_points_with_u(out_pts, out_u, tol=1e-9)
    if len(out_pts) < 2:
        out_pts = np.vstack([start, end])
        out_u = np.array([lo, hi], dtype=float)
    if reverse:
        return out_pts[::-1].copy(), out_u[::-1].copy()
    return out_pts, out_u


def _chaikin_refine_with_u(
    points: np.ndarray,
    u: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    pts, uu = _dedupe_points_with_u(np.asarray(points, dtype=float), np.asarray(u, dtype=float), tol=1e-9)
    iters = max(0, int(iterations))
    for _ in range(iters):
        if len(pts) < 3:
            break
        new_pts: list[np.ndarray] = [pts[0].copy()]
        new_u: list[float] = [float(uu[0])]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            u0 = float(uu[i])
            u1 = float(uu[i + 1])
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            uq = 0.75 * u0 + 0.25 * u1
            ur = 0.25 * u0 + 0.75 * u1
            new_pts.append(np.asarray(q, dtype=float))
            new_pts.append(np.asarray(r, dtype=float))
            new_u.append(float(uq))
            new_u.append(float(ur))
        new_pts.append(pts[-1].copy())
        new_u.append(float(uu[-1]))
        pts, uu = _dedupe_points_with_u(np.asarray(new_pts, dtype=float), np.asarray(new_u, dtype=float), tol=1e-9)
    return pts, uu


def _signed_turn_angles_deg(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.empty((0,), dtype=float)

    segs = np.diff(pts, axis=0)
    out = np.zeros((max(0, len(segs) - 1),), dtype=float)
    for i in range(1, len(segs)):
        a = segs[i - 1]
        b = segs[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            continue
        dot = float(np.dot(a, b) / (na * nb))
        dot = max(-1.0, min(1.0, dot))
        ang = float(math.degrees(math.acos(dot)))
        cross = float(a[0] * b[1] - a[1] * b[0])
        if cross > 1e-12:
            out[i - 1] = ang
        elif cross < -1e-12:
            out[i - 1] = -ang
        else:
            out[i - 1] = 0.0
    return out


def _alternating_sharp_turn_vertex_mask(points: np.ndarray, sharp_turn_deg: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    mask = np.zeros((len(pts),), dtype=bool)
    turns = _signed_turn_angles_deg(pts)
    if len(turns) < 2:
        return mask

    threshold = max(1e-3, float(sharp_turn_deg))
    for i in range(len(turns)):
        ti = float(turns[i])
        if abs(ti) < threshold:
            continue
        alt_prev = (
            i > 0
            and abs(float(turns[i - 1])) >= threshold
            and float(turns[i - 1]) * ti < 0.0
        )
        alt_next = (
            i + 1 < len(turns)
            and abs(float(turns[i + 1])) >= threshold
            and float(turns[i + 1]) * ti < 0.0
        )
        if alt_prev or alt_next:
            v_idx = i + 1
            if 0 < v_idx < len(pts) - 1:
                mask[v_idx] = True
    return mask


def _count_alternating_sharp_turn_pairs(points: np.ndarray, sharp_turn_deg: float) -> int:
    turns = _signed_turn_angles_deg(np.asarray(points, dtype=float))
    if len(turns) < 2:
        return 0
    threshold = max(1e-3, float(sharp_turn_deg))
    count = 0
    for i in range(1, len(turns)):
        t0 = float(turns[i - 1])
        t1 = float(turns[i])
        if abs(t0) < threshold or abs(t1) < threshold:
            continue
        if t0 * t1 < 0.0:
            count += 1
    return int(count)


def _fallback_zigzag_turn_threshold_deg(cfg: PlannerConfig) -> float:
    # Use a moderate threshold so corridor "wiggles" are detected before they become visible artifacts.
    return max(4.0, min(16.0, 0.22 * float(cfg.sharp_turn_deg)))


def _blend_alternating_sharp_turn_vertices_once(
    points: np.ndarray,
    u: np.ndarray,
    sharp_turn_deg: float,
    blend_strength: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    pts, uu = _dedupe_points_with_u(np.asarray(points, dtype=float), np.asarray(u, dtype=float), tol=1e-9)
    if len(pts) < 4:
        return pts, uu, False

    mask = _alternating_sharp_turn_vertex_mask(pts, sharp_turn_deg)
    if not np.any(mask):
        return pts, uu, False

    w = max(0.10, min(0.70, float(blend_strength)))
    out = pts.copy()
    for idx in np.where(mask)[0]:
        mid = 0.5 * (pts[idx - 1] + pts[idx + 1])
        out[idx] = (1.0 - w) * pts[idx] + w * mid
    out[0] = pts[0]
    out[-1] = pts[-1]
    out_pts, out_u = _dedupe_points_with_u(out, uu, tol=1e-9)
    return out_pts, out_u, True


def _reduce_alternating_sharp_turn_artifacts(
    points: np.ndarray,
    u: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pts, uu = _dedupe_points_with_u(np.asarray(points, dtype=float), np.asarray(u, dtype=float), tol=1e-9)
    sharp_turn_deg = _fallback_zigzag_turn_threshold_deg(cfg)
    before = _count_alternating_sharp_turn_pairs(pts, sharp_turn_deg)
    if len(pts) < 4 or before <= 1:
        return pts, uu, {
            "applied": False,
            "passes": 0,
            "sharpTurnDeg": float(sharp_turn_deg),
            "beforePairs": int(before),
            "afterPairs": int(before),
        }

    max_passes = 2 if cfg.fast_runtime else 4
    base_strength = 0.36 if cfg.fast_runtime else 0.42
    best_pts = pts.copy()
    best_u = uu.copy()
    best_pairs = int(before)
    work_pts = pts.copy()
    work_u = uu.copy()
    passes_used = 0
    applied = False
    for it in range(max_passes):
        strength = min(0.68, base_strength + 0.08 * float(it))
        blended_pts, blended_u, touched = _blend_alternating_sharp_turn_vertices_once(
            work_pts,
            work_u,
            sharp_turn_deg=sharp_turn_deg,
            blend_strength=strength,
        )
        if not touched:
            break
        applied = True
        passes_used = it + 1
        pairs = _count_alternating_sharp_turn_pairs(blended_pts, sharp_turn_deg)
        if pairs <= best_pairs:
            best_pts = blended_pts.copy()
            best_u = blended_u.copy()
            best_pairs = int(pairs)
        work_pts = blended_pts
        work_u = blended_u
        if best_pairs <= 1:
            break

    return best_pts, best_u, {
        "applied": bool(applied),
        "passes": int(passes_used),
        "sharpTurnDeg": float(sharp_turn_deg),
        "beforePairs": int(before),
        "afterPairs": int(best_pairs),
    }


def _apply_heat_chord_guard(
    candidate_points: np.ndarray,
    candidate_u: np.ndarray,
    raw_points: np.ndarray,
    raw_u: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if (
        context is None
        or context.cost_density_field is None
        or len(candidate_points) < 2
        or len(raw_points) < 2
    ):
        return candidate_points.copy(), candidate_u.copy(), 0

    heat_field = context.cost_density_field
    sample_ds = max(0.01, min(0.04, 0.5 * _fallback_dense_sample_ds_m(cfg)))
    integral_tol_ratio = 1.001
    integral_tol_abs = 1e-5
    peak_tol_abs = 1e-5

    out_pts: list[np.ndarray] = [candidate_points[0].copy()]
    out_u: list[float] = [float(candidate_u[0])]
    rejected = 0
    for i in range(len(candidate_points) - 1):
        u0 = float(candidate_u[i])
        u1 = float(candidate_u[i + 1])
        chord = np.vstack([candidate_points[i], candidate_points[i + 1]])
        raw_section, raw_section_u = _polyline_subpath_by_u(raw_points, raw_u, u0, u1)
        if len(raw_section) < 2:
            raw_section = chord.copy()
            raw_section_u = np.array([u0, u1], dtype=float)

        chord_integral = _integrate_cost_along_polyline(chord, heat_field, cfg.resolution_m_per_cell)
        raw_integral = _integrate_cost_along_polyline(raw_section, heat_field, cfg.resolution_m_per_cell)
        chord_peak = _max_cost_along_polyline(
            chord,
            heat_field,
            cfg.resolution_m_per_cell,
            sample_ds_m=sample_ds,
        )
        raw_peak = _max_cost_along_polyline(
            raw_section,
            heat_field,
            cfg.resolution_m_per_cell,
            sample_ds_m=sample_ds,
        )
        worse_integral = chord_integral > raw_integral * integral_tol_ratio + integral_tol_abs
        worse_peak = chord_peak > raw_peak + peak_tol_abs
        if worse_integral or worse_peak:
            rejected += 1
            for j in range(1, len(raw_section)):
                out_pts.append(raw_section[j].copy())
                out_u.append(float(raw_section_u[j]))
        else:
            out_pts.append(candidate_points[i + 1].copy())
            out_u.append(float(candidate_u[i + 1]))

    guarded_pts, guarded_u = _dedupe_points_with_u(
        np.asarray(out_pts, dtype=float),
        np.asarray(out_u, dtype=float),
        tol=1e-9,
    )
    if len(guarded_pts) < 2:
        return raw_points.copy(), raw_u.copy(), max(1, rejected)
    return guarded_pts, guarded_u, rejected


def _polyline_to_linear_bezier_chain(
    points: np.ndarray,
) -> tuple[list[BezierSegment], np.ndarray, np.ndarray]:
    anchors = dedupe_consecutive(np.asarray(points, dtype=float), tol=1e-9)
    if len(anchors) < 2:
        return [], anchors.copy(), np.empty((0, 2), dtype=float)
    if len(anchors) == 2:
        mid = 0.5 * (anchors[0] + anchors[1])
        anchors = np.vstack([anchors[0], mid, anchors[1]])

    segments: list[BezierSegment] = []
    for i in range(len(anchors) - 1):
        p0 = anchors[i]
        p3 = anchors[i + 1]
        d = p3 - p0
        p1 = p0 + d / 3.0
        p2 = p0 + 2.0 * d / 3.0
        segments.append(BezierSegment(p0=p0.copy(), p1=p1.copy(), p2=p2.copy(), p3=p3.copy()))

    tangent_vectors = np.zeros((len(anchors), 2), dtype=float)
    tangent_vectors[0] = anchors[1] - anchors[0]
    tangent_vectors[-1] = anchors[-1] - anchors[-2]
    for i in range(1, len(anchors) - 1):
        tangent_vectors[i] = 0.5 * (anchors[i + 1] - anchors[i - 1])
    return segments, anchors, tangent_vectors


def _fallback_polyline_metrics(
    points: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
) -> dict[str, Any]:
    endpoint_zone = max(
        float(cfg.endpoint_zone_m),
        float(cfg.start_approach_lock_distance_m),
        float(cfg.goal_approach_lock_distance_m),
    )
    diag = _polyline_diagnostics(points, endpoint_zone_m=endpoint_zone, cfg=cfg, context=context)
    integrated_heat_cost = 0.0
    max_heat_cost = 0.0
    if context is not None and context.cost_density_field is not None:
        integrated_heat_cost = _integrate_cost_along_polyline(
            points,
            context.cost_density_field,
            cfg.resolution_m_per_cell,
        )
        max_heat_cost = _max_cost_along_polyline(
            points,
            context.cost_density_field,
            cfg.resolution_m_per_cell,
            sample_ds_m=max(0.01, 0.5 * _fallback_dense_sample_ds_m(cfg)),
        )
    return {
        "diagnostics": diag,
        "integratedHeatCost": float(integrated_heat_cost),
        "maxHeatCost": float(max_heat_cost),
        "minWallClearanceM": float(diag.get("minWallClearanceM", 0.0)),
        "minHeatRegionClearanceM": float(diag.get("minHeatRegionClearanceM", -1.0)),
        "terminalOvershootCount": float(diag.get("terminalOvershootCount", 0.0)),
        "startTerminalHookOrOvershootFlag": float(diag.get("startTerminalHookOrOvershootFlag", 0.0)),
        "goalTerminalHookOrOvershootFlag": float(diag.get("goalTerminalHookOrOvershootFlag", 0.0)),
    }


def _terminal_safe_dense_fallback_bezier_chain(
    raw_points: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
    trigger_reasons: list[str],
) -> tuple[list[BezierSegment], np.ndarray, np.ndarray, dict[str, Any]]:
    base_points = dedupe_consecutive(np.asarray(raw_points, dtype=float), tol=1e-9)
    if len(base_points) < 2:
        return [], base_points.copy(), np.empty((0, 2), dtype=float), {
            "fallbackReason": "terminal_degradation_guard_triggered_no_accepted_candidate",
            "triggerReasons": list(trigger_reasons),
            "denseSampleDsM": float(_fallback_dense_sample_ds_m(cfg)),
            "denseRawPointCount": int(len(base_points)),
            "finalPointCount": int(len(base_points)),
            "smoothingAttempted": False,
            "smoothingAccepted": False,
            "smoothingRejected": False,
            "smoothingRejectedReasons": [],
            "heatChordGuardEnabled": False,
            "heatChordRejectCount": 0,
            "rawAlternatingSharpTurnPairs": 0,
            "finalAlternatingSharpTurnPairs": 0,
            "zigzagAveragingApplied": False,
            "zigzagAveragingPasses": 0,
            "zigzagAveragingBeforePairs": 0,
            "zigzagAveragingAfterPairs": 0,
            "rawIntegratedHeatCost": 0.0,
            "finalIntegratedHeatCost": 0.0,
            "rawMaxHeatCost": 0.0,
            "finalMaxHeatCost": 0.0,
            "rawMinWallClearanceM": 0.0,
            "finalMinWallClearanceM": 0.0,
            "rawMinHeatRegionClearanceM": -1.0,
            "finalMinHeatRegionClearanceM": -1.0,
        }

    dense_ds = _fallback_dense_sample_ds_m(cfg)
    dense_raw = dedupe_consecutive(resample_polyline(base_points, dense_ds), tol=1e-9)
    if len(dense_raw) < 2:
        dense_raw = base_points.copy()
    if len(dense_raw) == 2:
        mid = 0.5 * (dense_raw[0] + dense_raw[1])
        dense_raw = np.vstack([dense_raw[0], mid, dense_raw[1]])

    s_raw = cumulative_arc_length(dense_raw)
    total_raw = float(s_raw[-1]) if len(s_raw) else 0.0
    raw_u = np.zeros((len(dense_raw),), dtype=float)
    if total_raw > 1e-9:
        raw_u = np.clip(s_raw / total_raw, 0.0, 1.0)
    if len(raw_u):
        raw_u[0] = 0.0
        raw_u[-1] = 1.0

    sharp_turn_deg = _fallback_zigzag_turn_threshold_deg(cfg)
    raw_alt_turn_pairs = _count_alternating_sharp_turn_pairs(dense_raw, sharp_turn_deg)
    final_alt_turn_pairs = int(raw_alt_turn_pairs)

    raw_metrics = _fallback_polyline_metrics(dense_raw, cfg, context)
    final_points = dense_raw.copy()
    final_metrics = raw_metrics
    smoothing_attempted = False
    smoothing_accepted = False
    smoothing_reject_reasons: list[str] = []
    heat_chord_reject_count = 0
    heat_guard_enabled = bool(context is not None and context.cost_density_field is not None)
    zigzag_averaging_applied = False
    zigzag_averaging_passes = 0
    zigzag_averaging_before_pairs = 0
    zigzag_averaging_after_pairs = 0

    if len(dense_raw) >= 3:
        smoothing_attempted = True
        chaikin_iters = 1 if cfg.fast_runtime else 2
        cand_points, cand_u = _chaikin_refine_with_u(dense_raw, raw_u, chaikin_iters)
        cand_points, cand_u, heat_chord_reject_count = _apply_heat_chord_guard(
            candidate_points=cand_points,
            candidate_u=cand_u,
            raw_points=dense_raw,
            raw_u=raw_u,
            cfg=cfg,
            context=context,
        )
        cand_points, cand_u, zigzag_diag = _reduce_alternating_sharp_turn_artifacts(
            points=cand_points,
            u=cand_u,
            cfg=cfg,
        )
        zigzag_averaging_applied = bool(zigzag_diag.get("applied", False))
        zigzag_averaging_passes = int(zigzag_diag.get("passes", 0))
        zigzag_averaging_before_pairs = int(zigzag_diag.get("beforePairs", 0))
        zigzag_averaging_after_pairs = int(zigzag_diag.get("afterPairs", 0))
        cand_metrics = _fallback_polyline_metrics(cand_points, cfg, context)
        cand_alt_turn_pairs = _count_alternating_sharp_turn_pairs(cand_points, sharp_turn_deg)

        clearance_tol_m = 1e-3
        heat_tol_ratio = 1.001
        heat_tol_abs = 1e-4

        raw_heat = float(raw_metrics["integratedHeatCost"])
        cand_heat = float(cand_metrics["integratedHeatCost"])
        if heat_guard_enabled and cand_heat > raw_heat * heat_tol_ratio + heat_tol_abs:
            smoothing_reject_reasons.append("integrated_heat_cost_worse_than_dense_raw")

        raw_wall = float(raw_metrics["minWallClearanceM"])
        cand_wall = float(cand_metrics["minWallClearanceM"])
        if cand_wall < raw_wall - clearance_tol_m:
            smoothing_reject_reasons.append("wall_clearance_worse_than_dense_raw")

        raw_heat_clear = float(raw_metrics["minHeatRegionClearanceM"])
        cand_heat_clear = float(cand_metrics["minHeatRegionClearanceM"])
        if (
            raw_heat_clear >= 0.0
            and cand_heat_clear >= 0.0
            and cand_heat_clear < raw_heat_clear - clearance_tol_m
        ):
            smoothing_reject_reasons.append("heat_region_clearance_worse_than_dense_raw")

        raw_overshoot = float(raw_metrics["terminalOvershootCount"])
        cand_overshoot = float(cand_metrics["terminalOvershootCount"])
        if cand_overshoot > raw_overshoot + 1e-6:
            smoothing_reject_reasons.append("terminal_overshoot_introduced")

        raw_start_hook = float(raw_metrics["startTerminalHookOrOvershootFlag"])
        cand_start_hook = float(cand_metrics["startTerminalHookOrOvershootFlag"])
        if cand_start_hook > raw_start_hook + 1e-6:
            smoothing_reject_reasons.append("start_terminal_hook_introduced")

        raw_goal_hook = float(raw_metrics["goalTerminalHookOrOvershootFlag"])
        cand_goal_hook = float(cand_metrics["goalTerminalHookOrOvershootFlag"])
        if cand_goal_hook > raw_goal_hook + 1e-6:
            smoothing_reject_reasons.append("goal_terminal_hook_introduced")

        if cand_alt_turn_pairs > raw_alt_turn_pairs:
            smoothing_reject_reasons.append("alternating_sharp_turns_worse_than_dense_raw")

        if not smoothing_reject_reasons:
            smoothing_accepted = True
            final_points = cand_points
            final_metrics = cand_metrics
            final_alt_turn_pairs = int(cand_alt_turn_pairs)

    segments, anchors, tangent_vectors = _polyline_to_linear_bezier_chain(final_points)
    fallback_diag: dict[str, Any] = {
        "fallbackReason": "terminal_degradation_guard_triggered_no_accepted_candidate",
        "triggerReasons": list(trigger_reasons),
        "denseSampleDsM": float(dense_ds),
        "denseRawPointCount": int(len(dense_raw)),
        "finalPointCount": int(len(final_points)),
        "smoothingAttempted": bool(smoothing_attempted),
        "smoothingAccepted": bool(smoothing_accepted),
        "smoothingRejected": bool(smoothing_attempted and not smoothing_accepted),
        "smoothingRejectedReasons": list(smoothing_reject_reasons),
        "heatChordGuardEnabled": bool(heat_guard_enabled),
        "heatChordRejectCount": int(heat_chord_reject_count),
        "rawAlternatingSharpTurnPairs": int(raw_alt_turn_pairs),
        "finalAlternatingSharpTurnPairs": int(final_alt_turn_pairs),
        "zigzagAveragingApplied": bool(zigzag_averaging_applied),
        "zigzagAveragingPasses": int(zigzag_averaging_passes),
        "zigzagAveragingBeforePairs": int(zigzag_averaging_before_pairs),
        "zigzagAveragingAfterPairs": int(zigzag_averaging_after_pairs),
        "rawIntegratedHeatCost": float(raw_metrics["integratedHeatCost"]),
        "finalIntegratedHeatCost": float(final_metrics["integratedHeatCost"]),
        "rawHeatCostRatio": float(
            _safe_ratio(
                float(final_metrics["integratedHeatCost"]),
                float(raw_metrics["integratedHeatCost"]),
                default=1.0,
            )
        ),
        "rawMaxHeatCost": float(raw_metrics["maxHeatCost"]),
        "finalMaxHeatCost": float(final_metrics["maxHeatCost"]),
        "rawMinWallClearanceM": float(raw_metrics["minWallClearanceM"]),
        "finalMinWallClearanceM": float(final_metrics["minWallClearanceM"]),
        "rawMinHeatRegionClearanceM": float(raw_metrics["minHeatRegionClearanceM"]),
        "finalMinHeatRegionClearanceM": float(final_metrics["minHeatRegionClearanceM"]),
    }
    return segments, anchors, tangent_vectors, fallback_diag


def _bezier_segment_length(seg: BezierSegment, samples: int = 30) -> float:
    ts = np.linspace(0.0, 1.0, max(2, samples))
    pts = np.asarray([bezier_point(seg, float(t)) for t in ts], dtype=float)
    return polyline_length(pts)


def _join_diagnostics(
    segments: list[BezierSegment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(segments) <= 1:
        return (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            0.0,
        )

    angle_jumps: list[float] = []
    mag_jumps: list[float] = []
    seg_lengths = np.asarray([_bezier_segment_length(seg) for seg in segments], dtype=float)
    join_positions = np.cumsum(seg_lengths)[:-1] if len(seg_lengths) > 1 else np.empty((0,), dtype=float)
    total_len = float(np.sum(seg_lengths))
    for i in range(1, len(segments)):
        d_prev = 3.0 * (segments[i - 1].p3 - segments[i - 1].p2)
        d_next = 3.0 * (segments[i].p1 - segments[i].p0)
        angle_jumps.append(_angle_between_deg(d_prev, d_next))
        n_prev = float(np.linalg.norm(d_prev))
        n_next = float(np.linalg.norm(d_next))
        if max(n_prev, n_next) <= 1e-12:
            mag_jumps.append(0.0)
        else:
            mag_jumps.append(abs(n_next - n_prev) / max(n_prev, n_next))
    return (
        np.asarray(angle_jumps, dtype=float),
        np.asarray(mag_jumps, dtype=float),
        join_positions,
        total_len,
    )


def _safe_stat_min(vals: np.ndarray, default: float = 0.0) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(default)
    return float(np.min(finite))


def _safe_stat_mean(vals: np.ndarray, default: float = 0.0) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _safe_stat_pctl(vals: np.ndarray, q: float, default: float = 0.0) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(finite, q))


def _endpoint_and_terminal_diagnostics(
    points: np.ndarray,
    tangent_headings: np.ndarray,
    s: np.ndarray,
    cfg: PlannerConfig,
    context: SmoothingContext | None,
) -> dict[str, float]:
    total_len = float(s[-1]) if len(s) else 0.0
    start_heading = _resolved_start_approach_heading_deg(cfg)
    goal_heading = _resolved_goal_approach_heading_deg(cfg)
    start_err = 0.0
    end_err = 0.0
    if len(points) >= 2:
        start_err = _angle_error_to_heading_deg(points[1] - points[0], start_heading)
        end_err = _angle_error_to_heading_deg(points[-1] - points[-2], goal_heading)

    start_zone = min(total_len, max(cfg.start_approach_lock_distance_m, cfg.endpoint_zone_m))
    goal_zone = min(
        total_len,
        max(cfg.goal_approach_lock_distance_m, cfg.endpoint_zone_m, cfg.terminal_progress_window_m),
    )
    start_zone_heading_max = 0.0
    goal_zone_heading_max = 0.0
    start_zone_heading_raw_max = 0.0
    goal_zone_heading_raw_max = 0.0
    if len(tangent_headings) == len(points):
        if start_heading is not None and start_zone > 1e-9:
            mask = s <= start_zone
            if np.any(mask):
                errs = np.abs(
                    np.degrees(
                        np.array(
                            [_wrap_to_pi(float(t) - math.radians(float(start_heading))) for t in tangent_headings[mask]]
                        )
                    )
                )
                dvals = s[mask]
                w = np.clip(1.0 - (dvals / max(start_zone, 1e-9)), 0.0, 1.0)
                start_zone_heading_raw_max = float(np.max(errs)) if errs.size else 0.0
                start_zone_heading_max = float(np.max(errs * w)) if errs.size else 0.0
        if goal_heading is not None and goal_zone > 1e-9:
            mask = s >= max(0.0, total_len - goal_zone)
            if np.any(mask):
                errs = np.abs(
                    np.degrees(
                        np.array(
                            [_wrap_to_pi(float(t) - math.radians(float(goal_heading))) for t in tangent_headings[mask]]
                        )
                    )
                )
                dvals = total_len - s[mask]
                w = np.clip(1.0 - (dvals / max(goal_zone, 1e-9)), 0.0, 1.0)
                goal_zone_heading_raw_max = float(np.max(errs)) if errs.size else 0.0
                goal_zone_heading_max = float(np.max(errs * w)) if errs.size else 0.0

    curvatures, curvature_s = _curvature_samples_for_polyline(points)
    start_terminal = _terminal_zone_metrics(
        points=points,
        s=s,
        curvatures=curvatures,
        curvature_s=curvature_s,
        cfg=cfg,
        context=context,
        from_start=True,
        heading_deg=start_heading,
    )
    goal_terminal = _terminal_zone_metrics(
        points=points,
        s=s,
        curvatures=curvatures,
        curvature_s=curvature_s,
        cfg=cfg,
        context=context,
        from_start=False,
        heading_deg=goal_heading,
    )
    return {
        "startEndpointAlignmentErrorDeg": float(start_err),
        "endEndpointAlignmentErrorDeg": float(end_err),
        "maxStartLockHeadingErrorDeg": float(start_zone_heading_max),
        "maxGoalLockHeadingErrorDeg": float(goal_zone_heading_max),
        "maxStartLockHeadingErrorRawDeg": float(start_zone_heading_raw_max),
        "maxGoalLockHeadingErrorRawDeg": float(goal_zone_heading_raw_max),
        "terminalOvershootCount": float(goal_terminal["overshootCount"]),
        "terminalGoalProjectionMonotonicViolations": float(goal_terminal["projectionMonotonicViolations"]),
        "terminalGoalDistanceMonotonicViolations": float(goal_terminal["distanceMonotonicViolations"]),
        "terminalProgressRatio": float(goal_terminal["progressRatio"]),
        "terminalStartProjectionMonotonicViolations": float(start_terminal["projectionMonotonicViolations"]),
        "terminalStartDistanceMonotonicViolations": float(start_terminal["distanceMonotonicViolations"]),
        "terminalStartOvershootCount": float(start_terminal["overshootCount"]),
        "terminalStartProgressRatio": float(start_terminal["progressRatio"]),
        "terminalHookOrOvershootDetected": float(
            max(start_terminal["hookOrOvershootFlag"], goal_terminal["hookOrOvershootFlag"])
        ),
        **_prefixed_terminal_metrics("startTerminal", start_terminal),
        **_prefixed_terminal_metrics("goalTerminal", goal_terminal),
    }


def _bezier_chain_diagnostics(
    segments: list[BezierSegment],
    sample_ds_m: float,
    cfg: PlannerConfig,
    context: SmoothingContext | None = None,
    lightweight: bool = False,
) -> dict[str, float]:
    sampled_points, tangent_headings, sampled_curvatures = sample_bezier_chain(
        segments,
        sample_per_segment=max(
            16 if lightweight else 35,
            int((0.65 if lightweight else 1.0) / max(sample_ds_m, 1e-3)),
        ),
    )
    finite_curv = sampled_curvatures[np.isfinite(sampled_curvatures)]
    curv_pct = _percentiles(finite_curv)

    angle_jumps, mag_jumps, join_positions, join_total = _join_diagnostics(segments)
    seg_lengths = np.asarray([_bezier_segment_length(seg) for seg in segments], dtype=float)
    s = cumulative_arc_length(sampled_points)
    total_len = float(s[-1]) if len(s) else 0.0
    endpoint_zone = max(
        0.0,
        min(
            total_len,
            max(
                cfg.endpoint_zone_m,
                cfg.start_approach_lock_distance_m,
                cfg.goal_approach_lock_distance_m,
            ),
        ),
    )
    zone_05 = min(0.5, total_len)
    finite_mask = np.isfinite(sampled_curvatures)
    finite_s = s[finite_mask]
    start_curv_max = _zone_max(finite_curv, finite_s, total_len, endpoint_zone, True)
    end_curv_max = _zone_max(finite_curv, finite_s, total_len, endpoint_zone, False)
    start_jump_max = _zone_max(angle_jumps, join_positions, join_total, endpoint_zone, True)
    end_jump_max = _zone_max(angle_jumps, join_positions, join_total, endpoint_zone, False)
    if lightweight:
        intersections_total = 0
        endpoint_intersections = 0
    else:
        intersections_total, endpoint_intersections = _self_intersection_counts(sampled_points, endpoint_zone)
    endpoint_diag = _endpoint_and_terminal_diagnostics(
        sampled_points,
        tangent_headings,
        s,
        cfg,
        context,
    )
    wall_min = 0.0
    wall_mean = 0.0
    wall_p05 = 0.0
    heat_min = -1.0
    heat_mean = -1.0
    heat_p05 = -1.0
    if context is not None and context.wall_clearance_field is not None:
        wall_samples = sample_field_along_path(
            sampled_points, context.wall_clearance_field, cfg.resolution_m_per_cell
        )
        wall_min = _safe_stat_min(wall_samples, default=0.0)
        wall_mean = _safe_stat_mean(wall_samples, default=0.0)
        wall_p05 = _safe_stat_pctl(wall_samples, 5.0, default=0.0)
    if context is not None and context.heat_region_clearance_field is not None:
        heat_samples = sample_field_along_path(
            sampled_points,
            context.heat_region_clearance_field,
            cfg.resolution_m_per_cell,
        )
        heat_min = _safe_stat_min(heat_samples, default=-1.0)
        heat_mean = _safe_stat_mean(heat_samples, default=-1.0)
        heat_p05 = _safe_stat_pctl(heat_samples, 5.0, default=-1.0)

    return {
        "endpointZoneM": float(endpoint_zone),
        "maxTangentJumpDeg": float(np.max(angle_jumps)) if angle_jumps.size else 0.0,
        "maxTangentJumpNearStartDeg": start_jump_max,
        "maxTangentJumpNearEndDeg": end_jump_max,
        "maxTangentJumpFirst0p5mDeg": _zone_max(
            angle_jumps, join_positions, join_total, zone_05, from_start=True
        ),
        "maxTangentJumpLast0p5mDeg": _zone_max(
            angle_jumps, join_positions, join_total, zone_05, from_start=False
        ),
        "maxTangentMagJumpRatio": float(np.max(mag_jumps)) if mag_jumps.size else 0.0,
        "maxCurvature": float(np.max(finite_curv)) if finite_curv.size else 0.0,
        "maxCurvatureNearStart": start_curv_max,
        "maxCurvatureNearEnd": end_curv_max,
        "maxCurvatureFirst0p5m": _zone_max(finite_curv, finite_s, total_len, zone_05, from_start=True),
        "maxCurvatureLast0p5m": _zone_max(finite_curv, finite_s, total_len, zone_05, from_start=False),
        "curvatureP50": curv_pct["p50"],
        "curvatureP90": curv_pct["p90"],
        "curvatureP95": curv_pct["p95"],
        "curvatureP99": curv_pct["p99"],
        "segmentLengthMinM": float(np.min(seg_lengths)) if seg_lengths.size else 0.0,
        "segmentLengthMeanM": float(np.mean(seg_lengths)) if seg_lengths.size else 0.0,
        "segmentLengthMaxM": float(np.max(seg_lengths)) if seg_lengths.size else 0.0,
        "selfIntersectionCount": float(intersections_total),
        "endpointSelfIntersectionCount": float(endpoint_intersections),
        "minWallClearanceM": float(wall_min),
        "meanWallClearanceM": float(wall_mean),
        "p05WallClearanceM": float(wall_p05),
        "minHeatRegionClearanceM": float(heat_min),
        "meanHeatRegionClearanceM": float(heat_mean),
        "p05HeatRegionClearanceM": float(heat_p05),
        **endpoint_diag,
    }


def _terminal_degradation_reasons(
    candidate: dict[str, float],
    raw: dict[str, float],
    cfg: PlannerConfig,
) -> list[str]:
    reasons: list[str] = []
    for prefix, label in (("start", "start"), ("goal", "goal")):
        cand_heat_exp = float(candidate.get(f"{prefix}TerminalHeatExposure", 0.0))
        raw_heat_exp = float(raw.get(f"{prefix}TerminalHeatExposure", 0.0))
        if raw_heat_exp > 1e-9 and cand_heat_exp > raw_heat_exp * 1.15:
            reasons.append(f"{label}_terminal_heat_exposure_worse")

        cand_heat_min = float(candidate.get(f"{prefix}TerminalMinHeatRegionClearanceM", -1.0))
        raw_heat_min = float(raw.get(f"{prefix}TerminalMinHeatRegionClearanceM", -1.0))
        if raw_heat_min >= 0.0 and cand_heat_min >= 0.0 and cand_heat_min < raw_heat_min - 0.06:
            reasons.append(f"{label}_terminal_heat_clearance_worse")

        cand_wall_min = float(candidate.get(f"{prefix}TerminalMinWallClearanceM", 0.0))
        raw_wall_min = float(raw.get(f"{prefix}TerminalMinWallClearanceM", 0.0))
        if raw_wall_min > 0.0 and cand_wall_min < raw_wall_min - 0.05:
            reasons.append(f"{label}_terminal_wall_clearance_worse")

        cand_directness = float(candidate.get(f"{prefix}TerminalDirectnessPenalty", 0.0))
        raw_directness = float(raw.get(f"{prefix}TerminalDirectnessPenalty", 0.0))
        if cand_directness > raw_directness + 0.18:
            reasons.append(f"{label}_terminal_directness_worse")

        cand_curv = float(candidate.get(f"{prefix}TerminalMaxCurvature", 0.0))
        raw_curv = float(raw.get(f"{prefix}TerminalMaxCurvature", 0.0))
        if cand_curv > max(raw_curv + 0.3, raw_curv * 1.18 + 1e-6):
            reasons.append(f"{label}_terminal_curvature_worse")

        cand_hook = float(candidate.get(f"{prefix}TerminalHookOrOvershootFlag", 0.0))
        raw_hook = float(raw.get(f"{prefix}TerminalHookOrOvershootFlag", 0.0))
        if cand_hook > 0.5 and raw_hook <= 0.5:
            reasons.append(f"{label}_terminal_hook_introduced")

        cand_progress = float(candidate.get(f"{prefix}TerminalProgressRatio", 1.0))
        raw_progress = float(raw.get(f"{prefix}TerminalProgressRatio", 1.0))
        if cand_progress < min(raw_progress - 0.08, float(cfg.min_terminal_progress_ratio)):
            reasons.append(f"{label}_terminal_progress_worse")

    return reasons


def _clamp_terminal_metrics_to_raw(candidate: dict[str, float], raw: dict[str, float]) -> dict[str, float]:
    out = dict(candidate)
    for prefix in ("start", "goal"):
        lower_keys = (
            "TerminalHeatExposure",
            "TerminalDirectnessPenalty",
            "TerminalLateralDeviationMeanM",
            "TerminalLateralDeviationMaxM",
            "TerminalMaxCurvature",
            "TerminalProjectionMonotonicViolations",
            "TerminalDistanceMonotonicViolations",
            "TerminalOvershootCount",
            "TerminalHookOrOvershootFlag",
            "TerminalAlignmentErrorDeg",
        )
        higher_keys = (
            "TerminalMinWallClearanceM",
            "TerminalDirectnessScore",
            "TerminalProgressRatio",
        )
        for key in lower_keys:
            full = f"{prefix}{key}"
            if full not in out or full not in raw:
                continue
            out[full] = float(min(float(out[full]), float(raw[full])))
        for key in higher_keys:
            full = f"{prefix}{key}"
            if full not in out or full not in raw:
                continue
            out[full] = float(max(float(out[full]), float(raw[full])))

        heat_key = f"{prefix}TerminalMinHeatRegionClearanceM"
        if heat_key in out and heat_key in raw:
            cand_val = float(out[heat_key])
            raw_val = float(raw[heat_key])
            if raw_val >= 0.0 and cand_val >= 0.0:
                out[heat_key] = float(max(cand_val, raw_val))

    out["terminalOvershootCount"] = float(
        min(float(out.get("terminalOvershootCount", 0.0)), float(raw.get("terminalOvershootCount", 0.0)))
    )
    out["terminalGoalProjectionMonotonicViolations"] = float(
        min(
            float(out.get("terminalGoalProjectionMonotonicViolations", 0.0)),
            float(raw.get("terminalGoalProjectionMonotonicViolations", 0.0)),
        )
    )
    out["terminalGoalDistanceMonotonicViolations"] = float(
        min(
            float(out.get("terminalGoalDistanceMonotonicViolations", 0.0)),
            float(raw.get("terminalGoalDistanceMonotonicViolations", 0.0)),
        )
    )
    out["terminalProgressRatio"] = float(
        max(float(out.get("terminalProgressRatio", 1.0)), float(raw.get("terminalProgressRatio", 1.0)))
    )
    out["startEndpointAlignmentErrorDeg"] = float(
        min(
            float(out.get("startEndpointAlignmentErrorDeg", 0.0)),
            float(raw.get("startEndpointAlignmentErrorDeg", 0.0)),
        )
    )
    out["endEndpointAlignmentErrorDeg"] = float(
        min(
            float(out.get("endEndpointAlignmentErrorDeg", 0.0)),
            float(raw.get("endEndpointAlignmentErrorDeg", 0.0)),
        )
    )
    return out


def _candidate_worse_than_raw(candidate: dict[str, float], raw: dict[str, float], cfg: PlannerConfig) -> bool:
    if candidate["maxTangentJumpDeg"] > raw["maxTangentJumpDeg"] * cfg.raw_tangent_worse_factor + 0.25:
        return True

    raw_score = raw["curvatureP95"] + raw["maxTangentJumpDeg"] / 8.0
    candidate_score = candidate["curvatureP95"] + candidate["maxTangentJumpDeg"] / 8.0
    if raw_score > 1e-9 and candidate_score > raw_score * cfg.raw_curvature_worse_factor:
        return True

    if _terminal_degradation_reasons(candidate, raw, cfg):
        return True

    return False


def _needs_refit(
    candidate: dict[str, float],
    raw: dict[str, float],
    cfg: PlannerConfig,
    context: SmoothingContext | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate["maxTangentJumpDeg"] > cfg.max_tangent_jump_deg:
        reasons.append("max_tangent_jump_exceeded")
    if candidate["maxTangentMagJumpRatio"] > cfg.max_tangent_mag_jump_ratio:
        reasons.append("max_tangent_magnitude_jump_exceeded")
    if candidate["curvatureP99"] > cfg.max_curvature * cfg.curvature_spike_factor:
        reasons.append("curvature_spike_exceeded")
    if candidate["maxCurvatureNearStart"] > cfg.max_endpoint_curvature:
        reasons.append("start_endpoint_curvature_exceeded")
    if candidate["maxCurvatureNearEnd"] > cfg.max_endpoint_curvature:
        reasons.append("end_endpoint_curvature_exceeded")
    if candidate["maxTangentJumpNearStartDeg"] > cfg.max_endpoint_tangent_jump_deg:
        reasons.append("start_endpoint_tangent_jump_exceeded")
    if candidate["maxTangentJumpNearEndDeg"] > cfg.max_endpoint_tangent_jump_deg:
        reasons.append("end_endpoint_tangent_jump_exceeded")
    if candidate.get("endpointSelfIntersectionCount", 0.0) > 0.0:
        reasons.append("endpoint_self_intersection")
    if candidate.get("selfIntersectionCount", 0.0) > 0.0:
        reasons.append("self_intersection")
    if candidate.get("startTerminalProjectionMonotonicViolations", 0.0) > 0.0:
        reasons.append("start_terminal_projection_non_monotonic")
    if candidate.get("startTerminalDistanceMonotonicViolations", 0.0) > 0.0:
        reasons.append("start_terminal_distance_non_monotonic")
    if candidate.get("startTerminalOvershootCount", 0.0) > 0.0 and not cfg.allow_terminal_overshoot:
        reasons.append("start_terminal_overshoot_detected")
    if candidate.get("startTerminalProgressRatio", 1.0) < cfg.min_terminal_progress_ratio:
        reasons.append("start_terminal_progress_ratio_low")
    if candidate.get("terminalGoalProjectionMonotonicViolations", 0.0) > 0.0:
        reasons.append("terminal_projection_non_monotonic")
    if candidate.get("terminalGoalDistanceMonotonicViolations", 0.0) > 0.0:
        reasons.append("terminal_distance_non_monotonic")
    if candidate.get("terminalOvershootCount", 0.0) > 0.0 and not cfg.allow_terminal_overshoot:
        reasons.append("terminal_overshoot_detected")
    if candidate.get("terminalProgressRatio", 1.0) < cfg.min_terminal_progress_ratio:
        reasons.append("terminal_progress_ratio_low")
    reasons.extend(_terminal_degradation_reasons(candidate, raw, cfg))
    if context is not None and context.hard_clearance_feasible:
        required = max(0.0, float(context.required_clearance_m))
        if candidate.get("minWallClearanceM", required) < required - 1e-3:
            reasons.append("wall_clearance_hard_constraint_failed")
    clearance_floor = max(0.0, cfg.clearance_refit_threshold_m)
    if clearance_floor > 0.0 and candidate.get("minWallClearanceM", clearance_floor) < clearance_floor:
        reasons.append("wall_clearance_below_refit_threshold")
    if _candidate_worse_than_raw(candidate, raw, cfg):
        reasons.append("worse_than_raw")
    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)
    return len(deduped) > 0, deduped


def smooth_path_to_beziers(
    raw_path_world: list[tuple[float, float]],
    cfg: PlannerConfig,
    context: SmoothingContext | None = None,
) -> SmoothingResult:
    if len(raw_path_world) < 2:
        raise ValueError("Need at least 2 points to smooth.")

    raw_points = np.asarray(raw_path_world, dtype=float)
    raw_points = dedupe_consecutive(raw_points, tol=1e-8)
    if len(raw_points) < 2:
        raise ValueError("Path has no movement after deduplication.")

    # Required: uniform arc-length resampling before fitting.
    resampled = resample_polyline(raw_points, cfg.sample_ds_m)
    resampled = dedupe_consecutive(resampled, tol=1e-8)
    if len(resampled) < 2:
        raise ValueError("Resampled path has insufficient points.")
    if len(resampled) == 2:
        mid = 0.5 * (resampled[0] + resampled[-1])
        resampled = np.vstack([resampled[0], mid, resampled[-1]])

    raw_diag = _polyline_diagnostics(
        resampled,
        endpoint_zone_m=max(
            cfg.endpoint_zone_m,
            cfg.start_approach_lock_distance_m,
            cfg.goal_approach_lock_distance_m,
        ),
        cfg=cfg,
        context=context,
    )

    attempts: list[dict[str, Any]] = []
    best_result: tuple[list[BezierSegment], np.ndarray, np.ndarray] | None = None
    best_diag: dict[str, float] | None = None
    best_score = float("inf")
    best_accepted_result: tuple[list[BezierSegment], np.ndarray, np.ndarray] | None = None
    best_accepted_diag: dict[str, float] | None = None
    best_accepted_score = float("inf")
    accepted_attempt = -1
    terminal_guard_triggered = False
    terminal_guard_reasons: set[str] = set()

    runtime_fast = bool(cfg.fast_runtime)
    attempt_limit = cfg.max_smoothing_refits
    if runtime_fast:
        attempt_limit = max(1, min(cfg.max_smoothing_refits, cfg.runtime_fast_max_refits))
    lightweight_diag = runtime_fast

    for attempt in range(attempt_limit):
        smoothing_scale = cfg.spline_smoothing * (cfg.spline_smoothing_growth**attempt)
        spline_x, spline_y, total = _fit_centerline_with_constraints(
            resampled=resampled,
            cfg=cfg,
            smoothing_scale=smoothing_scale,
            attempt=attempt,
            context=context,
        )
        segments, anchors, tangent_vectors = _build_bezier_chain_from_centerline(
            spline_x=spline_x,
            spline_y=spline_y,
            total_len=total,
            cfg=cfg,
            attempt=attempt,
        )
        diag = _bezier_chain_diagnostics(
            segments,
            cfg.sample_ds_m,
            cfg,
            context=context,
            lightweight=lightweight_diag,
        )
        reject, reasons = _needs_refit(diag, raw_diag, cfg, context=context)
        if reject:
            for reason in reasons:
                if "terminal" in reason:
                    terminal_guard_triggered = True
                    terminal_guard_reasons.add(str(reason))

        attempt_entry: dict[str, Any] = {
            "attempt": int(attempt),
            "splineSmoothingScale": float(smoothing_scale),
            "diagnostics": diag,
            "rejected": bool(reject),
            "reasons": reasons,
        }
        attempts.append(attempt_entry)

        score = (
            5.0 * diag["maxTangentJumpDeg"]
            + 200.0 * diag["maxTangentMagJumpRatio"]
            + 3.0 * diag["curvatureP95"]
            + 2.0 * diag["maxCurvature"]
            + 2.5 * diag["maxCurvatureNearStart"]
            + 2.5 * diag["maxCurvatureNearEnd"]
            + 2.0 * diag["maxTangentJumpNearStartDeg"]
            + 2.0 * diag["maxTangentJumpNearEndDeg"]
            + cfg.segment_count_penalty_weight * float(len(segments))
            + cfg.hook_penalty_weight * float(diag.get("terminalOvershootCount", 0.0))
            + cfg.hook_penalty_weight
            * float(diag.get("terminalGoalProjectionMonotonicViolations", 0.0))
            + cfg.hook_penalty_weight * 0.8 * float(diag.get("selfIntersectionCount", 0.0))
            + 6.0 * float(diag.get("startTerminalDirectnessPenalty", 0.0))
            + 8.0 * float(diag.get("goalTerminalDirectnessPenalty", 0.0))
            + 2.2 * float(diag.get("startTerminalLateralDeviationMeanM", 0.0))
            + 3.0 * float(diag.get("goalTerminalLateralDeviationMeanM", 0.0))
            + cfg.hook_penalty_weight * float(diag.get("startTerminalHookOrOvershootFlag", 0.0))
            + cfg.hook_penalty_weight * float(diag.get("goalTerminalHookOrOvershootFlag", 0.0))
            + 2.2 * float(diag.get("startTerminalMaxCurvature", 0.0))
            + 2.6 * float(diag.get("goalTerminalMaxCurvature", 0.0))
        )
        if context is not None:
            clearance_deficit = max(0.0, context.required_clearance_m - diag.get("minWallClearanceM", 0.0))
            score += 18.0 * clearance_deficit
            if math.isfinite(diag.get("p05HeatRegionClearanceM", float("inf"))):
                score += 0.4 / max(0.05, float(diag["p05HeatRegionClearanceM"]))
            goal_heat_min = float(diag.get("goalTerminalMinHeatRegionClearanceM", -1.0))
            start_heat_min = float(diag.get("startTerminalMinHeatRegionClearanceM", -1.0))
            if goal_heat_min >= 0.0:
                score += 0.5 / max(0.05, goal_heat_min)
            if start_heat_min >= 0.0:
                score += 0.35 / max(0.05, start_heat_min)
            score += 0.35 * float(diag.get("goalTerminalHeatExposure", 0.0))
            score += 0.25 * float(diag.get("startTerminalHeatExposure", 0.0))
        if score < best_score:
            best_score = score
            best_result = (segments, anchors, tangent_vectors)
            best_diag = diag

        if not reject:
            if score < best_accepted_score:
                best_accepted_score = score
                best_accepted_result = (segments, anchors, tangent_vectors)
                best_accepted_diag = diag
                accepted_attempt = attempt
            if runtime_fast and attempt == 0:
                break

    if best_result is None or best_diag is None:
        raise RuntimeError("Failed to generate any Bezier smoothing candidate.")

    used_terminal_safe_dense_fallback = False
    dense_fallback_diag: dict[str, Any] = {}
    dense_fallback_smoothing_accepted = False
    dense_fallback_smoothing_rejected = False
    if best_accepted_result is not None and best_accepted_diag is not None:
        segments, anchors, tangent_vectors = best_accepted_result
        final_diag = best_accepted_diag
    else:
        if terminal_guard_triggered:
            segments, anchors, tangent_vectors, dense_fallback_diag = _terminal_safe_dense_fallback_bezier_chain(
                resampled,
                cfg,
                context=context,
                trigger_reasons=sorted(terminal_guard_reasons),
            )
            final_diag = _bezier_chain_diagnostics(
                segments,
                cfg.sample_ds_m,
                cfg,
                context=context,
                lightweight=runtime_fast,
            )
            used_terminal_safe_dense_fallback = True
            dense_fallback_smoothing_accepted = bool(dense_fallback_diag.get("smoothingAccepted", False))
            dense_fallback_smoothing_rejected = bool(dense_fallback_diag.get("smoothingRejected", False))
            accepted_attempt = -2
            terminal_guard_reasons.add("terminal_safe_dense_fallback")
        else:
            segments, anchors, tangent_vectors = best_result
            final_diag = _bezier_chain_diagnostics(
                segments,
                cfg.sample_ds_m,
                cfg,
                context=context,
                lightweight=runtime_fast,
            )

    diagnostics: dict[str, Any] = {
        "attemptCount": int(len(attempts)),
        "acceptedAttempt": int(accepted_attempt),
        "refitTriggered": bool(accepted_attempt > 0 or accepted_attempt < 0),
        "terminalSafeDenseFallbackUsed": bool(used_terminal_safe_dense_fallback),
        "terminalSafeDenseRawFallbackUsed": bool(
            used_terminal_safe_dense_fallback and not dense_fallback_smoothing_accepted
        ),
        "terminalSafeDenseFallbackSmoothedAccepted": bool(dense_fallback_smoothing_accepted),
        "terminalSafeDenseFallbackSmoothedRejected": bool(dense_fallback_smoothing_rejected),
        "terminalSafeDenseFallbackDiagnostics": dict(dense_fallback_diag),
        # Legacy key retained for compatibility with older consumers.
        "terminalSafeRawFallbackUsed": bool(used_terminal_safe_dense_fallback),
        "terminalDegradationRefitTriggered": bool(terminal_guard_triggered),
        "terminalDegradationReasons": sorted(terminal_guard_reasons),
        "rawPathDiagnostics": raw_diag,
        "smoothedPathDiagnostics": final_diag,
        "attempts": attempts,
    }

    return SmoothingResult(
        segments=segments,
        anchors=anchors,
        tangent_vectors=tangent_vectors,
        diagnostics=diagnostics,
    )
