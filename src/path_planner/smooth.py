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


def _polyline_diagnostics(points: np.ndarray, endpoint_zone_m: float) -> dict[str, float]:
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.empty((0,))
    total_len = float(np.sum(seg_lengths)) if seg_lengths.size else 0.0
    zone_05 = min(0.5, total_len)
    tangent_jumps, tangent_jump_s = _tangent_jumps_for_polyline(points)
    curvatures, curvature_s = _curvature_samples_for_polyline(points)
    pct = _percentiles(curvatures)
    return {
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

    start_dir = _resolved_start_dir(cfg, smooth_points[1] - smooth_points[0])
    end_dir = _resolved_goal_dir(cfg, smooth_points[-1] - smooth_points[-2])

    start_zone = _start_endpoint_zone_length(total, cfg, attempt)
    goal_zone = _goal_endpoint_zone_length(total, cfg, attempt)
    if (start_zone > 1e-9 or goal_zone > 1e-9) and len(smooth_points) > 2:
        blend_power = max(1.0, float(cfg.endpoint_heading_blend_power))
        blended = smooth_points.copy()
        start_lock = max(0.0, float(cfg.start_approach_lock_distance_m))
        goal_lock = max(0.0, float(cfg.goal_approach_lock_distance_m))
        for i in range(1, len(blended) - 1):
            dist_start = float(s[i])
            dist_end = total - dist_start
            if _resolved_start_approach_heading_deg(cfg) is not None:
                w_start = _endpoint_blend_weight(dist_start, start_zone, blend_power)
                if start_lock > 1e-9 and dist_start <= start_lock:
                    t_lock = 1.0 - dist_start / start_lock
                    w_start = max(w_start, 0.35 + 0.65 * (t_lock * t_lock * (3.0 - 2.0 * t_lock)))
                if w_start > 0.0:
                    desired = blended[0] + start_dir * dist_start
                    blended[i] = (1.0 - w_start) * blended[i] + w_start * desired
            if _resolved_goal_approach_heading_deg(cfg) is not None:
                w_end = _endpoint_blend_weight(dist_end, goal_zone, blend_power)
                if goal_lock > 1e-9 and dist_end <= goal_lock:
                    t_lock = 1.0 - dist_end / goal_lock
                    w_end = max(w_end, 0.35 + 0.65 * (t_lock * t_lock * (3.0 - 2.0 * t_lock)))
                if w_end > 0.0:
                    desired = blended[-1] - end_dir * dist_end
                    blended[i] = (1.0 - w_end) * blended[i] + w_end * desired

        smooth_points = blended
        s = cumulative_arc_length(smooth_points)
        total = float(s[-1])
        if total <= 1e-9:
            raise ValueError("Spline source path collapsed after endpoint blending.")

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
            dirs[i] = _resolved_start_dir(cfg, anchors[1] - anchors[0])
        elif i == n - 1:
            dirs[i] = _resolved_goal_dir(cfg, anchors[-1] - anchors[-2])
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

    s_anchor = cumulative_arc_length(anchors)
    total_len = float(s_anchor[-1]) if len(s_anchor) else 0.0
    start_zone = _start_endpoint_zone_length(total_len, cfg, attempt)
    goal_zone = _goal_endpoint_zone_length(total_len, cfg, attempt)
    blend_power = max(1.0, float(cfg.endpoint_heading_blend_power))

    start_heading = _resolved_start_approach_heading_deg(cfg)
    goal_heading = _resolved_goal_approach_heading_deg(cfg)

    if start_zone > 1e-9 and start_heading is not None:
        start_ref = heading_to_unit(start_heading)
        for i in range(n):
            w = _endpoint_blend_weight(float(s_anchor[i]), start_zone, blend_power)
            if w > 0.0:
                dirs[i] = normalize((1.0 - w) * dirs[i] + w * start_ref)

    if goal_zone > 1e-9 and goal_heading is not None:
        end_ref = heading_to_unit(goal_heading)
        for i in range(n):
            dist_end = total_len - float(s_anchor[i])
            w = _endpoint_blend_weight(dist_end, goal_zone, blend_power)
            if w > 0.0:
                dirs[i] = normalize((1.0 - w) * dirs[i] + w * end_ref)

    if start_heading is not None:
        dirs[0] = heading_to_unit(start_heading)
    if goal_heading is not None:
        dirs[-1] = heading_to_unit(goal_heading)

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
    start_heading = _resolved_start_approach_heading_deg(cfg)
    goal_heading = _resolved_goal_approach_heading_deg(cfg)
    if start_heading is not None:
        dirs[0] = heading_to_unit(start_heading)
    if goal_heading is not None:
        dirs[-1] = heading_to_unit(goal_heading)

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

    terminal_overshoot_count = 0
    terminal_goal_proj_violations = 0
    terminal_goal_dist_violations = 0
    terminal_progress_ratio = 1.0
    if (
        context is not None
        and len(points) >= 3
        and goal_heading is not None
        and goal_zone > 1e-9
    ):
        goal_xy = np.asarray(context.goal_xy, dtype=float)
        goal_dir = heading_to_unit(goal_heading)
        zone_start_s = max(0.0, total_len - goal_zone)
        zone_mask = s >= zone_start_s
        zone_pts = points[zone_mask]
        if len(zone_pts) >= 2:
            to_goal = goal_xy[None, :] - zone_pts
            proj = np.asarray(np.dot(to_goal, goal_dir), dtype=float)
            dists = np.linalg.norm(to_goal, axis=1)
            proj_delta = np.diff(proj)
            dist_delta = np.diff(dists)
            tol = max(0.01, 0.15 * cfg.sample_ds_m)
            terminal_goal_proj_violations = int(np.sum(proj_delta > tol))
            terminal_goal_dist_violations = int(np.sum(dist_delta > tol))
            terminal_overshoot_count = int(
                np.sum(proj[:-1] < -max(cfg.endpoint_overshoot_tolerance_m, 1e-6))
            )
            forward = float(proj[0] - proj[-1])
            total_motion = float(np.sum(np.abs(proj_delta)))
            if total_motion > 1e-9:
                terminal_progress_ratio = max(0.0, min(1.0, forward / total_motion))

    return {
        "startEndpointAlignmentErrorDeg": float(start_err),
        "endEndpointAlignmentErrorDeg": float(end_err),
        "maxStartLockHeadingErrorDeg": float(start_zone_heading_max),
        "maxGoalLockHeadingErrorDeg": float(goal_zone_heading_max),
        "maxStartLockHeadingErrorRawDeg": float(start_zone_heading_raw_max),
        "maxGoalLockHeadingErrorRawDeg": float(goal_zone_heading_raw_max),
        "terminalOvershootCount": float(terminal_overshoot_count),
        "terminalGoalProjectionMonotonicViolations": float(terminal_goal_proj_violations),
        "terminalGoalDistanceMonotonicViolations": float(terminal_goal_dist_violations),
        "terminalProgressRatio": float(terminal_progress_ratio),
    }


def _bezier_chain_diagnostics(
    segments: list[BezierSegment],
    sample_ds_m: float,
    cfg: PlannerConfig,
    context: SmoothingContext | None = None,
) -> dict[str, float]:
    sampled_points, tangent_headings, sampled_curvatures = sample_bezier_chain(
        segments,
        sample_per_segment=max(35, int(1.0 / max(sample_ds_m, 1e-3))),
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


def _candidate_worse_than_raw(candidate: dict[str, float], raw: dict[str, float], cfg: PlannerConfig) -> bool:
    if candidate["maxTangentJumpDeg"] > raw["maxTangentJumpDeg"] * cfg.raw_tangent_worse_factor + 0.25:
        return True

    raw_score = raw["curvatureP95"] + raw["maxTangentJumpDeg"] / 8.0
    candidate_score = candidate["curvatureP95"] + candidate["maxTangentJumpDeg"] / 8.0
    if raw_score > 1e-9 and candidate_score > raw_score * cfg.raw_curvature_worse_factor:
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
    if candidate.get("startEndpointAlignmentErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg:
        reasons.append("start_endpoint_alignment_exceeded")
    if candidate.get("endEndpointAlignmentErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg:
        reasons.append("end_endpoint_alignment_exceeded")
    if candidate.get("maxStartLockHeadingErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg * 1.6:
        reasons.append("start_lock_alignment_exceeded")
    if candidate.get("maxGoalLockHeadingErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg * 1.6:
        reasons.append("goal_lock_alignment_exceeded")
    if candidate.get("endpointSelfIntersectionCount", 0.0) > 0.0:
        reasons.append("endpoint_self_intersection")
    if candidate.get("selfIntersectionCount", 0.0) > 0.0:
        reasons.append("self_intersection")
    if candidate.get("terminalGoalProjectionMonotonicViolations", 0.0) > 0.0:
        reasons.append("terminal_projection_non_monotonic")
    if candidate.get("terminalGoalDistanceMonotonicViolations", 0.0) > 0.0:
        reasons.append("terminal_distance_non_monotonic")
    if candidate.get("terminalOvershootCount", 0.0) > 0.0 and not cfg.allow_terminal_overshoot:
        reasons.append("terminal_overshoot_detected")
    if candidate.get("terminalProgressRatio", 1.0) < cfg.min_terminal_progress_ratio:
        reasons.append("terminal_progress_ratio_low")
    if context is not None and context.hard_clearance_feasible:
        required = max(0.0, float(context.required_clearance_m))
        if candidate.get("minWallClearanceM", required) < required - 1e-3:
            reasons.append("wall_clearance_hard_constraint_failed")
    clearance_floor = max(0.0, cfg.clearance_refit_threshold_m)
    if clearance_floor > 0.0 and candidate.get("minWallClearanceM", clearance_floor) < clearance_floor:
        reasons.append("wall_clearance_below_refit_threshold")
    if _candidate_worse_than_raw(candidate, raw, cfg):
        reasons.append("worse_than_raw")
    return len(reasons) > 0, reasons


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
    )

    attempts: list[dict[str, Any]] = []
    best_result: tuple[list[BezierSegment], np.ndarray, np.ndarray] | None = None
    best_diag: dict[str, float] | None = None
    best_score = float("inf")
    best_accepted_result: tuple[list[BezierSegment], np.ndarray, np.ndarray] | None = None
    best_accepted_diag: dict[str, float] | None = None
    best_accepted_score = float("inf")
    accepted_attempt = -1

    for attempt in range(cfg.max_smoothing_refits):
        smoothing_scale = cfg.spline_smoothing * (cfg.spline_smoothing_growth**attempt)
        spline_x, spline_y, total = _fit_centerline_with_constraints(
            resampled=resampled,
            cfg=cfg,
            smoothing_scale=smoothing_scale,
            attempt=attempt,
        )
        segments, anchors, tangent_vectors = _build_bezier_chain_from_centerline(
            spline_x=spline_x,
            spline_y=spline_y,
            total_len=total,
            cfg=cfg,
            attempt=attempt,
        )
        diag = _bezier_chain_diagnostics(segments, cfg.sample_ds_m, cfg, context=context)
        reject, reasons = _needs_refit(diag, raw_diag, cfg, context=context)

        attempts.append(
            {
                "attempt": int(attempt),
                "splineSmoothingScale": float(smoothing_scale),
                "diagnostics": diag,
                "rejected": bool(reject),
                "reasons": reasons,
            }
        )

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
        )
        if context is not None:
            clearance_deficit = max(0.0, context.required_clearance_m - diag.get("minWallClearanceM", 0.0))
            score += 18.0 * clearance_deficit
            if math.isfinite(diag.get("p05HeatRegionClearanceM", float("inf"))):
                score += 0.4 / max(0.05, float(diag["p05HeatRegionClearanceM"]))
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

    if best_result is None or best_diag is None:
        raise RuntimeError("Failed to generate any Bezier smoothing candidate.")

    if best_accepted_result is not None and best_accepted_diag is not None:
        segments, anchors, tangent_vectors = best_accepted_result
        final_diag = best_accepted_diag
    else:
        segments, anchors, tangent_vectors = best_result
        final_diag = _bezier_chain_diagnostics(segments, cfg.sample_ds_m, cfg, context=context)

    diagnostics: dict[str, Any] = {
        "attemptCount": int(len(attempts)),
        "acceptedAttempt": int(accepted_attempt),
        "refitTriggered": bool(accepted_attempt > 0 or accepted_attempt < 0),
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
