from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clearance import sample_field_along_path
from .config import PlannerConfig
from .extract import extract_path_from_cost_to_go
from .geometry import (
    bilinear_sample_grid_vectorized,
    dedupe_consecutive,
    polyline_length,
    resample_polyline,
    sample_bezier_chain,
)
from .heatmap import world_to_cell_index
from .models import BezierSegment
from .performance import StageTimer
from .rotation import compute_holonomic_rotation_profile, progress_from_points
from .runtime import PlannerRuntime
from .smooth import SmoothingContext, SmoothingResult, _polyline_diagnostics, smooth_path_to_beziers
from .speed_profile import compute_speed_profile


@dataclass
class PlannerArtifacts:
    t_field: np.ndarray
    cost_density: np.ndarray
    blocked: np.ndarray
    planning_blocked: np.ndarray
    wall_clearance_m: np.ndarray
    heat_region_clearance_m: np.ndarray
    required_clearance_m: float
    hard_clearance_feasible: bool
    raw_path_world: list[tuple[float, float]]
    raw_path_world_resampled: np.ndarray
    raw_path_tangent_headings_rad: np.ndarray
    raw_path_holonomic_rotations_rad: np.ndarray
    raw_speed_profile: list[dict[str, float]]
    raw_path_cells: list[tuple[int, int]]
    final_geometry_source: str
    bezier_segments: list[BezierSegment]
    sampled_smoothed_points: np.ndarray
    sampled_path_tangent_headings_rad: np.ndarray
    sampled_holonomic_rotations_rad: np.ndarray
    sampled_curvatures: np.ndarray
    speed_profile: list[dict[str, float]]
    smoothing_diagnostics: dict[str, Any]
    stage_timings_ms: dict[str, float]
    backend_status: dict[str, str]
    runtime_cache_stats: dict[str, int]
    summary: dict[str, Any]


@dataclass(frozen=True)
class _PathMetrics:
    length_m: float
    integrated_cost: float
    objective_cost: float
    min_wall_clearance_m: float
    min_heat_region_clearance_m: float
    max_curvature: float
    p95_curvature: float


def _integrate_heat_cost_along_polyline(
    points_world: np.ndarray,
    cost_density: np.ndarray,
    resolution_m: float,
) -> float:
    if len(points_world) < 2:
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
    w = bilinear_sample_grid_vectorized(cost_density, x, y)
    w0 = w[:-1]
    w1 = w[1:]
    valid = valid_seg & np.isfinite(w0) & np.isfinite(w1)
    if not np.any(valid):
        return 0.0
    return float(np.sum(0.5 * (w0[valid] + w1[valid]) * ds[valid]))


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= 1e-9:
        return float(default)
    return float(num / den)


def _diag_float(diag: dict[str, Any], key: str, default: float) -> float:
    try:
        out = float(diag.get(key, default))
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _polyline_headings(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n <= 0:
        return np.empty((0,), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float)
    seg = np.diff(points, axis=0)
    seg_head = np.arctan2(seg[:, 1], seg[:, 0])
    headings = np.empty((n,), dtype=float)
    headings[0] = float(seg_head[0])
    headings[-1] = float(seg_head[-1])
    if n > 2:
        avg = seg[:-1] + seg[1:]
        avg_norm = np.linalg.norm(avg, axis=1)
        mid = seg_head[:-1].copy()
        valid = avg_norm > 1e-9
        if np.any(valid):
            mid[valid] = np.arctan2(avg[valid, 1], avg[valid, 0])
        headings[1:-1] = mid
    return headings


def _polyline_curvatures(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n <= 0:
        return np.empty((0,), dtype=float)
    out = np.zeros((n,), dtype=float)
    if n < 3:
        return out
    for i in range(1, n - 1):
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
        out[i] = 2.0 * abs(cross) / denom
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def _build_raw_reference(
    raw_path_world: list[tuple[float, float]],
    cfg: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = dedupe_consecutive(np.asarray(raw_path_world, dtype=float), tol=1e-9)
    if len(points) < 2:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )
    ds = float(cfg.raw_reference_sample_ds_m)
    if ds <= 1e-9:
        ds = float(cfg.sample_ds_m)
    ds = max(0.02, ds)
    resampled = dedupe_consecutive(resample_polyline(points, ds), tol=1e-9)
    if len(resampled) < 2:
        resampled = points
    return resampled, _polyline_headings(resampled), _polyline_curvatures(resampled)


def _raw_polyline_to_beziers(
    points_world: np.ndarray,
    cfg: PlannerConfig,
    target_ds: float | None = None,
) -> list[BezierSegment]:
    points = dedupe_consecutive(np.asarray(points_world, dtype=float), tol=1e-9)
    if len(points) < 2:
        return []
    if target_ds is None:
        target_ds = float(cfg.raw_linear_bezier_ds_m)
    ds = max(0.03, float(target_ds))
    anchors = dedupe_consecutive(resample_polyline(points, ds), tol=1e-9)
    if len(anchors) < 2:
        anchors = points
    max_segments = max(1, int(cfg.max_bezier_segments))
    if len(anchors) - 1 > max_segments:
        idx = np.linspace(0, len(anchors) - 1, max_segments + 1, dtype=int)
        anchors = anchors[np.unique(idx)]
    segs: list[BezierSegment] = []
    for i in range(len(anchors) - 1):
        p0 = anchors[i]
        p3 = anchors[i + 1]
        d = p3 - p0
        p1 = p0 + d / 3.0
        p2 = p0 + 2.0 * d / 3.0
        segs.append(BezierSegment(p0=p0.copy(), p1=p1.copy(), p2=p2.copy(), p3=p3.copy()))
    return segs

def _path_metrics(
    points: np.ndarray,
    curvatures: np.ndarray,
    cost_density: np.ndarray,
    planning_density: np.ndarray,
    wall_clearance_m: np.ndarray,
    heat_region_clearance_m: np.ndarray,
    resolution_m_per_cell: float,
) -> _PathMetrics:
    length_m = polyline_length(points)
    integrated_cost = _integrate_heat_cost_along_polyline(points, cost_density, resolution_m_per_cell)
    objective_cost = _integrate_heat_cost_along_polyline(points, planning_density, resolution_m_per_cell)

    wall_samples = sample_field_along_path(points, wall_clearance_m, resolution_m_per_cell)
    finite_wall = wall_samples[np.isfinite(wall_samples)]
    min_wall = float(np.min(finite_wall)) if finite_wall.size else 0.0

    heat_samples = sample_field_along_path(points, heat_region_clearance_m, resolution_m_per_cell)
    finite_heat = heat_samples[np.isfinite(heat_samples)]
    min_heat = float(np.min(finite_heat)) if finite_heat.size else -1.0

    curv = np.asarray(curvatures, dtype=float)
    curv = curv[np.isfinite(curv)]
    max_curvature = float(np.max(curv)) if curv.size else 0.0
    p95_curvature = float(np.percentile(curv, 95)) if curv.size else 0.0
    return _PathMetrics(
        length_m=float(length_m),
        integrated_cost=float(integrated_cost),
        objective_cost=float(objective_cost),
        min_wall_clearance_m=float(min_wall),
        min_heat_region_clearance_m=float(min_heat),
        max_curvature=float(max_curvature),
        p95_curvature=float(p95_curvature),
    )


def _default_smoothing_diag(mode: str) -> dict[str, Any]:
    return {
        "attemptCount": 0,
        "acceptedAttempt": -1,
        "refitTriggered": False,
        "rawPathDiagnostics": {},
        "smoothedPathDiagnostics": {},
        "attempts": [],
        "mode": mode,
    }


def _endpoint_zone_for_diag(cfg: PlannerConfig) -> float:
    return max(
        float(cfg.endpoint_zone_m),
        float(cfg.start_approach_lock_distance_m),
        float(cfg.goal_approach_lock_distance_m),
    )


def _raw_vs_final(
    raw_metrics: _PathMetrics,
    final_metrics: _PathMetrics,
    raw_diag: dict[str, Any],
    final_diag: dict[str, Any],
) -> dict[str, float | bool]:
    raw_heat_min = float(raw_metrics.min_heat_region_clearance_m)
    final_heat_min = float(final_metrics.min_heat_region_clearance_m)
    heat_clear_delta = 0.0
    if raw_heat_min >= 0.0 and final_heat_min >= 0.0:
        heat_clear_delta = final_heat_min - raw_heat_min

    raw_start_exp = _diag_float(raw_diag, "startTerminalHeatExposure", 0.0)
    raw_goal_exp = _diag_float(raw_diag, "goalTerminalHeatExposure", 0.0)
    final_start_exp = _diag_float(final_diag, "startTerminalHeatExposure", 0.0)
    final_goal_exp = _diag_float(final_diag, "goalTerminalHeatExposure", 0.0)

    raw_start_hook = _diag_float(raw_diag, "startTerminalHookOrOvershootFlag", 0.0)
    raw_goal_hook = _diag_float(raw_diag, "goalTerminalHookOrOvershootFlag", 0.0)
    final_start_hook = _diag_float(final_diag, "startTerminalHookOrOvershootFlag", 0.0)
    final_goal_hook = _diag_float(final_diag, "goalTerminalHookOrOvershootFlag", 0.0)

    return {
        "heatCostRatio": _safe_ratio(final_metrics.integrated_cost, raw_metrics.integrated_cost, default=1.0),
        "objectiveCostRatio": _safe_ratio(final_metrics.objective_cost, raw_metrics.objective_cost, default=1.0),
        "lengthRatio": _safe_ratio(final_metrics.length_m, raw_metrics.length_m, default=1.0),
        "maxCurvatureRatio": _safe_ratio(final_metrics.max_curvature, raw_metrics.max_curvature, default=1.0),
        "curvatureP95Ratio": _safe_ratio(final_metrics.p95_curvature, raw_metrics.p95_curvature, default=1.0),
        "minWallClearanceDeltaM": float(final_metrics.min_wall_clearance_m - raw_metrics.min_wall_clearance_m),
        "minHeatRegionClearanceDeltaM": float(heat_clear_delta),
        "maxCurvatureDelta": float(final_metrics.max_curvature - raw_metrics.max_curvature),
        "curvatureP95Delta": float(final_metrics.p95_curvature - raw_metrics.p95_curvature),
        "startTerminalDirectnessDelta": float(
            _diag_float(final_diag, "startTerminalDirectnessScore", 1.0)
            - _diag_float(raw_diag, "startTerminalDirectnessScore", 1.0)
        ),
        "goalTerminalDirectnessDelta": float(
            _diag_float(final_diag, "goalTerminalDirectnessScore", 1.0)
            - _diag_float(raw_diag, "goalTerminalDirectnessScore", 1.0)
        ),
        "startTerminalHeatExposureRatio": _safe_ratio(final_start_exp, raw_start_exp, default=1.0),
        "goalTerminalHeatExposureRatio": _safe_ratio(final_goal_exp, raw_goal_exp, default=1.0),
        "startTerminalHookIntroduced": bool(final_start_hook > 0.5 and raw_start_hook <= 0.5),
        "goalTerminalHookIntroduced": bool(final_goal_hook > 0.5 and raw_goal_hook <= 0.5),
        "terminalOvershootDelta": float(
            _diag_float(final_diag, "terminalOvershootCount", 0.0)
            - _diag_float(raw_diag, "terminalOvershootCount", 0.0)
        ),
    }


def _raw_preferred_reasons(comparison: dict[str, float | bool], cfg: PlannerConfig) -> list[str]:
    reasons: list[str] = []
    eps = 1e-4
    heat_ratio = float(comparison.get("heatCostRatio", 1.0))
    objective_ratio = float(comparison.get("objectiveCostRatio", 1.0))
    length_ratio = float(comparison.get("lengthRatio", 1.0))
    wall_delta = float(comparison.get("minWallClearanceDeltaM", 0.0))
    heat_clear_delta = float(comparison.get("minHeatRegionClearanceDeltaM", 0.0))
    max_curv_ratio = float(comparison.get("maxCurvatureRatio", 1.0))
    p95_curv_ratio = float(comparison.get("curvatureP95Ratio", 1.0))
    max_curv_delta = float(comparison.get("maxCurvatureDelta", 0.0))
    p95_curv_delta = float(comparison.get("curvatureP95Delta", 0.0))

    if heat_ratio > float(cfg.raw_preferred_max_heat_cost_ratio) + eps:
        reasons.append("integrated_heat_cost_worse_than_raw")
    if objective_ratio > float(cfg.raw_preferred_max_objective_cost_ratio) + eps:
        reasons.append("objective_cost_worse_than_raw")
    if wall_delta < -float(cfg.raw_preferred_max_wall_clearance_drop_m) - eps:
        reasons.append("wall_clearance_worse_than_raw")
    if heat_clear_delta < -float(cfg.raw_preferred_max_heat_region_clearance_drop_m) - eps:
        reasons.append("heat_region_clearance_worse_than_raw")
    if (
        max_curv_delta > float(cfg.raw_preferred_max_curvature_abs_increase)
        or (
            max_curv_ratio > float(cfg.raw_preferred_max_curvature_ratio)
            and max_curv_delta > 0.05
        )
    ):
        reasons.append("max_curvature_worse_than_raw")
    if (
        p95_curv_delta > 0.12
        or (
            p95_curv_ratio > float(cfg.raw_preferred_max_curvature_ratio)
            and p95_curv_delta > 0.05
        )
    ):
        reasons.append("curvature_profile_worse_than_raw")
    if (
        length_ratio > float(cfg.raw_preferred_max_length_ratio) + eps
        and heat_ratio > float(cfg.raw_preferred_length_cost_improve_ratio) + eps
        and objective_ratio > float(cfg.raw_preferred_length_cost_improve_ratio) + eps
    ):
        reasons.append("path_length_inflation_without_cost_benefit")

    if float(comparison.get("startTerminalDirectnessDelta", 0.0)) < -float(cfg.raw_preferred_max_terminal_directness_drop) - eps:
        reasons.append("start_terminal_directness_worse_than_raw")
    if float(comparison.get("goalTerminalDirectnessDelta", 0.0)) < -float(cfg.raw_preferred_max_terminal_directness_drop) - eps:
        reasons.append("goal_terminal_directness_worse_than_raw")
    if float(comparison.get("startTerminalHeatExposureRatio", 1.0)) > float(cfg.raw_preferred_max_terminal_heat_exposure_ratio) + eps:
        reasons.append("start_terminal_heat_exposure_worse_than_raw")
    if float(comparison.get("goalTerminalHeatExposureRatio", 1.0)) > float(cfg.raw_preferred_max_terminal_heat_exposure_ratio) + eps:
        reasons.append("goal_terminal_heat_exposure_worse_than_raw")
    if bool(comparison.get("startTerminalHookIntroduced", False)):
        reasons.append("start_terminal_hook_introduced")
    if bool(comparison.get("goalTerminalHookIntroduced", False)):
        reasons.append("goal_terminal_hook_introduced")
    if float(comparison.get("terminalOvershootDelta", 0.0)) > 0.5:
        reasons.append("terminal_overshoot_worse_than_raw")

    out: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        out.append(reason)
    return out


def _planning_block_cache_key(
    blocked: np.ndarray,
    wall_clearance: np.ndarray,
    required_clearance_m: float,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    hard_feasible: bool,
    enforce_hard_clearance: bool,
) -> tuple[Any, ...]:
    if not (hard_feasible and enforce_hard_clearance):
        return ("base_blocked_only",)
    sr, sc = start_rc
    gr, gc = goal_rc
    needs_start_override = bool(blocked[sr, sc] or wall_clearance[sr, sc] < required_clearance_m)
    needs_goal_override = bool(blocked[gr, gc] or wall_clearance[gr, gc] < required_clearance_m)
    return (
        "hard_clearance",
        int(needs_start_override),
        start_rc if needs_start_override else (-1, -1),
        int(needs_goal_override),
        goal_rc if needs_goal_override else (-1, -1),
    )


def run_planner(
    heat: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    cfg: PlannerConfig,
    blocked_mask: np.ndarray | None = None,
    runtime: PlannerRuntime | None = None,
    environment_id: str | int | None = None,
) -> PlannerArtifacts:
    np.random.seed(cfg.deterministic_seed)
    timer = StageTimer(enabled=cfg.stage_timing_enabled)
    timer.start_total()
    local_runtime = runtime if runtime is not None else PlannerRuntime(
        max_goal_cache_entries=cfg.max_goal_field_cache_entries
    )

    with timer.stage("preprocess"):
        env = local_runtime.prepare_environment(
            heat=heat,
            cfg=cfg,
            blocked_mask=blocked_mask,
            environment_id=environment_id,
        )
        cost_density = env.cost_density
        blocked = env.blocked
        start_rc = world_to_cell_index(start_xy[0], start_xy[1], cfg.resolution_m_per_cell, heat.shape)
        goal_rc = world_to_cell_index(goal_xy[0], goal_xy[1], cfg.resolution_m_per_cell, heat.shape)

    if blocked[start_rc]:
        raise ValueError("Start cell is blocked.")
    if blocked[goal_rc]:
        raise ValueError("Goal cell is blocked.")

    with timer.stage("clearance_validation"):
        clearance = local_runtime.build_clearance_layers(
            env=env,
            cfg=cfg,
            start_rc=start_rc,
            goal_rc=goal_rc,
        )
        planning_density = local_runtime.planning_density(env, clearance.planning_blocked)
        block_cache_key = _planning_block_cache_key(
            blocked=blocked,
            wall_clearance=clearance.wall_clearance_m,
            required_clearance_m=float(clearance.required_clearance_m),
            start_rc=start_rc,
            goal_rc=goal_rc,
            hard_feasible=bool(clearance.hard_clearance_feasible),
            enforce_hard_clearance=bool(cfg.enforce_hard_clearance_if_feasible),
        )

    with timer.stage("propagation"):
        t_field = local_runtime.get_cost_to_go(
            planning_density=planning_density,
            planning_blocked=clearance.planning_blocked,
            goal_rc=goal_rc,
            cfg=cfg,
            blocked_key=block_cache_key,
        )

    if not np.isfinite(t_field[start_rc]):
        raise RuntimeError(
            "No finite path from START to GOAL. This indicates blocked-cell disconnection."
        )

    with timer.stage("extraction"):
        raw_path_world, raw_path_cells = extract_path_from_cost_to_go(
            t_field=t_field,
            start_xy=start_xy,
            goal_xy=goal_xy,
            blocked=clearance.planning_blocked,
            cfg=cfg,
        )

    smoothing_context = SmoothingContext(
        start_xy=np.asarray(start_xy, dtype=float),
        goal_xy=np.asarray(goal_xy, dtype=float),
        wall_clearance_field=clearance.wall_clearance_m,
        heat_region_clearance_field=clearance.heat_region_clearance_m,
        required_clearance_m=clearance.required_clearance_m,
        hard_clearance_feasible=clearance.hard_clearance_feasible,
    )
    raw_points, raw_headings, raw_curvatures = _build_raw_reference(raw_path_world, cfg)
    if len(raw_points) < 2:
        raise RuntimeError("Extracted raw path is too short to run.")
    raw_segments = _raw_polyline_to_beziers(
        raw_points,
        cfg,
        target_ds=max(float(cfg.raw_linear_bezier_ds_m), float(cfg.sample_ds_m)),
    )

    sample_per_segment = max(16, int(1.0 / max(cfg.sample_ds_m, 1e-3)))
    with timer.stage("smoothing"):
        if cfg.geometry_mode == "raw_only":
            candidate_segments = raw_segments
            smoothing_diagnostics = _default_smoothing_diag("raw_only")
            candidate_source = "raw_resampled_polyline"
        elif cfg.enable_smoothing:
            smoothing_cfg = cfg
            if cfg.geometry_mode == "raw_preferred":
                smoothing_cfg = cfg.with_updates(
                    spline_smoothing=min(float(cfg.spline_smoothing), 0.95),
                    bezier_target_segment_length_m=min(
                        float(cfg.bezier_target_segment_length_m),
                        max(0.55, 5.0 * float(cfg.sample_ds_m)),
                    ),
                )
            smoothing: SmoothingResult = smooth_path_to_beziers(raw_path_world, smoothing_cfg, context=smoothing_context)
            candidate_segments = smoothing.segments
            smoothing_diagnostics = dict(smoothing.diagnostics)
            candidate_source = "smoothed_bezier"
        else:
            candidate_segments = _raw_polyline_to_beziers(
                raw_points,
                cfg,
                target_ds=max(float(cfg.raw_linear_bezier_ds_m), float(cfg.sample_ds_m)),
            )
            smoothing_diagnostics = _default_smoothing_diag("disabled")
            candidate_source = "raw_linear_bezier"

    if candidate_source == "smoothed_bezier" and candidate_segments:
        cand_points, cand_headings, cand_curvatures = sample_bezier_chain(
            candidate_segments,
            sample_per_segment=sample_per_segment,
        )
        if len(cand_points) < 2:
            cand_points = raw_points.copy()
            cand_headings = raw_headings.copy()
            cand_curvatures = raw_curvatures.copy()
            candidate_segments = raw_segments
            candidate_source = "raw_resampled_polyline"
    else:
        cand_points = raw_points.copy()
        cand_headings = raw_headings.copy()
        cand_curvatures = raw_curvatures.copy()
        candidate_segments = raw_segments

    raw_diag = dict(smoothing_diagnostics.get("rawPathDiagnostics", {}))
    if not raw_diag:
        raw_diag = _polyline_diagnostics(
            raw_points,
            endpoint_zone_m=_endpoint_zone_for_diag(cfg),
            cfg=cfg,
            context=smoothing_context,
        )
    cand_diag = dict(smoothing_diagnostics.get("smoothedPathDiagnostics", {}))
    if not cand_diag:
        cand_diag = _polyline_diagnostics(
            cand_points,
            endpoint_zone_m=_endpoint_zone_for_diag(cfg),
            cfg=cfg,
            context=smoothing_context,
        )

    with timer.stage("clearance_validation"):
        raw_metrics = _path_metrics(
            points=raw_points,
            curvatures=raw_curvatures,
            cost_density=cost_density,
            planning_density=planning_density,
            wall_clearance_m=clearance.wall_clearance_m,
            heat_region_clearance_m=clearance.heat_region_clearance_m,
            resolution_m_per_cell=cfg.resolution_m_per_cell,
        )
        cand_metrics = _path_metrics(
            points=cand_points,
            curvatures=cand_curvatures,
            cost_density=cost_density,
            planning_density=planning_density,
            wall_clearance_m=clearance.wall_clearance_m,
            heat_region_clearance_m=clearance.heat_region_clearance_m,
            resolution_m_per_cell=cfg.resolution_m_per_cell,
        )

    comparison_candidate = _raw_vs_final(raw_metrics, cand_metrics, raw_diag, cand_diag)
    reasons: list[str] = []
    decision = "accepted_final"
    raw_preferred_fallback = False

    final_points = cand_points
    final_headings = cand_headings
    final_curvatures = cand_curvatures
    final_segments = candidate_segments
    final_diag = cand_diag
    final_metrics = cand_metrics
    final_source = candidate_source

    if cfg.geometry_mode == "raw_only":
        decision = "raw_only_selected"
        reasons = ["geometry_mode_raw_only"]
        final_points = raw_points
        final_headings = raw_headings
        final_curvatures = raw_curvatures
        final_segments = raw_segments
        final_diag = raw_diag
        final_metrics = raw_metrics
        final_source = "raw_resampled_polyline"
    elif cfg.geometry_mode == "raw_preferred" and candidate_source == "smoothed_bezier":
        reasons = _raw_preferred_reasons(comparison_candidate, cfg)
        if reasons:
            raw_preferred_fallback = True
            decision = "rejected_final_fallback_raw"
            final_points = raw_points
            final_headings = raw_headings
            final_curvatures = raw_curvatures
            final_segments = raw_segments
            final_diag = raw_diag
            final_metrics = raw_metrics
            final_source = "raw_resampled_polyline"
    elif cfg.geometry_mode == "raw_preferred":
        decision = "raw_preferred_raw_source"
        final_source = "raw_resampled_polyline"

    comparison_final = _raw_vs_final(raw_metrics, final_metrics, raw_diag, final_diag)

    smoothing_diagnostics = dict(smoothing_diagnostics)
    smoothing_diagnostics["rawPathDiagnostics"] = dict(raw_diag)
    smoothing_diagnostics["smoothedPathDiagnostics"] = dict(final_diag)
    if raw_preferred_fallback:
        smoothing_diagnostics["candidatePathDiagnostics"] = dict(cand_diag)
    smoothing_diagnostics["geometryMode"] = str(cfg.geometry_mode)
    smoothing_diagnostics["geometryDecision"] = str(decision)
    smoothing_diagnostics["geometryDecisionReasons"] = list(reasons)
    smoothing_diagnostics["rawPreferredFallbackUsed"] = bool(raw_preferred_fallback)
    smoothing_diagnostics["finalGeometrySource"] = str(final_source)
    smoothing_diagnostics["rawVsCandidateComparison"] = dict(comparison_candidate)
    smoothing_diagnostics["rawVsFinalComparison"] = dict(comparison_final)

    with timer.stage("rotation_profile_generation"):
        sampled_progress = progress_from_points(final_points)
        sampled_holonomic_rotations = compute_holonomic_rotation_profile(
            path_tangent_headings_rad=final_headings,
            progress_u=sampled_progress,
            cfg=cfg,
        )
        raw_progress = progress_from_points(raw_points)
        raw_holonomic_rotations = compute_holonomic_rotation_profile(
            path_tangent_headings_rad=raw_headings,
            progress_u=raw_progress,
            cfg=cfg,
        )

    with timer.stage("speed_profile_generation"):
        speed_profile = compute_speed_profile(
            points=final_points,
            curvatures=final_curvatures,
            max_speed_mps=cfg.max_speed_mps,
            max_accel_mps2=cfg.max_accel_mps2,
            max_centripetal_accel_mps2=cfg.max_centripetal_accel_mps2,
            start_velocity_mps=cfg.start_velocity_mps,
            end_velocity_mps=cfg.end_velocity_mps,
        )
        raw_speed_profile = compute_speed_profile(
            points=raw_points,
            curvatures=raw_curvatures,
            max_speed_mps=cfg.max_speed_mps,
            max_accel_mps2=cfg.max_accel_mps2,
            max_centripetal_accel_mps2=cfg.max_centripetal_accel_mps2,
            start_velocity_mps=cfg.start_velocity_mps,
            end_velocity_mps=cfg.end_velocity_mps,
        )

    start_align_err = _diag_float(final_diag, "startEndpointAlignmentErrorDeg", 0.0)
    end_align_err = _diag_float(final_diag, "endEndpointAlignmentErrorDeg", 0.0)
    terminal_guard_refit = bool(smoothing_diagnostics.get("terminalDegradationRefitTriggered", False))
    timer.stop_total()
    stage_timings = timer.as_dict()
    runtime_stats = local_runtime.stats.as_dict()
    backend_status = {
        "requested": str(env.backend_status.requested),
        "used": str(env.backend_status.used),
        "reason": str(env.backend_status.reason),
    }

    summary: dict[str, Any] = {
        "plannerMode": cfg.planner_mode,
        "runtimeMode": cfg.runtime_mode,
        "geometryMode": cfg.geometry_mode,
        "geometryDecision": decision,
        "geometryDecisionReasons": list(reasons),
        "finalGeometrySource": final_source,
        "rawPreferredFallbackUsed": bool(raw_preferred_fallback),
        "pathExists": True,
        "holonomicRotationMode": cfg.holonomic_rotation_mode,
        "requiredClearanceM": float(clearance.required_clearance_m),
        "hardClearanceFeasible": bool(clearance.hard_clearance_feasible),
        "rawLengthM": float(raw_metrics.length_m),
        "finalLengthM": float(final_metrics.length_m),
        "smoothedLengthM": float(final_metrics.length_m),
        "rawIntegratedCost": float(raw_metrics.integrated_cost),
        "finalIntegratedCost": float(final_metrics.integrated_cost),
        "smoothedIntegratedCost": float(final_metrics.integrated_cost),
        "rawObjectiveIntegratedCost": float(raw_metrics.objective_cost),
        "finalObjectiveIntegratedCost": float(final_metrics.objective_cost),
        "smoothedObjectiveIntegratedCost": float(final_metrics.objective_cost),
        "rawMaxCurvature": float(raw_metrics.max_curvature),
        "finalMaxCurvature": float(final_metrics.max_curvature),
        "rawCurvatureP95": float(raw_metrics.p95_curvature),
        "finalCurvatureP95": float(final_metrics.p95_curvature),
        "minWallClearanceM": float(final_metrics.min_wall_clearance_m),
        "minHeatRegionClearanceM": float(final_metrics.min_heat_region_clearance_m),
        "startEndpointAlignmentErrorDeg": float(start_align_err),
        "endEndpointAlignmentErrorDeg": float(end_align_err),
        "startCostToGo": float(t_field[start_rc]),
        "startVelocityMpsRequested": float(cfg.start_velocity_mps),
        "startVelocityMpsActual": float(speed_profile[0]["v"]) if speed_profile else 0.0,
        "endVelocityMpsRequested": float(cfg.end_velocity_mps),
        "endVelocityMpsActual": float(speed_profile[-1]["v"]) if speed_profile else 0.0,
        "bezierSegmentCount": len(final_segments),
        "smoothingAcceptedAttempt": int(smoothing_diagnostics.get("acceptedAttempt", -1)),
        "smoothingAttemptCount": int(smoothing_diagnostics.get("attemptCount", 0)),
        "terminalSafeRawFallbackUsed": bool(smoothing_diagnostics.get("terminalSafeRawFallbackUsed", False)),
        "terminalDegradationRefitTriggered": bool(terminal_guard_refit),
        "startTerminalHeatExposureRaw": _diag_float(raw_diag, "startTerminalHeatExposure", 0.0),
        "startTerminalHeatExposureSmoothed": _diag_float(final_diag, "startTerminalHeatExposure", 0.0),
        "goalTerminalHeatExposureRaw": _diag_float(raw_diag, "goalTerminalHeatExposure", 0.0),
        "goalTerminalHeatExposureSmoothed": _diag_float(final_diag, "goalTerminalHeatExposure", 0.0),
        "startTerminalMinWallClearanceRawM": _diag_float(raw_diag, "startTerminalMinWallClearanceM", 0.0),
        "startTerminalMinWallClearanceSmoothedM": _diag_float(final_diag, "startTerminalMinWallClearanceM", 0.0),
        "goalTerminalMinWallClearanceRawM": _diag_float(raw_diag, "goalTerminalMinWallClearanceM", 0.0),
        "goalTerminalMinWallClearanceSmoothedM": _diag_float(final_diag, "goalTerminalMinWallClearanceM", 0.0),
        "startTerminalDirectnessScoreRaw": _diag_float(raw_diag, "startTerminalDirectnessScore", 1.0),
        "startTerminalDirectnessScoreSmoothed": _diag_float(final_diag, "startTerminalDirectnessScore", 1.0),
        "goalTerminalDirectnessScoreRaw": _diag_float(raw_diag, "goalTerminalDirectnessScore", 1.0),
        "goalTerminalDirectnessScoreSmoothed": _diag_float(final_diag, "goalTerminalDirectnessScore", 1.0),
        "startTerminalHookFlagSmoothed": _diag_float(final_diag, "startTerminalHookOrOvershootFlag", 0.0),
        "goalTerminalHookFlagSmoothed": _diag_float(final_diag, "goalTerminalHookOrOvershootFlag", 0.0),
        "rawVsFinalHeatCostRatio": float(comparison_final.get("heatCostRatio", 1.0)),
        "rawVsFinalObjectiveCostRatio": float(comparison_final.get("objectiveCostRatio", 1.0)),
        "rawVsFinalLengthRatio": float(comparison_final.get("lengthRatio", 1.0)),
        "rawVsFinalMaxCurvatureRatio": float(comparison_final.get("maxCurvatureRatio", 1.0)),
        "rawVsFinalCurvatureP95Ratio": float(comparison_final.get("curvatureP95Ratio", 1.0)),
        "rawVsFinalMinWallClearanceDeltaM": float(comparison_final.get("minWallClearanceDeltaM", 0.0)),
        "rawVsFinalMinHeatRegionClearanceDeltaM": float(comparison_final.get("minHeatRegionClearanceDeltaM", 0.0)),
        "rawVsFinalMaxCurvatureDelta": float(comparison_final.get("maxCurvatureDelta", 0.0)),
        "rawVsFinalCurvatureP95Delta": float(comparison_final.get("curvatureP95Delta", 0.0)),
        "rawVsFinalStartTerminalDirectnessDelta": float(comparison_final.get("startTerminalDirectnessDelta", 0.0)),
        "rawVsFinalGoalTerminalDirectnessDelta": float(comparison_final.get("goalTerminalDirectnessDelta", 0.0)),
        "rawVsFinalStartTerminalHeatExposureRatio": float(comparison_final.get("startTerminalHeatExposureRatio", 1.0)),
        "rawVsFinalGoalTerminalHeatExposureRatio": float(comparison_final.get("goalTerminalHeatExposureRatio", 1.0)),
        "rawVsFinalStartTerminalHookIntroduced": bool(comparison_final.get("startTerminalHookIntroduced", False)),
        "rawVsFinalGoalTerminalHookIntroduced": bool(comparison_final.get("goalTerminalHookIntroduced", False)),
        "rawVsFinalTerminalOvershootDelta": float(comparison_final.get("terminalOvershootDelta", 0.0)),
        "timingMs": stage_timings,
        "backend": backend_status,
        "runtimeCache": runtime_stats,
    }

    return PlannerArtifacts(
        t_field=t_field,
        cost_density=cost_density,
        blocked=blocked,
        planning_blocked=clearance.planning_blocked,
        wall_clearance_m=clearance.wall_clearance_m,
        heat_region_clearance_m=clearance.heat_region_clearance_m,
        required_clearance_m=float(clearance.required_clearance_m),
        hard_clearance_feasible=bool(clearance.hard_clearance_feasible),
        raw_path_world=raw_path_world,
        raw_path_world_resampled=raw_points,
        raw_path_tangent_headings_rad=raw_headings,
        raw_path_holonomic_rotations_rad=raw_holonomic_rotations,
        raw_speed_profile=raw_speed_profile,
        raw_path_cells=raw_path_cells,
        final_geometry_source=final_source,
        bezier_segments=final_segments,
        sampled_smoothed_points=final_points,
        sampled_path_tangent_headings_rad=final_headings,
        sampled_holonomic_rotations_rad=sampled_holonomic_rotations,
        sampled_curvatures=final_curvatures,
        speed_profile=speed_profile,
        smoothing_diagnostics=smoothing_diagnostics,
        stage_timings_ms=stage_timings,
        backend_status=backend_status,
        runtime_cache_stats=runtime_stats,
        summary=summary,
    )
