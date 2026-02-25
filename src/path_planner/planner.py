from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clearance import sample_field_along_path
from .config import PlannerConfig
from .extract import extract_path_from_cost_to_go
from .geometry import (
    bilinear_sample_grid_vectorized,
    polyline_length,
    resample_polyline,
    sample_bezier_chain,
)
from .heatmap import world_to_cell_index
from .models import BezierSegment
from .performance import StageTimer
from .rotation import compute_holonomic_rotation_profile, progress_from_points
from .runtime import PlannerRuntime
from .smooth import SmoothingContext, SmoothingResult, smooth_path_to_beziers
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
    raw_path_cells: list[tuple[int, int]]
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


def _raw_polyline_to_beziers(raw_path_world: list[tuple[float, float]], cfg: PlannerConfig) -> list[BezierSegment]:
    raw_points = np.asarray(raw_path_world, dtype=float)
    if len(raw_points) < 2:
        return []
    ds = max(0.25, float(cfg.bezier_target_segment_length_m))
    anchors = resample_polyline(raw_points, ds)
    if len(anchors) < 2:
        anchors = raw_points
    segs: list[BezierSegment] = []
    for i in range(len(anchors) - 1):
        p0 = anchors[i]
        p3 = anchors[i + 1]
        d = p3 - p0
        p1 = p0 + d / 3.0
        p2 = p0 + 2.0 * d / 3.0
        segs.append(BezierSegment(p0=p0.copy(), p1=p1.copy(), p2=p2.copy(), p3=p3.copy()))
    return segs


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

    with timer.stage("smoothing"):
        if cfg.enable_smoothing:
            smoothing_context = SmoothingContext(
                start_xy=np.asarray(start_xy, dtype=float),
                goal_xy=np.asarray(goal_xy, dtype=float),
                wall_clearance_field=clearance.wall_clearance_m,
                heat_region_clearance_field=clearance.heat_region_clearance_m,
                required_clearance_m=clearance.required_clearance_m,
                hard_clearance_feasible=clearance.hard_clearance_feasible,
            )
            smoothing: SmoothingResult = smooth_path_to_beziers(raw_path_world, cfg, context=smoothing_context)
            bezier_segments = smoothing.segments
            smoothing_diagnostics = smoothing.diagnostics
        else:
            bezier_segments = _raw_polyline_to_beziers(raw_path_world, cfg)
            smoothing_diagnostics = {
                "attemptCount": 0,
                "acceptedAttempt": -1,
                "refitTriggered": False,
                "rawPathDiagnostics": {},
                "smoothedPathDiagnostics": {},
                "attempts": [],
                "mode": "disabled",
            }
    sampled_points, sampled_path_tangent_headings, sampled_curvatures = sample_bezier_chain(
        bezier_segments,
        sample_per_segment=max(16, int(1.0 / max(cfg.sample_ds_m, 1e-3))),
    )
    sampled_progress = progress_from_points(sampled_points)
    with timer.stage("rotation_profile_generation"):
        sampled_holonomic_rotations = compute_holonomic_rotation_profile(
            path_tangent_headings_rad=sampled_path_tangent_headings,
            progress_u=sampled_progress,
            cfg=cfg,
        )

    with timer.stage("speed_profile_generation"):
        speed_profile = compute_speed_profile(
            points=sampled_points,
            curvatures=sampled_curvatures,
            max_speed_mps=cfg.max_speed_mps,
            max_accel_mps2=cfg.max_accel_mps2,
            max_centripetal_accel_mps2=cfg.max_centripetal_accel_mps2,
            end_velocity_mps=cfg.end_velocity_mps,
        )

    raw_np = np.asarray(raw_path_world, dtype=float)
    raw_len = polyline_length(raw_np)
    smooth_len = polyline_length(sampled_points)
    raw_cost = _integrate_heat_cost_along_polyline(raw_np, cost_density, cfg.resolution_m_per_cell)
    smooth_cost = _integrate_heat_cost_along_polyline(
        sampled_points, cost_density, cfg.resolution_m_per_cell
    )
    raw_objective_cost = _integrate_heat_cost_along_polyline(
        raw_np,
        planning_density,
        cfg.resolution_m_per_cell,
    )
    smooth_objective_cost = _integrate_heat_cost_along_polyline(
        sampled_points,
        planning_density,
        cfg.resolution_m_per_cell,
    )
    with timer.stage("clearance_validation"):
        sampled_wall_clearance = sample_field_along_path(
            sampled_points,
            clearance.wall_clearance_m,
            cfg.resolution_m_per_cell,
        )
        sampled_heat_region_clearance = sample_field_along_path(
            sampled_points,
            clearance.heat_region_clearance_m,
            cfg.resolution_m_per_cell,
        )
    finite_wall = sampled_wall_clearance[np.isfinite(sampled_wall_clearance)]
    finite_heat_region_clearance = sampled_heat_region_clearance[np.isfinite(sampled_heat_region_clearance)]
    min_wall_clearance = float(np.min(finite_wall)) if finite_wall.size else 0.0
    min_heat_region_clearance = (
        float(np.min(finite_heat_region_clearance))
        if finite_heat_region_clearance.size
        else -1.0
    )
    smoothed_diag = dict(smoothing_diagnostics.get("smoothedPathDiagnostics", {}))
    start_align_err = float(smoothed_diag.get("startEndpointAlignmentErrorDeg", 0.0))
    end_align_err = float(smoothed_diag.get("endEndpointAlignmentErrorDeg", 0.0))
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
        "pathExists": True,
        "holonomicRotationMode": cfg.holonomic_rotation_mode,
        "requiredClearanceM": float(clearance.required_clearance_m),
        "hardClearanceFeasible": bool(clearance.hard_clearance_feasible),
        "rawLengthM": float(raw_len),
        "smoothedLengthM": float(smooth_len),
        "rawIntegratedCost": float(raw_cost),
        "smoothedIntegratedCost": float(smooth_cost),
        "rawObjectiveIntegratedCost": float(raw_objective_cost),
        "smoothedObjectiveIntegratedCost": float(smooth_objective_cost),
        "minWallClearanceM": min_wall_clearance,
        "minHeatRegionClearanceM": min_heat_region_clearance,
        "startEndpointAlignmentErrorDeg": start_align_err,
        "endEndpointAlignmentErrorDeg": end_align_err,
        "startCostToGo": float(t_field[start_rc]),
        "bezierSegmentCount": len(bezier_segments),
        "smoothingAcceptedAttempt": int(smoothing_diagnostics.get("acceptedAttempt", -1)),
        "smoothingAttemptCount": int(smoothing_diagnostics.get("attemptCount", 0)),
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
        raw_path_cells=raw_path_cells,
        bezier_segments=bezier_segments,
        sampled_smoothed_points=sampled_points,
        sampled_path_tangent_headings_rad=sampled_path_tangent_headings,
        sampled_holonomic_rotations_rad=sampled_holonomic_rotations,
        sampled_curvatures=sampled_curvatures,
        speed_profile=speed_profile,
        smoothing_diagnostics=smoothing_diagnostics,
        stage_timings_ms=stage_timings,
        backend_status=backend_status,
        runtime_cache_stats=runtime_stats,
        summary=summary,
    )
