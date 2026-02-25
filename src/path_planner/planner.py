from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .clearance import build_clearance_layers, sample_field_along_path
from .config import PlannerConfig
from .dijkstra_approx import compute_cost_to_go_dijkstra
from .extract import extract_path_from_cost_to_go
from .fmm import compute_cost_to_go_fmm
from .geometry import (
    bilinear_sample_grid,
    polyline_length,
    sample_bezier_chain,
    world_to_grid,
)
from .heatmap import build_cost_density, world_to_cell_index
from .models import BezierSegment
from .rotation import compute_holonomic_rotation_profile, progress_from_points
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
    summary: dict[str, Any]


def _integrate_heat_cost_along_polyline(
    points_world: np.ndarray,
    cost_density: np.ndarray,
    resolution_m: float,
) -> float:
    if len(points_world) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points_world) - 1):
        p0 = points_world[i]
        p1 = points_world[i + 1]
        ds = math.dist((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
        if ds <= 1e-12:
            continue
        x0, y0 = world_to_grid(float(p0[0]), float(p0[1]), resolution_m)
        x1, y1 = world_to_grid(float(p1[0]), float(p1[1]), resolution_m)
        w0 = bilinear_sample_grid(cost_density, x0, y0)
        w1 = bilinear_sample_grid(cost_density, x1, y1)
        if not np.isfinite(w0) or not np.isfinite(w1):
            continue
        total += 0.5 * (w0 + w1) * ds
    return float(total)


def _compute_cost_to_go(
    cost_density: np.ndarray,
    blocked: np.ndarray,
    goal_rc: tuple[int, int],
    cfg: PlannerConfig,
) -> np.ndarray:
    if cfg.planner_mode == "fmm":
        return compute_cost_to_go_fmm(
            cost_density=cost_density,
            goal_rc=goal_rc,
            blocked=blocked,
            resolution_m=cfg.resolution_m_per_cell,
        )
    if cfg.planner_mode == "dijkstra_approx":
        return compute_cost_to_go_dijkstra(
            cost_density=cost_density,
            goal_rc=goal_rc,
            blocked=blocked,
            resolution_m=cfg.resolution_m_per_cell,
        )
    raise ValueError(f"Unsupported planner mode: {cfg.planner_mode}")


def run_planner(
    heat: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    cfg: PlannerConfig,
    blocked_mask: np.ndarray | None = None,
) -> PlannerArtifacts:
    np.random.seed(cfg.deterministic_seed)

    cost_density, blocked = build_cost_density(heat=heat, cfg=cfg, blocked_mask=blocked_mask)
    start_rc = world_to_cell_index(start_xy[0], start_xy[1], cfg.resolution_m_per_cell, heat.shape)
    goal_rc = world_to_cell_index(goal_xy[0], goal_xy[1], cfg.resolution_m_per_cell, heat.shape)

    if blocked[start_rc]:
        raise ValueError("Start cell is blocked.")
    if blocked[goal_rc]:
        raise ValueError("Goal cell is blocked.")

    clearance = build_clearance_layers(
        heat=heat,
        blocked=blocked,
        cfg=cfg,
        start_rc=start_rc,
        goal_rc=goal_rc,
    )
    planning_density = np.asarray(cost_density, dtype=float).copy()
    traversable = ~clearance.planning_blocked
    planning_density[traversable] = (
        planning_density[traversable] + clearance.combined_penalty[traversable]
    )
    planning_density[clearance.planning_blocked] = np.inf

    t_field = _compute_cost_to_go(planning_density, clearance.planning_blocked, goal_rc, cfg)

    if not np.isfinite(t_field[start_rc]):
        raise RuntimeError(
            "No finite path from START to GOAL. This indicates blocked-cell disconnection."
        )

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
    smoothing: SmoothingResult = smooth_path_to_beziers(raw_path_world, cfg, context=smoothing_context)
    bezier_segments = smoothing.segments
    sampled_points, sampled_path_tangent_headings, sampled_curvatures = sample_bezier_chain(
        bezier_segments, sample_per_segment=max(20, int(1.0 / max(cfg.sample_ds_m, 1e-3)))
    )
    sampled_progress = progress_from_points(sampled_points)
    sampled_holonomic_rotations = compute_holonomic_rotation_profile(
        path_tangent_headings_rad=sampled_path_tangent_headings,
        progress_u=sampled_progress,
        cfg=cfg,
    )

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
    smoothed_diag = dict(smoothing.diagnostics.get("smoothedPathDiagnostics", {}))
    start_align_err = float(smoothed_diag.get("startEndpointAlignmentErrorDeg", 0.0))
    end_align_err = float(smoothed_diag.get("endEndpointAlignmentErrorDeg", 0.0))

    summary: dict[str, Any] = {
        "plannerMode": cfg.planner_mode,
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
        "smoothingAcceptedAttempt": int(smoothing.diagnostics.get("acceptedAttempt", -1)),
        "smoothingAttemptCount": int(smoothing.diagnostics.get("attemptCount", 0)),
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
        smoothing_diagnostics=smoothing.diagnostics,
        summary=summary,
    )
