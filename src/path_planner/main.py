from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import PlannerConfig
from .export_pathplanner import (
    beziers_to_pathplanner_waypoints,
    build_best_effort_path_file,
    build_concept_export,
    build_runtime_payload_compact,
    write_json,
    write_waypoints_csv,
)
from .planner import PlannerArtifacts, run_planner
from .runtime import PlannerRuntime
from .scenarios import Scenario, get_scenarios
from .visualize import (
    save_comparison_plot,
    save_cost_to_go_plot,
    save_heatmap_plot,
    save_path_animation_gif,
    save_path_overlay_plot,
    save_scalar_overlay_plot,
)


def _parse_blocked_sentinel(value: str | None) -> float | None:
    if value is None:
        return None
    if value.strip().lower() == "nan":
        return float("nan")
    return float(value)


def _build_cfg(args: argparse.Namespace, scenario: Scenario) -> PlannerConfig:
    cfg = PlannerConfig(
        runtime_mode=args.mode,
        geometry_mode=args.geometry_mode,
        compute_backend=args.compute_backend,
        cache_goal_fields=args.cache_goal_fields,
        max_goal_field_cache_entries=args.max_goal_cache_entries,
        resolution_m_per_cell=scenario.resolution_m_per_cell,
        planner_mode=args.planner,
        cost_mode=args.cost_mode,
        alpha=args.alpha,
        base_cost=args.base_cost,
        blocked_sentinel=_parse_blocked_sentinel(args.blocked_sentinel),
        start_heading_deg=args.start_heading if args.start_heading is not None else scenario.start_heading_deg,
        end_heading_deg=args.end_heading if args.end_heading is not None else scenario.end_heading_deg,
        start_approach_heading_deg=(
            args.start_approach_heading
            if args.start_approach_heading is not None
            else scenario.start_approach_heading_deg
        ),
        goal_approach_heading_deg=(
            args.goal_approach_heading
            if args.goal_approach_heading is not None
            else scenario.goal_approach_heading_deg
        ),
        start_approach_lock_distance_m=(
            args.start_approach_lock_distance_m
            if args.start_approach_lock_distance_m is not None
            else scenario.start_approach_lock_distance_m
        ),
        goal_approach_lock_distance_m=(
            args.goal_approach_lock_distance_m
            if args.goal_approach_lock_distance_m is not None
            else scenario.goal_approach_lock_distance_m
        ),
        start_velocity_mps=args.start_velocity,
        end_velocity_mps=args.end_velocity,
        max_curvature=args.max_curvature,
        max_endpoint_curvature=args.max_endpoint_curvature,
        endpoint_zone_m=args.endpoint_zone_m,
        endpoint_spacing_exponent=args.endpoint_spacing_exponent,
        sample_ds_m=args.sample_ds,
        rdp_tolerance_m=args.rdp_tolerance,
        path_step_m=args.path_step,
        spline_smoothing=args.spline_smoothing,
        max_smoothing_refits=args.max_refits,
        bezier_target_segment_length_m=args.bezier_seg_len,
        max_tangent_jump_deg=args.max_tangent_jump_deg,
        max_endpoint_tangent_jump_deg=args.max_endpoint_tangent_jump_deg,
        handle_clamp_ratio=args.handle_clamp_ratio,
        enable_smoothing=args.enable_smoothing,
        enable_clearance_constraints=args.enable_clearance_constraints,
        holonomic_rotation_mode=args.holonomic_rotation_mode,
        rotation_finish_progress=args.rotation_finish_progress,
        endpoint_alignment_tolerance_deg=args.endpoint_alignment_tolerance_deg,
        endpoint_overshoot_tolerance_m=args.endpoint_overshoot_tolerance_m,
        terminal_progress_window_m=args.terminal_progress_window_m,
        allow_terminal_overshoot=args.allow_terminal_overshoot,
        object_width_m=args.object_width_m,
        object_height_m=args.object_height_m,
        object_shape=args.object_shape,
        safe_space_m=args.safe_space_m,
        wall_clearance_weight=args.wall_clearance_weight,
        wall_clearance_power=args.wall_clearance_power,
        wall_clearance_soft_ratio=args.wall_clearance_soft_ratio,
        enforce_hard_clearance_if_feasible=args.enforce_hard_clearance_if_feasible,
        heat_region_clearance_enabled=args.heat_region_clearance_enabled,
        heat_region_threshold=args.heat_region_threshold,
        heat_region_quantile=args.heat_region_quantile,
        heat_region_clearance_weight=args.heat_region_clearance_weight,
        heat_region_clearance_decay_m=args.heat_region_clearance_decay_m,
        min_terminal_progress_ratio=args.min_terminal_progress_ratio,
        clearance_refit_threshold_m=args.clearance_refit_threshold_m,
    )
    return cfg


def _serialize_summary_payload(
    artifacts: PlannerArtifacts,
    waypoints: list[dict[str, Any]],
    cfg: PlannerConfig,
) -> dict[str, Any]:
    sampled_heading_profile: list[dict[str, float]] = []
    raw_heading_profile: list[dict[str, float]] = []
    n = len(artifacts.sampled_smoothed_points)
    speed_s = [float(p.get("s", 0.0)) for p in artifacts.speed_profile]
    raw_speed_s = [float(p.get("s", 0.0)) for p in artifacts.raw_speed_profile]
    for i in range(n):
        tangent_rad = (
            float(artifacts.sampled_path_tangent_headings_rad[i])
            if i < len(artifacts.sampled_path_tangent_headings_rad)
            else 0.0
        )
        holo_rad = (
            float(artifacts.sampled_holonomic_rotations_rad[i])
            if i < len(artifacts.sampled_holonomic_rotations_rad)
            else tangent_rad
        )
        sampled_heading_profile.append(
            {
                "idx": i,
                "s": speed_s[i] if i < len(speed_s) else 0.0,
                "x": float(artifacts.sampled_smoothed_points[i, 0]),
                "y": float(artifacts.sampled_smoothed_points[i, 1]),
                "pathTangentHeadingDeg": float(math.degrees(tangent_rad)),
                "pathTangentHeadingRad": tangent_rad,
                "holonomicRotationDeg": float(math.degrees(holo_rad)),
                "holonomicRotationRad": holo_rad,
            }
        )
    raw_n = len(artifacts.raw_path_world_resampled)
    for i in range(raw_n):
        tangent_rad = (
            float(artifacts.raw_path_tangent_headings_rad[i])
            if i < len(artifacts.raw_path_tangent_headings_rad)
            else 0.0
        )
        holo_rad = (
            float(artifacts.raw_path_holonomic_rotations_rad[i])
            if i < len(artifacts.raw_path_holonomic_rotations_rad)
            else tangent_rad
        )
        raw_heading_profile.append(
            {
                "idx": i,
                "s": raw_speed_s[i] if i < len(raw_speed_s) else 0.0,
                "x": float(artifacts.raw_path_world_resampled[i, 0]),
                "y": float(artifacts.raw_path_world_resampled[i, 1]),
                "pathTangentHeadingDeg": float(math.degrees(tangent_rad)),
                "pathTangentHeadingRad": tangent_rad,
                "holonomicRotationDeg": float(math.degrees(holo_rad)),
                "holonomicRotationRad": holo_rad,
            }
        )

    return {
        "summary": artifacts.summary,
        "heatVsObstacle": {
            "heat": "Finite heat values are traversable with increased cost.",
            "obstacle": "Blocked/infinite cells are explicitly untraversable.",
        },
        "clearanceModel": {
            "objectShape": cfg.object_shape,
            "objectWidthM": cfg.object_width_m,
            "objectHeightM": cfg.object_height_m,
            "footprintRadiusM": cfg.footprint_radius_m,
            "safeSpaceM": cfg.safe_space_m,
            "requiredClearanceM": cfg.required_clearance_m,
            "hardClearanceFeasible": artifacts.hard_clearance_feasible,
            "heatRegionClearanceEnabled": cfg.heat_region_clearance_enabled,
        },
        "rawPathPoints": [{"x": x, "y": y} for x, y in artifacts.raw_path_world],
        "rawPathWorldResampled": [
            {"x": float(p[0]), "y": float(p[1])} for p in np.asarray(artifacts.raw_path_world_resampled, dtype=float)
        ],
        "smoothedBezierSegments": [seg.as_dict() for seg in artifacts.bezier_segments],
        "sampledHeadingProfile": sampled_heading_profile,
        "rawSampledHeadingProfile": raw_heading_profile,
        "pathPlannerWaypoints": waypoints,
        "smoothingDiagnostics": artifacts.smoothing_diagnostics,
        "goalEndState": {
            "velocityMps": cfg.end_velocity_mps,
            "rotationDeg": cfg.end_heading_deg,
            "rotationRad": None
            if cfg.end_heading_deg is None
            else float(math.radians(cfg.end_heading_deg)),
        },
        "startState": {
            "velocityMps": cfg.start_velocity_mps,
            "rotationDeg": cfg.start_heading_deg,
            "rotationRad": None
            if cfg.start_heading_deg is None
            else float(math.radians(cfg.start_heading_deg)),
        },
        "tuningKnobs": {
            "runtime_mode": cfg.runtime_mode,
            "geometry_mode": cfg.geometry_mode,
            "compute_backend": cfg.compute_backend,
            "cache_goal_fields": cfg.cache_goal_fields,
            "max_goal_field_cache_entries": cfg.max_goal_field_cache_entries,
            "alpha": cfg.alpha,
            "base_cost": cfg.base_cost,
            "epsilon": cfg.epsilon,
            "cost_mode": cfg.cost_mode,
            "path_step_m": cfg.path_step_m,
            "rdp_tolerance_m": cfg.rdp_tolerance_m,
            "sample_ds_m": cfg.sample_ds_m,
            "handle_scale": cfg.handle_scale,
            "max_curvature": cfg.max_curvature,
            "spline_smoothing": cfg.spline_smoothing,
            "max_smoothing_refits": cfg.max_smoothing_refits,
            "runtime_fast_max_refits": cfg.runtime_fast_max_refits,
            "bezier_target_segment_length_m": cfg.bezier_target_segment_length_m,
            "raw_reference_sample_ds_m": cfg.raw_reference_sample_ds_m,
            "raw_linear_bezier_ds_m": cfg.raw_linear_bezier_ds_m,
            "max_tangent_jump_deg": cfg.max_tangent_jump_deg,
            "max_tangent_mag_jump_ratio": cfg.max_tangent_mag_jump_ratio,
            "curvature_spike_factor": cfg.curvature_spike_factor,
            "handle_clamp_ratio": cfg.handle_clamp_ratio,
            "min_handle_ratio": cfg.min_handle_ratio,
            "sharp_turn_deg": cfg.sharp_turn_deg,
            "sharp_turn_handle_scale": cfg.sharp_turn_handle_scale,
            "c2_regularization_weight": cfg.c2_regularization_weight,
            "c2_regularization_iters": cfg.c2_regularization_iters,
            "max_endpoint_curvature": cfg.max_endpoint_curvature,
            "endpoint_zone_m": cfg.endpoint_zone_m,
            "endpoint_zone_growth": cfg.endpoint_zone_growth,
            "endpoint_heading_blend_power": cfg.endpoint_heading_blend_power,
            "endpoint_spacing_exponent": cfg.endpoint_spacing_exponent,
            "endpoint_handle_scale": cfg.endpoint_handle_scale,
            "endpoint_handle_decay": cfg.endpoint_handle_decay,
            "min_endpoint_handle_scale": cfg.min_endpoint_handle_scale,
            "max_endpoint_tangent_jump_deg": cfg.max_endpoint_tangent_jump_deg,
            "start_approach_heading_deg": cfg.resolved_start_approach_heading_deg,
            "goal_approach_heading_deg": cfg.resolved_goal_approach_heading_deg,
            "start_approach_lock_distance_m": cfg.start_approach_lock_distance_m,
            "goal_approach_lock_distance_m": cfg.goal_approach_lock_distance_m,
            "endpoint_alignment_tolerance_deg": cfg.endpoint_alignment_tolerance_deg,
            "endpoint_overshoot_tolerance_m": cfg.endpoint_overshoot_tolerance_m,
            "terminal_progress_window_m": cfg.terminal_progress_window_m,
            "allow_terminal_overshoot": cfg.allow_terminal_overshoot,
            "start_heading_deg": cfg.start_heading_deg,
            "end_heading_deg": cfg.end_heading_deg,
            "holonomic_rotation_mode": cfg.holonomic_rotation_mode,
            "rotation_finish_progress": cfg.rotation_finish_progress,
            "start_velocity_mps": cfg.start_velocity_mps,
            "object_width_m": cfg.object_width_m,
            "object_height_m": cfg.object_height_m,
            "object_shape": cfg.object_shape,
            "safe_space_m": cfg.safe_space_m,
            "footprint_radius_m": cfg.footprint_radius_m,
            "required_clearance_m": cfg.required_clearance_m,
            "wall_clearance_weight": cfg.wall_clearance_weight,
            "wall_clearance_power": cfg.wall_clearance_power,
            "wall_clearance_soft_ratio": cfg.wall_clearance_soft_ratio,
            "heat_region_clearance_enabled": cfg.heat_region_clearance_enabled,
            "heat_region_threshold": cfg.heat_region_threshold,
            "heat_region_quantile": cfg.heat_region_quantile,
            "heat_region_clearance_weight": cfg.heat_region_clearance_weight,
            "heat_region_clearance_decay_m": cfg.heat_region_clearance_decay_m,
            "enforce_hard_clearance_if_feasible": cfg.enforce_hard_clearance_if_feasible,
            "raw_preferred_max_heat_cost_ratio": cfg.raw_preferred_max_heat_cost_ratio,
            "raw_preferred_max_objective_cost_ratio": cfg.raw_preferred_max_objective_cost_ratio,
            "raw_preferred_max_length_ratio": cfg.raw_preferred_max_length_ratio,
            "raw_preferred_max_curvature_ratio": cfg.raw_preferred_max_curvature_ratio,
            "raw_preferred_max_curvature_abs_increase": cfg.raw_preferred_max_curvature_abs_increase,
            "raw_preferred_max_wall_clearance_drop_m": cfg.raw_preferred_max_wall_clearance_drop_m,
            "raw_preferred_max_heat_region_clearance_drop_m": cfg.raw_preferred_max_heat_region_clearance_drop_m,
            "raw_preferred_max_terminal_directness_drop": cfg.raw_preferred_max_terminal_directness_drop,
            "raw_preferred_max_terminal_heat_exposure_ratio": cfg.raw_preferred_max_terminal_heat_exposure_ratio,
            "raw_preferred_length_cost_improve_ratio": cfg.raw_preferred_length_cost_improve_ratio,
            "clearance_refit_threshold_m": cfg.clearance_refit_threshold_m,
            "enable_smoothing": cfg.enable_smoothing,
            "enable_clearance_constraints": cfg.enable_clearance_constraints,
            "end_velocity_mps": cfg.end_velocity_mps,
            "max_speed_mps": cfg.max_speed_mps,
            "max_accel_mps2": cfg.max_accel_mps2,
            "max_centripetal_accel_mps2": cfg.max_centripetal_accel_mps2,
            "blocked_sentinel": cfg.blocked_sentinel,
            "planner_mode": cfg.planner_mode,
        },
    }


def _save_visuals(
    artifacts: PlannerArtifacts,
    scenario: Scenario,
    out_dir: Path,
    animate: bool,
) -> dict[str, str]:
    files: dict[str, str] = {}

    heat_p = out_dir / "heatmap.png"
    save_heatmap_plot(
        out_path=heat_p,
        heat=scenario.heat,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        resolution_m=scenario.resolution_m_per_cell,
        blocked=artifacts.blocked,
    )
    files["heatmap"] = str(heat_p)

    t_p = out_dir / "cost_to_go.png"
    save_cost_to_go_plot(
        out_path=t_p,
        t_field=artifacts.t_field,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        resolution_m=scenario.resolution_m_per_cell,
    )
    files["cost_to_go"] = str(t_p)

    overlay_p = out_dir / "path_overlay.png"
    save_path_overlay_plot(
        out_path=overlay_p,
        heat=scenario.heat,
        resolution_m=scenario.resolution_m_per_cell,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        raw_path_world=artifacts.raw_path_world,
        bezier_segments=artifacts.bezier_segments,
        blocked=artifacts.blocked,
    )
    files["path_overlay"] = str(overlay_p)

    curv_p = out_dir / "curvature_overlay.png"
    save_scalar_overlay_plot(
        out_path=curv_p,
        heat=scenario.heat,
        resolution_m=scenario.resolution_m_per_cell,
        points=artifacts.sampled_smoothed_points,
        scalar=artifacts.sampled_curvatures,
        label="Curvature (1/m)",
        title="Curvature Overlay Along Smoothed Path",
    )
    files["curvature_overlay"] = str(curv_p)

    speed_vals = np.array([p["v"] for p in artifacts.speed_profile], dtype=float)
    speed_p = out_dir / "speed_overlay.png"
    save_scalar_overlay_plot(
        out_path=speed_p,
        heat=scenario.heat,
        resolution_m=scenario.resolution_m_per_cell,
        points=artifacts.sampled_smoothed_points,
        scalar=speed_vals,
        label="Speed (m/s)",
        title="Speed Profile Overlay Along Smoothed Path",
    )
    files["speed_overlay"] = str(speed_p)

    if animate:
        gif_p = out_dir / "path_animation.gif"
        gif = save_path_animation_gif(
            out_path=gif_p,
            heat=scenario.heat,
            resolution_m=scenario.resolution_m_per_cell,
            sampled_points=artifacts.sampled_smoothed_points,
            robot_rotations_rad=artifacts.sampled_holonomic_rotations_rad,
            start_xy=scenario.start_xy,
            goal_xy=scenario.goal_xy,
        )
        if gif is not None:
            files["animation_gif"] = gif

    return files


def _print_summary_line(scenario: Scenario, artifacts: PlannerArtifacts) -> None:
    s = artifacts.summary
    timing = s.get("timingMs", {})
    total_ms = float(timing.get("totalRuntimeMs", 0.0))
    print(
        f"[{scenario.name}] mode={s['plannerMode']} rawCost={s['rawIntegratedCost']:.2f} "
        f"smoothCost={s['smoothedIntegratedCost']:.2f} rawLen={s['rawLengthM']:.2f}m "
        f"smoothLen={s['smoothedLengthM']:.2f}m segments={int(s['bezierSegmentCount'])} "
        f"geom={s.get('finalGeometrySource', '-')} decision={s.get('geometryDecision', '-')} "
        f"minWallClr={s['minWallClearanceM']:.3f}m reqClr={s['requiredClearanceM']:.3f}m "
        f"lat={total_ms:.2f}ms"
    )


def _run_single(
    scenario: Scenario,
    args: argparse.Namespace,
    all_scenarios: dict[str, Scenario],
    runtime: PlannerRuntime | None = None,
) -> dict[str, Any]:
    cfg = _build_cfg(args, scenario)
    out_dir = Path(args.output_dir) / scenario.name if args.write_artifacts else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = run_planner(
        heat=scenario.heat,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        blocked_mask=scenario.blocked_mask,
        cfg=cfg,
        runtime=runtime,
        environment_id=scenario.name,
    )
    _print_summary_line(scenario, artifacts)

    export_start = time.perf_counter()
    files: dict[str, str] = {}
    runtime_payload = build_runtime_payload_compact(
        bezier_segments=artifacts.bezier_segments,
        sampled_points=artifacts.sampled_smoothed_points,
        sampled_path_tangent_headings_rad=artifacts.sampled_path_tangent_headings_rad,
        sampled_holonomic_rotations_rad=artifacts.sampled_holonomic_rotations_rad,
        raw_path_world_resampled=artifacts.raw_path_world_resampled,
        raw_path_tangent_headings_rad=artifacts.raw_path_tangent_headings_rad,
        raw_path_holonomic_rotations_rad=artifacts.raw_path_holonomic_rotations_rad,
        raw_speed_profile=artifacts.raw_speed_profile,
        final_geometry_source=artifacts.final_geometry_source,
        summary=artifacts.summary,
        required_clearance_m=artifacts.required_clearance_m,
        backend_status=artifacts.backend_status,
        cfg=cfg,
    )
    waypoints: list[dict[str, Any]] = []
    summary_path: Path | None = None

    if args.mode == "debug_diagnostics" or args.write_artifacts:
        waypoints = beziers_to_pathplanner_waypoints(artifacts.bezier_segments, cfg)

    if args.write_artifacts and out_dir is not None:
        if args.mode == "runtime_fast":
            compact_path = out_dir / "runtime_payload.json"
            files["runtime_payload_json"] = str(compact_path)
        else:
            concept = build_concept_export(artifacts.bezier_segments, waypoints, cfg)
            concept_path = out_dir / "pathplanner_concept.json"
            write_json(concept_path, concept)
            files["pathplanner_concept_json"] = str(concept_path)

            if args.export in ("both", "exact"):
                best_effort = build_best_effort_path_file(waypoints, cfg)
                best_path = out_dir / "pathplanner_best_effort.path.json"
                write_json(best_path, best_effort)
                files["best_effort_path_json"] = str(best_path)

            csv_path = out_dir / "pathplanner_waypoints.csv"
            write_waypoints_csv(csv_path, waypoints)
            files["waypoints_csv"] = str(csv_path)

            summary_payload = _serialize_summary_payload(artifacts, waypoints, cfg)
            summary_path = out_dir / "plan_summary.json"
            write_json(summary_path, summary_payload)
            files["plan_summary_json"] = str(summary_path)

            if args.enable_plots:
                visuals = _save_visuals(
                    artifacts=artifacts,
                    scenario=scenario,
                    out_dir=out_dir,
                    animate=args.animation,
                )
                files.update(visuals)

            if args.compare and scenario.name == "hot_island":
                shortest_cfg = cfg.with_updates(alpha=0.0, planner_mode="dijkstra_approx")
                shortest = run_planner(
                    heat=scenario.heat,
                    start_xy=scenario.start_xy,
                    goal_xy=scenario.goal_xy,
                    blocked_mask=scenario.blocked_mask,
                    cfg=shortest_cfg,
                    runtime=runtime,
                    environment_id=f"{scenario.name}_shortest",
                )
                compare_path = out_dir / "comparison.png"
                save_comparison_plot(
                    out_path=compare_path,
                    heat=scenario.heat,
                    resolution_m=scenario.resolution_m_per_cell,
                    shortest_path=shortest.raw_path_world,
                    low_heat_path=artifacts.raw_path_world,
                    smoothed_points=artifacts.sampled_smoothed_points,
                    start_xy=scenario.start_xy,
                    goal_xy=scenario.goal_xy,
                )
                files["comparison_plot"] = str(compare_path)

    export_ms = (time.perf_counter() - export_start) * 1000.0
    artifacts.stage_timings_ms["exportSerializationMs"] = float(export_ms)
    artifacts.stage_timings_ms["totalRuntimeWithExportMs"] = float(
        artifacts.stage_timings_ms.get("totalRuntimeMs", 0.0) + export_ms
    )
    artifacts.summary["timingMs"] = artifacts.stage_timings_ms
    runtime_payload = build_runtime_payload_compact(
        bezier_segments=artifacts.bezier_segments,
        sampled_points=artifacts.sampled_smoothed_points,
        sampled_path_tangent_headings_rad=artifacts.sampled_path_tangent_headings_rad,
        sampled_holonomic_rotations_rad=artifacts.sampled_holonomic_rotations_rad,
        raw_path_world_resampled=artifacts.raw_path_world_resampled,
        raw_path_tangent_headings_rad=artifacts.raw_path_tangent_headings_rad,
        raw_path_holonomic_rotations_rad=artifacts.raw_path_holonomic_rotations_rad,
        raw_speed_profile=artifacts.raw_speed_profile,
        final_geometry_source=artifacts.final_geometry_source,
        summary=artifacts.summary,
        required_clearance_m=artifacts.required_clearance_m,
        backend_status=artifacts.backend_status,
        cfg=cfg,
    )
    if summary_path is not None:
        write_json(summary_path, _serialize_summary_payload(artifacts, waypoints, cfg))
    if args.write_artifacts and out_dir is not None and args.mode == "runtime_fast":
        compact_path = out_dir / "runtime_payload.json"
        write_json(compact_path, runtime_payload, compact=True)

    result = {
        "scenario": scenario.name,
        "description": scenario.description,
        "summary": artifacts.summary,
        "files": files,
        "runtimePayload": runtime_payload if args.mode == "runtime_fast" else None,
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Heatmap-based FMM path planner with Bezier smoothing and PathPlanner-style export."
    )
    p.add_argument("--interactive-visualizer", action="store_true")
    p.add_argument("--mode", default="debug_diagnostics", choices=["runtime_fast", "debug_diagnostics"])
    p.add_argument(
        "--geometry-mode",
        default="raw_preferred",
        choices=["raw_only", "raw_preferred", "spline_then_bezier", "bezier_optimized"],
    )
    p.add_argument("--compute-backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--cache-goal-fields", dest="cache_goal_fields", action="store_true")
    p.add_argument("--no-cache-goal-fields", dest="cache_goal_fields", action="store_false")
    p.set_defaults(cache_goal_fields=True)
    p.add_argument("--max-goal-cache-entries", type=int, default=12)
    p.add_argument(
        "--scenario",
        default="all",
        choices=[
            "all",
            "hot_island",
            "uniform_high",
            "blocked_gap",
            "double_hot_mid_corridor",
            "small_islands_weave",
        ],
    )
    p.add_argument("--planner", default="fmm", choices=["fmm", "dijkstra_approx"])
    p.add_argument("--cost-mode", default="density", choices=["density", "inverse_speed"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--base-cost", type=float, default=1.0)
    p.add_argument("--blocked-sentinel", default=None)
    p.add_argument("--start-heading", type=float, default=None)
    p.add_argument("--end-heading", type=float, default=None)
    p.add_argument("--start-approach-heading", type=float, default=None)
    p.add_argument("--goal-approach-heading", type=float, default=None)
    p.add_argument("--start-approach-lock-distance-m", type=float, default=None)
    p.add_argument("--goal-approach-lock-distance-m", type=float, default=None)
    p.add_argument("--start-velocity", type=float, default=0.0)
    p.add_argument("--end-velocity", type=float, default=0.0)
    p.add_argument("--path-step", type=float, default=0.08)
    p.add_argument("--sample-ds", type=float, default=0.08)
    p.add_argument("--rdp-tolerance", type=float, default=0.20)
    p.add_argument("--max-curvature", type=float, default=2.6)
    p.add_argument("--max-endpoint-curvature", type=float, default=5.0)
    p.add_argument("--endpoint-zone-m", type=float, default=0.5)
    p.add_argument("--endpoint-spacing-exponent", type=float, default=0.55)
    p.add_argument("--endpoint-alignment-tolerance-deg", type=float, default=8.0)
    p.add_argument("--endpoint-overshoot-tolerance-m", type=float, default=0.06)
    p.add_argument("--terminal-progress-window-m", type=float, default=0.75)
    p.add_argument("--min-terminal-progress-ratio", type=float, default=0.92)
    p.add_argument("--allow-terminal-overshoot", dest="allow_terminal_overshoot", action="store_true")
    p.add_argument("--no-terminal-overshoot", dest="allow_terminal_overshoot", action="store_false")
    p.set_defaults(allow_terminal_overshoot=False)
    p.add_argument("--spline-smoothing", type=float, default=1.35)
    p.add_argument("--max-refits", type=int, default=6)
    p.add_argument("--bezier-seg-len", type=float, default=1.8)
    p.add_argument("--max-tangent-jump-deg", type=float, default=1.0)
    p.add_argument("--max-endpoint-tangent-jump-deg", type=float, default=2.0)
    p.add_argument("--handle-clamp-ratio", type=float, default=0.45)
    p.add_argument("--enable-smoothing", dest="enable_smoothing", action="store_true")
    p.add_argument("--no-smoothing", dest="enable_smoothing", action="store_false")
    p.set_defaults(enable_smoothing=True)
    p.add_argument(
        "--enable-clearance-constraints",
        dest="enable_clearance_constraints",
        action="store_true",
    )
    p.add_argument(
        "--no-clearance-constraints",
        dest="enable_clearance_constraints",
        action="store_false",
    )
    p.set_defaults(enable_clearance_constraints=True)
    p.add_argument("--clearance-refit-threshold-m", type=float, default=0.0)
    p.add_argument("--object-width-m", type=float, default=0.85)
    p.add_argument("--object-height-m", type=float, default=0.85)
    p.add_argument("--object-shape", choices=["rectangle", "circle"], default="rectangle")
    p.add_argument("--safe-space-m", type=float, default=0.12)
    p.add_argument("--wall-clearance-weight", type=float, default=1.1)
    p.add_argument("--wall-clearance-power", type=float, default=2.0)
    p.add_argument("--wall-clearance-soft-ratio", type=float, default=1.6)
    p.add_argument(
        "--enforce-hard-clearance-if-feasible",
        dest="enforce_hard_clearance_if_feasible",
        action="store_true",
    )
    p.add_argument(
        "--no-enforce-hard-clearance-if-feasible",
        dest="enforce_hard_clearance_if_feasible",
        action="store_false",
    )
    p.set_defaults(enforce_hard_clearance_if_feasible=True)
    p.add_argument(
        "--heat-region-clearance-enabled",
        dest="heat_region_clearance_enabled",
        action="store_true",
    )
    p.add_argument(
        "--no-heat-region-clearance",
        dest="heat_region_clearance_enabled",
        action="store_false",
    )
    p.set_defaults(heat_region_clearance_enabled=True)
    p.add_argument("--heat-region-threshold", type=float, default=None)
    p.add_argument("--heat-region-quantile", type=float, default=0.86)
    p.add_argument("--heat-region-clearance-weight", type=float, default=0.42)
    p.add_argument("--heat-region-clearance-decay-m", type=float, default=0.55)
    p.add_argument(
        "--holonomic-rotation-mode",
        default="independent_profile",
        choices=["independent_profile", "tangent_follow"],
    )
    p.add_argument("--rotation-finish-progress", type=float, default=0.8)
    p.add_argument("--export", default="both", choices=["concept", "both", "exact"])
    p.add_argument("--write-artifacts", dest="write_artifacts", action="store_true")
    p.add_argument("--no-write-artifacts", dest="write_artifacts", action="store_false")
    p.set_defaults(write_artifacts=None)
    p.add_argument("--enable-plots", dest="enable_plots", action="store_true")
    p.add_argument("--no-plots", dest="enable_plots", action="store_false")
    p.set_defaults(enable_plots=None)
    p.add_argument("--animation", dest="animation", action="store_true")
    p.add_argument("--no-animation", dest="animation", action="store_false")
    p.add_argument("--compare", dest="compare", action="store_true")
    p.add_argument("--no-compare", dest="compare", action="store_false")
    p.set_defaults(compare=None, animation=None)
    return p


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if bool(args.interactive_visualizer):
        from .interactive_visualizer import launch_interactive_visualizer

        initial = args.scenario if args.scenario != "all" else "hot_island"
        return launch_interactive_visualizer(initial_scenario=initial)

    if args.mode == "runtime_fast":
        if args.write_artifacts is None:
            args.write_artifacts = False
        if args.enable_plots is None:
            args.enable_plots = False
        if args.animation is None:
            args.animation = False
        if args.compare is None:
            args.compare = False
    else:
        if args.write_artifacts is None:
            args.write_artifacts = True
        if args.enable_plots is None:
            args.enable_plots = True
        if args.animation is None:
            args.animation = False
        if args.compare is None:
            args.compare = True

    scenarios = get_scenarios()
    runtime = PlannerRuntime(max_goal_cache_entries=max(1, int(args.max_goal_cache_entries)))

    run_list: list[Scenario]
    if args.scenario == "all":
        run_list = [
            scenarios["hot_island"],
            scenarios["uniform_high"],
            scenarios["double_hot_mid_corridor"],
            scenarios["small_islands_weave"],
        ]
    else:
        run_list = [scenarios[args.scenario]]

    results: list[dict[str, Any]] = []
    for scenario in run_list:
        result = _run_single(scenario, args, scenarios, runtime=runtime)
        results.append(result)

    if args.write_artifacts:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        out_root_msg = str(out_root.resolve())
    else:
        out_root_msg = "(disabled in runtime_fast/no-write-artifacts mode)"

    print()
    print("Heat vs obstacle semantics:")
    print("  heat: finite -> traversable with weighted penalty")
    print("  obstacle: blocked/infinite -> untraversable")
    print(f"Artifacts written under: {out_root_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
