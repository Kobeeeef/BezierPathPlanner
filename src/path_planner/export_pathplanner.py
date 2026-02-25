from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import PlannerConfig
from .geometry import angle_rad, normalize
from .models import BezierSegment
from .rotation import compute_holonomic_rotation_profile, progress_from_points


def _point_dict(p: np.ndarray) -> dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1])}


def _anchor_heading(segments: list[BezierSegment], idx: int) -> np.ndarray:
    if idx == 0:
        return normalize(segments[0].p1 - segments[0].p0)
    if idx == len(segments):
        return normalize(segments[-1].p3 - segments[-1].p2)
    prev_t = normalize(segments[idx - 1].p3 - segments[idx - 1].p2)
    next_t = normalize(segments[idx].p1 - segments[idx].p0)
    mixed = normalize(prev_t + next_t)
    if np.linalg.norm(mixed) < 1e-9:
        return next_t
    return mixed


def beziers_to_pathplanner_waypoints(
    segments: list[BezierSegment],
    cfg: PlannerConfig,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    waypoints: list[dict[str, Any]] = []
    anchor_points: list[np.ndarray] = []
    path_tangent_headings_rad: list[float] = []
    anchor_count = len(segments) + 1

    for i in range(anchor_count):
        if i == 0:
            anchor = segments[0].p0
            prev_control = None
            next_control = _point_dict(segments[0].p1)
        elif i == anchor_count - 1:
            anchor = segments[-1].p3
            prev_control = _point_dict(segments[-1].p2)
            next_control = None
        else:
            anchor = segments[i].p0
            prev_control = _point_dict(segments[i - 1].p2)
            next_control = _point_dict(segments[i].p1)

        heading_vec = _anchor_heading(segments, i)
        path_tangent_headings_rad.append(float(angle_rad(heading_vec)))
        anchor_points.append(anchor.copy())
        entry: dict[str, Any] = {
            "anchor": _point_dict(anchor),
            "prevControl": prev_control,
            "nextControl": next_control,
        }
        waypoints.append(entry)

    anchor_np = np.asarray(anchor_points, dtype=float)
    progress_u = progress_from_points(anchor_np)
    path_tangent_rad = np.asarray(path_tangent_headings_rad, dtype=float)
    holonomic_rad = compute_holonomic_rotation_profile(
        path_tangent_headings_rad=path_tangent_rad,
        progress_u=progress_u,
        cfg=cfg,
    )

    for i, entry in enumerate(waypoints):
        tangent_rad = float(path_tangent_rad[i]) if i < len(path_tangent_rad) else 0.0
        holo_rad = float(holonomic_rad[i]) if i < len(holonomic_rad) else tangent_rad
        if cfg.emit_degrees:
            entry["pathTangentHeadingDeg"] = float(math.degrees(tangent_rad))
            entry["holonomicRotationDeg"] = float(math.degrees(holo_rad))
        if cfg.emit_radians:
            entry["pathTangentHeadingRad"] = tangent_rad
            entry["holonomicRotationRad"] = holo_rad
        entry["progress"] = float(progress_u[i]) if i < len(progress_u) else 0.0

    return waypoints


def build_concept_export(
    segments: list[BezierSegment],
    waypoints: list[dict[str, Any]],
    cfg: PlannerConfig,
) -> dict[str, Any]:
    beziers = [seg.as_dict() for seg in segments]

    end_rotation_deg = cfg.end_heading_deg if cfg.end_heading_deg is not None else 0.0
    end_rotation_rad = math.radians(end_rotation_deg)
    holonomic_targets: list[dict[str, Any]] = []
    for i, wp in enumerate(waypoints):
        holonomic_targets.append(
            {
                "waypointIndex": i,
                "progress": float(wp.get("progress", 0.0)),
                "holonomicRotationDeg": float(wp.get("holonomicRotationDeg", 0.0)),
                "holonomicRotationRad": float(wp.get("holonomicRotationRad", 0.0)),
            }
        )
    return {
        "format": "PathPlanner-concept",
        "notes": [
            "Heat is high-cost traversable terrain.",
            "Blocked cells are explicitly infinite/untraversable.",
            "Path tangent heading is geometric path direction, not robot-facing heading.",
        ],
        "approachConstraints": {
            "startApproachHeadingDeg": cfg.resolved_start_approach_heading_deg,
            "goalApproachHeadingDeg": cfg.resolved_goal_approach_heading_deg,
            "startApproachLockDistanceM": float(cfg.start_approach_lock_distance_m),
            "goalApproachLockDistanceM": float(cfg.goal_approach_lock_distance_m),
        },
        "clearanceModel": {
            "objectShape": cfg.object_shape,
            "objectWidthM": float(cfg.object_width_m),
            "objectHeightM": float(cfg.object_height_m),
            "footprintRadiusM": float(cfg.footprint_radius_m),
            "safeSpaceM": float(cfg.safe_space_m),
            "requiredClearanceM": float(cfg.required_clearance_m),
        },
        "holonomicRotationMode": cfg.holonomic_rotation_mode,
        "rotationFinishProgress": float(cfg.rotation_finish_progress),
        "waypoints": waypoints,
        "bezierSegments": beziers,
        "goalEndState": {
            "velocityMps": float(cfg.end_velocity_mps),
            "rotationDeg": float(end_rotation_deg),
            "rotationRad": float(end_rotation_rad),
        },
        "holonomicRotationTargets": holonomic_targets,
    }


def build_best_effort_path_file(
    waypoints: list[dict[str, Any]],
    cfg: PlannerConfig,
) -> dict[str, Any]:
    end_rotation_deg = cfg.end_heading_deg if cfg.end_heading_deg is not None else 0.0
    return {
        "version": "best-effort-2026-02",
        "pathType": "bezier",
        "waypoints": waypoints,
        "globalConstraints": {
            "maxVelocity": float(cfg.max_speed_mps),
            "maxAcceleration": float(cfg.max_accel_mps2),
        },
        "goalEndState": {
            "velocity": float(cfg.end_velocity_mps),
            "rotation": float(end_rotation_deg),
        },
        "reversed": False,
        "metadata": {
            "warning": "Best-effort adapter. Validate against your installed PathPlanner version."
        },
    }


def build_runtime_payload_compact(
    *,
    bezier_segments: list[BezierSegment],
    sampled_points: np.ndarray,
    sampled_path_tangent_headings_rad: np.ndarray,
    sampled_holonomic_rotations_rad: np.ndarray,
    summary: dict[str, Any],
    required_clearance_m: float,
    backend_status: dict[str, str],
    cfg: PlannerConfig,
) -> dict[str, Any]:
    n = len(sampled_points)
    sampled: list[dict[str, float]] = []
    for i in range(n):
        tangent = float(sampled_path_tangent_headings_rad[i]) if i < len(sampled_path_tangent_headings_rad) else 0.0
        holo = float(sampled_holonomic_rotations_rad[i]) if i < len(sampled_holonomic_rotations_rad) else tangent
        sampled.append(
            {
                "x": float(sampled_points[i, 0]),
                "y": float(sampled_points[i, 1]),
                "pathTangentHeadingRad": tangent,
                "pathTangentHeadingDeg": float(math.degrees(tangent)),
                "holonomicRotationRad": holo,
                "holonomicRotationDeg": float(math.degrees(holo)),
            }
        )

    return {
        "format": "PathPlanner-runtime-compact",
        "runtimeMode": cfg.runtime_mode,
        "plannerMode": cfg.planner_mode,
        "computeBackend": backend_status,
        "summary": summary,
        "requiredClearanceM": float(required_clearance_m),
        "bezierSegments": [seg.as_dict() for seg in bezier_segments],
        "sampledPath": sampled,
        "goalEndState": {
            "velocityMps": float(cfg.end_velocity_mps),
            "rotationDeg": None if cfg.end_heading_deg is None else float(cfg.end_heading_deg),
            "rotationRad": None
            if cfg.end_heading_deg is None
            else float(math.radians(cfg.end_heading_deg)),
        },
    }


def write_json(path: Path, payload: dict[str, Any], compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(payload, f, separators=(",", ":"))
        else:
            json.dump(payload, f, indent=2)


def write_waypoints_csv(path: Path, waypoints: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "anchor_x",
                "anchor_y",
                "prev_x",
                "prev_y",
                "next_x",
                "next_y",
                "progress",
                "path_tangent_heading_deg",
                "path_tangent_heading_rad",
                "holonomic_rotation_deg",
                "holonomic_rotation_rad",
            ]
        )
        for i, wp in enumerate(waypoints):
            prev = wp.get("prevControl")
            nxt = wp.get("nextControl")
            writer.writerow(
                [
                    i,
                    wp["anchor"]["x"],
                    wp["anchor"]["y"],
                    "" if prev is None else prev["x"],
                    "" if prev is None else prev["y"],
                    "" if nxt is None else nxt["x"],
                    "" if nxt is None else nxt["y"],
                    wp.get("progress", ""),
                    wp.get("pathTangentHeadingDeg", ""),
                    wp.get("pathTangentHeadingRad", ""),
                    wp.get("holonomicRotationDeg", ""),
                    wp.get("holonomicRotationRad", ""),
                ]
            )
