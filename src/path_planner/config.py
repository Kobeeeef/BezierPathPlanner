from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .backend import ComputeBackend


PlannerMode = Literal["fmm", "dijkstra_approx"]
CostMode = Literal["density", "inverse_speed"]
HolonomicRotationMode = Literal["independent_profile", "tangent_follow"]
ObjectShape = Literal["rectangle", "circle"]
RuntimeMode = Literal["runtime_fast", "debug_diagnostics"]


@dataclass(frozen=True)
class PlannerConfig:
    runtime_mode: RuntimeMode = "debug_diagnostics"
    compute_backend: ComputeBackend = "cpu"
    stage_timing_enabled: bool = True
    cache_goal_fields: bool = True
    max_goal_field_cache_entries: int = 12
    resolution_m_per_cell: float = 0.1
    planner_mode: PlannerMode = "fmm"
    cost_mode: CostMode = "density"
    base_cost: float = 1.0
    alpha: float = 4.0
    epsilon: float = 1e-3
    blocked_sentinel: float | None = None
    path_step_m: float = 0.08
    goal_tolerance_m: float = 0.12
    max_extract_steps: int = 12000
    loop_quantization: float = 6.0
    start_approach_heading_deg: float | None = None
    goal_approach_heading_deg: float | None = None
    start_approach_lock_distance_m: float = 0.5
    goal_approach_lock_distance_m: float = 0.5
    terminal_progress_window_m: float = 0.75
    endpoint_alignment_tolerance_deg: float = 8.0
    endpoint_overshoot_tolerance_m: float = 0.06
    # Legacy knobs kept for compatibility with prior CLI.
    rdp_tolerance_m: float = 0.20
    sample_ds_m: float = 0.08
    handle_scale: float = 0.32
    max_curvature: float = 2.6
    curvature_iters: int = 6
    max_endpoint_curvature: float = 5.0
    endpoint_zone_m: float = 0.5
    endpoint_zone_growth: float = 1.25
    endpoint_heading_blend_power: float = 1.6
    endpoint_spacing_exponent: float = 0.55
    endpoint_handle_scale: float = 0.96
    endpoint_handle_decay: float = 1.0
    min_endpoint_handle_scale: float = 0.82
    max_endpoint_tangent_jump_deg: float = 2.0
    allow_terminal_overshoot: bool = False
    # New smoothing pipeline knobs.
    spline_smoothing: float = 1.35
    spline_smoothing_growth: float = 1.65
    max_smoothing_refits: int = 6
    runtime_fast_max_refits: int = 3
    bezier_target_segment_length_m: float = 1.8
    min_bezier_segments: int = 4
    max_bezier_segments: int = 220
    handle_clamp_ratio: float = 0.45
    min_handle_length_m: float = 0.01
    min_handle_ratio: float = 0.10
    sharp_turn_deg: float = 35.0
    sharp_turn_handle_scale: float = 0.42
    c2_regularization_weight: float = 0.35
    c2_regularization_iters: int = 2
    max_tangent_jump_deg: float = 1.0
    max_tangent_mag_jump_ratio: float = 0.22
    curvature_spike_factor: float = 8.0
    raw_tangent_worse_factor: float = 1.02
    raw_curvature_worse_factor: float = 1.08
    refit_handle_decay: float = 0.86
    refit_segment_length_growth: float = 1.12
    hook_penalty_weight: float = 3.0
    segment_count_penalty_weight: float = 0.14
    min_terminal_progress_ratio: float = 0.92
    enable_smoothing: bool = True
    enable_clearance_constraints: bool = True
    clearance_refit_threshold_m: float = 0.0
    wall_clearance_weight: float = 1.1
    wall_clearance_power: float = 2.0
    wall_clearance_soft_ratio: float = 1.6
    enforce_hard_clearance_if_feasible: bool = True
    heat_region_clearance_enabled: bool = True
    heat_region_threshold: float | None = None
    heat_region_quantile: float = 0.86
    heat_region_clearance_weight: float = 0.42
    heat_region_clearance_decay_m: float = 0.55
    object_width_m: float = 0.85
    object_height_m: float = 0.85
    object_shape: ObjectShape = "rectangle"
    safe_space_m: float = 0.12
    start_heading_deg: float | None = None
    end_heading_deg: float | None = 0.0
    holonomic_rotation_mode: HolonomicRotationMode = "independent_profile"
    rotation_finish_progress: float = 0.8
    start_velocity_mps: float = 0.0
    end_velocity_mps: float = 0.0
    max_speed_mps: float = 4.5
    max_accel_mps2: float = 2.8
    max_centripetal_accel_mps2: float = 2.2
    emit_degrees: bool = True
    emit_radians: bool = True
    deterministic_seed: int = 0

    def with_updates(self, **kwargs: object) -> "PlannerConfig":
        data = self.__dict__.copy()
        data.update(kwargs)
        return PlannerConfig(**data)

    @property
    def resolved_start_approach_heading_deg(self) -> float | None:
        if self.start_approach_heading_deg is not None:
            return self.start_approach_heading_deg
        return self.start_heading_deg

    @property
    def resolved_goal_approach_heading_deg(self) -> float | None:
        if self.goal_approach_heading_deg is not None:
            return self.goal_approach_heading_deg
        return self.end_heading_deg

    @property
    def footprint_radius_m(self) -> float:
        return 0.5 * ((self.object_width_m**2 + self.object_height_m**2) ** 0.5)

    @property
    def required_clearance_m(self) -> float:
        return float(self.footprint_radius_m + self.safe_space_m)

    @property
    def fast_runtime(self) -> bool:
        return self.runtime_mode == "runtime_fast"
