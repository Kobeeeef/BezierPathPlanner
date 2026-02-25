from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .config import PlannerConfig
from .export_pathplanner import (
    beziers_to_pathplanner_waypoints,
    build_concept_export,
    build_runtime_payload_compact,
    write_json,
    write_waypoints_csv,
)
from .planner import PlannerArtifacts, run_planner
from .runtime import PlannerRuntime
from .scenarios import Scenario, get_scenarios


def _fmt(value: float | None, ndigits: int = 3) -> str:
    if value is None:
        return ""
    if ndigits <= 0:
        return str(int(round(float(value))))
    return f"{float(value):.{ndigits}f}".rstrip("0").rstrip(".")


@dataclass
class VisualizerMapState:
    scenario_name: str
    heat: np.ndarray
    blocked_mask: np.ndarray
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    resolution_m_per_cell: float


class InteractivePlannerVisualizer:
    def __init__(self, initial_scenario: str = "hot_island") -> None:
        self._scenarios = get_scenarios()
        if initial_scenario not in self._scenarios:
            initial_scenario = "hot_island"
        self._scenario_names = sorted(self._scenarios.keys())
        self._default_cfg = PlannerConfig(runtime_mode="runtime_fast")

        self._runtime = PlannerRuntime(max_goal_cache_entries=16)
        self._environment_revision = 0
        self._pending_plan_job: str | None = None
        self._suspend_replan = False
        self._planning_now = False
        self._drag_target: str | None = None
        self._painting = False
        self._last_paint_rc: tuple[int, int] | None = None
        self._brush_block_value: bool | None = None
        self._next_click_sets_start = True
        self._last_export_ms = 0.0

        self._artifacts: PlannerArtifacts | None = None
        self._current_cfg: PlannerConfig | None = None
        self._last_error: str | None = None

        self.map_state = self._scenario_to_map_state(self._scenarios[initial_scenario])

        self.root = tk.Tk()
        self.root.title("Planner Interactive Visualizer")
        self.root.geometry("1720x980")

        self.status_var = tk.StringVar(value="Ready.")
        self.warning_var = tk.StringVar(value="")
        self.diagnostics_var = tk.StringVar(value="Run a plan to view diagnostics.")

        self.vars: dict[str, tk.Variable] = {}
        self._build_ui()
        self._load_scenario(initial_scenario, apply_scenario_endpoint_defaults=True)
        self._schedule_plan(immediate=True)

    @staticmethod
    def _scenario_to_map_state(scenario: Scenario) -> VisualizerMapState:
        blocked = (
            np.asarray(scenario.blocked_mask, dtype=bool).copy()
            if scenario.blocked_mask is not None
            else np.zeros_like(scenario.heat, dtype=bool)
        )
        return VisualizerMapState(
            scenario_name=scenario.name,
            heat=np.asarray(scenario.heat, dtype=float).copy(),
            blocked_mask=blocked,
            start_xy=(float(scenario.start_xy[0]), float(scenario.start_xy[1])),
            goal_xy=(float(scenario.goal_xy[0]), float(scenario.goal_xy[1])),
            resolution_m_per_cell=float(scenario.resolution_m_per_cell),
        )

    def _build_ui(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        map_frame = ttk.Frame(paned)
        panel_frame = ttk.Frame(paned, width=520)
        paned.add(map_frame, weight=7)
        paned.add(panel_frame, weight=4)

        self.figure = Figure(figsize=(10.5, 8.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=map_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, map_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._build_plot_artists()
        self._connect_plot_events()

        self.controls_canvas = tk.Canvas(panel_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel_frame, orient=tk.VERTICAL, command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls_inner = ttk.Frame(self.controls_canvas)
        self.controls_canvas.create_window((0, 0), window=self.controls_inner, anchor="nw")
        self.controls_inner.bind("<Configure>", self._on_controls_configure)
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        self._build_control_panel(self.controls_inner)

    def _build_plot_artists(self) -> None:
        extent = self._extent()
        heat = self.map_state.heat
        self.background_im = self.ax.imshow(
            heat,
            origin="lower",
            cmap="viridis",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
        blocked_mask = np.ma.masked_where(~self.map_state.blocked_mask, self.map_state.blocked_mask.astype(float))
        self.blocked_im = self.ax.imshow(
            blocked_mask,
            origin="lower",
            cmap="gray",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            alpha=0.35,
        )

        (self.raw_line,) = self.ax.plot([], [], color="#f97316", linewidth=1.9, label="Raw path")
        (self.smooth_line,) = self.ax.plot([], [], color="#22c55e", linewidth=2.8, label="Smoothed Bezier")
        (self.start_marker,) = self.ax.plot([], [], marker="o", markersize=9, color="#65a30d", mec="black")
        (self.goal_marker,) = self.ax.plot([], [], marker="X", markersize=9, color="#dc2626", mec="black")

        self.control_lines = LineCollection([], colors="#86efac", linewidths=1.0, alpha=0.55)
        self.ax.add_collection(self.control_lines)
        self.control_points = self.ax.scatter([], [], s=14, c="#86efac", alpha=0.8)
        self.sample_points = self.ax.scatter([], [], s=8, c="#f8fafc", alpha=0.85)

        self.start_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", color="#38bdf8", lw=2.0, mutation_scale=13)
        self.goal_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", color="#38bdf8", lw=2.0, mutation_scale=13)
        self.ax.add_patch(self.start_arrow)
        self.ax.add_patch(self.goal_arrow)

        self.tangent_quiver = None
        self.holonomic_quiver = None

        self.overlay_text = self.ax.text(
            0.012,
            0.988,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            color="white",
            bbox={"facecolor": "#111827", "alpha": 0.68, "pad": 5.5},
        )
        self.overlay_warning = self.ax.text(
            0.012,
            0.012,
            "",
            transform=self.ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9.0,
            color="#fecaca",
            bbox={"facecolor": "#7f1d1d", "alpha": 0.78, "pad": 4.0},
        )

        self.colorbar = self.figure.colorbar(self.background_im, ax=self.ax, fraction=0.046, pad=0.03)
        self.colorbar.set_label("Heat")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Interactive Planner Map")
        self.ax.set_xlim(0.0, extent[1])
        self.ax.set_ylim(0.0, extent[3])

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Click map: goal update (default). Right click: set start. "
                "Drag markers directly or switch interaction modes."
            ),
            wraplength=480,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(parent, textvariable=self.status_var, anchor="w", fg="#1f2937").pack(fill=tk.X, padx=10, pady=(2, 2))
        self.warning_label = tk.Label(parent, textvariable=self.warning_var, anchor="w", fg="#991b1b", justify=tk.LEFT)
        self.warning_label.pack(fill=tk.X, padx=10, pady=(0, 8))

        diag_frame = ttk.LabelFrame(parent, text="Live Diagnostics")
        diag_frame.pack(fill=tk.X, padx=10, pady=(2, 8))
        tk.Label(
            diag_frame,
            textvariable=self.diagnostics_var,
            justify=tk.LEFT,
            anchor="nw",
            font=("Consolas", 9),
            wraplength=470,
        ).pack(fill=tk.X, padx=8, pady=8)

        self._build_scenario_section(parent)
        self._build_planner_section(parent)
        self._build_smoothing_section(parent)
        self._build_endpoint_section(parent)
        self._build_clearance_section(parent)
        self._build_runtime_section(parent)
        self._build_visual_section(parent)
        self._build_export_section(parent)

        ttk.Button(parent, text="Run Plan Now", command=lambda: self._schedule_plan(immediate=True)).pack(
            fill=tk.X, padx=10, pady=(6, 12)
        )

    def _build_scenario_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Scenario / Map Editing")
        section.pack(fill=tk.X, padx=10, pady=6)

        self._add_choice(
            section,
            key="scenario_name",
            label="Predefined scenario",
            default=self.map_state.scenario_name,
            values=self._scenario_names,
            on_change=self._on_scenario_selected,
        )
        self._add_choice(
            section,
            key="interaction_mode",
            label="Interaction mode",
            default="goal_click",
            values=[
                "goal_click",
                "start_then_goal",
                "drag_markers",
                "paint_heat",
                "erase_heat",
                "toggle_blocked",
            ],
            live_replan=False,
        )
        self._add_entry(section, key="brush_radius_m", label="Brush radius (m)", default=_fmt(0.35))
        self._add_entry(section, key="brush_heat_delta", label="Brush heat delta", default=_fmt(1.0))
        self._add_entry(section, key="random_seed", label="Random seed", default="0", live_replan=False)

        row = ttk.Frame(section)
        row.pack(fill=tk.X, padx=7, pady=4)
        ttk.Button(row, text="Reset Map", command=self._on_reset_map).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row, text="Clear Blocked", command=self._on_clear_blocked).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row, text="Randomize Heat", command=self._on_randomize_heat).pack(side=tk.LEFT)

        row2 = ttk.Frame(section)
        row2.pack(fill=tk.X, padx=7, pady=(2, 6))
        ttk.Button(row2, text="Save Scenario Config", command=self._save_scenario_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="Load Scenario Config", command=self._load_scenario_config).pack(side=tk.LEFT)

    def _build_planner_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Planner / Cost")
        section.pack(fill=tk.X, padx=10, pady=6)
        cfg = self._default_cfg
        self._add_choice(
            section,
            key="planner_mode",
            label="planner_mode",
            default=cfg.planner_mode,
            values=["fmm", "dijkstra_approx"],
        )
        self._add_choice(
            section,
            key="cost_mode",
            label="cost_mode",
            default=cfg.cost_mode,
            values=["density", "inverse_speed"],
        )
        self._add_entry(section, key="base_cost", label="base_cost", default=_fmt(cfg.base_cost))
        self._add_entry(section, key="alpha", label="alpha", default=_fmt(cfg.alpha))
        self._add_entry(section, key="epsilon", label="epsilon", default=_fmt(cfg.epsilon, ndigits=5))

    def _build_smoothing_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Smoothing")
        section.pack(fill=tk.X, padx=10, pady=6)
        cfg = self._default_cfg
        self._add_check(section, key="enable_smoothing", label="Enable smoothing", default=cfg.enable_smoothing)
        self._add_choice(
            section,
            key="smoothing_quality",
            label="Smoothing quality",
            default="fast",
            values=["fast", "balanced", "high"],
        )
        self._add_entry(section, key="fit_tension", label="fit_tension", default=_fmt(cfg.spline_smoothing))
        self._add_entry(section, key="max_curvature", label="max_curvature", default=_fmt(cfg.max_curvature))
        self._add_entry(
            section,
            key="handle_clamp_ratio",
            label="handle_scale_limit (max)",
            default=_fmt(cfg.handle_clamp_ratio),
        )
        self._add_entry(
            section,
            key="min_handle_ratio",
            label="handle_scale_limit (min)",
            default=_fmt(cfg.min_handle_ratio),
        )
        self._add_entry(section, key="sample_ds_m", label="sample_ds_m", default=_fmt(cfg.sample_ds_m))
        self._add_entry(
            section,
            key="endpoint_blend_strength",
            label="endpoint blend strength",
            default=_fmt(cfg.endpoint_heading_blend_power),
        )
        self._add_entry(
            section,
            key="start_approach_lock_distance_m",
            label="start_approach_lock_distance_m",
            default=_fmt(cfg.start_approach_lock_distance_m),
        )
        self._add_entry(
            section,
            key="goal_approach_lock_distance_m",
            label="goal_approach_lock_distance_m",
            default=_fmt(cfg.goal_approach_lock_distance_m),
        )
        self._add_entry(section, key="endpoint_zone_m", label="endpoint_zone_m", default=_fmt(cfg.endpoint_zone_m))
        self._add_entry(
            section,
            key="max_endpoint_curvature",
            label="max_endpoint_curvature",
            default=_fmt(cfg.max_endpoint_curvature),
        )

    def _build_endpoint_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Endpoint / Terminal Behavior")
        section.pack(fill=tk.X, padx=10, pady=6)
        cfg = self._default_cfg
        self._add_entry(section, key="start_approach_heading_deg", label="start_approach_heading", default="")
        self._add_entry(section, key="goal_approach_heading_deg", label="goal_approach_heading", default="")
        self._add_entry(section, key="start_velocity_mps", label="start_velocity_mps", default=_fmt(cfg.start_velocity_mps))
        self._add_entry(section, key="end_velocity_mps", label="end_velocity_mps", default=_fmt(cfg.end_velocity_mps))
        self._add_entry(section, key="start_heading_deg", label="start_heading (holonomic)", default="")
        self._add_entry(section, key="goal_heading_deg", label="goal_heading (holonomic)", default=_fmt(cfg.end_heading_deg))
        self._add_entry(
            section,
            key="rotation_finish_progress",
            label="rotation_finish_progress",
            default=_fmt(cfg.rotation_finish_progress),
        )
        self._add_choice(
            section,
            key="holonomic_rotation_mode",
            label="holonomic rotation mode",
            default=cfg.holonomic_rotation_mode,
            values=["independent_profile", "tangent_follow"],
        )
        self._add_entry(
            section,
            key="terminal_progress_window_m",
            label="terminal_progress_window_m",
            default=_fmt(cfg.terminal_progress_window_m),
        )
        self._add_check(
            section,
            key="allow_terminal_overshoot",
            label="Allow terminal overshoot",
            default=cfg.allow_terminal_overshoot,
        )

    def _build_clearance_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Footprint / Clearance")
        section.pack(fill=tk.X, padx=10, pady=6)
        cfg = self._default_cfg
        self._add_check(section, key="use_blocked_mask", label="Use blocked mask for planning", default=True)
        self._add_check(
            section,
            key="enable_clearance_constraints",
            label="Enable clearance constraints",
            default=cfg.enable_clearance_constraints,
        )
        self._add_check(
            section,
            key="enforce_hard_clearance_if_feasible",
            label="Hard clearance when feasible",
            default=cfg.enforce_hard_clearance_if_feasible,
        )
        self._add_entry(section, key="object_width_m", label="object_width_m", default=_fmt(cfg.object_width_m))
        self._add_entry(section, key="object_height_m", label="object_height_m", default=_fmt(cfg.object_height_m))
        self._add_choice(
            section,
            key="object_shape",
            label="object_shape",
            default=cfg.object_shape,
            values=["rectangle", "circle"],
        )
        self._add_entry(section, key="safe_space_m", label="safe_space_m", default=_fmt(cfg.safe_space_m))
        self._add_check(
            section,
            key="heat_region_clearance_enabled",
            label="Heat-region clearance preference",
            default=cfg.heat_region_clearance_enabled,
        )
        self._add_entry(section, key="heat_region_threshold", label="heat_region_threshold", default="")

    def _build_runtime_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Runtime / Performance")
        section.pack(fill=tk.X, padx=10, pady=6)
        self._add_check(section, key="runtime_fast", label="runtime_fast (default)", default=True)
        self._add_choice(
            section,
            key="compute_backend",
            label="backend",
            default="cpu",
            values=["cpu", "gpu"],
        )
        self._add_check(section, key="cache_goal_fields", label="Cache goal fields", default=True)
        self._add_entry(section, key="max_goal_cache_entries", label="max_goal_cache_entries", default="16")
        self._add_entry(section, key="debounce_ms", label="UI debounce ms", default="110", live_replan=False)

    def _build_visual_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Visualization Toggles")
        section.pack(fill=tk.X, padx=10, pady=6)
        self._add_check(section, key="show_heatmap", label="show heatmap", default=True)
        self._add_check(section, key="show_cost_to_go", label="show cost-to-go field", default=False)
        self._add_check(section, key="show_clearance_field", label="show clearance field", default=False)
        self._add_check(
            section,
            key="show_heat_region_clearance_field",
            label="show heat-region clearance field",
            default=False,
        )
        self._add_check(section, key="show_blocked_cells", label="show blocked cells", default=True)
        self._add_check(section, key="show_raw_path", label="show raw path", default=False)
        self._add_check(section, key="show_smoothed_bezier", label="show smoothed Bezier", default=True)
        self._add_check(
            section,
            key="show_control_polygons",
            label="show control polygons / handles",
            default=False,
        )
        self._add_check(section, key="show_path_samples", label="show path samples", default=False)
        self._add_check(section, key="show_tangent_arrows", label="show tangent arrows", default=False)
        self._add_check(
            section,
            key="show_holonomic_rotation_arrows",
            label="show holonomic rotation arrows",
            default=False,
        )
        self._add_check(section, key="show_endpoint_arrows", label="show endpoint approach arrows", default=True)
        self._add_check(
            section,
            key="show_diagnostics_overlay",
            label="show diagnostics text overlay",
            default=True,
        )

    def _build_export_section(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Export / Presets")
        section.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(section, text="Export runtime payload JSON", command=self._export_runtime_payload).pack(
            fill=tk.X, padx=7, pady=(6, 2)
        )
        ttk.Button(section, text="Export concept JSON + CSV", command=self._export_concept_bundle).pack(
            fill=tk.X, padx=7, pady=2
        )
        ttk.Button(section, text="Save screenshot", command=self._save_screenshot).pack(fill=tk.X, padx=7, pady=2)
        ttk.Button(section, text="Save parameter preset", command=self._save_parameter_preset).pack(
            fill=tk.X, padx=7, pady=2
        )
        ttk.Button(section, text="Load parameter preset", command=self._load_parameter_preset).pack(
            fill=tk.X, padx=7, pady=(2, 8)
        )

    def _add_entry(
        self,
        parent: ttk.Frame,
        *,
        key: str,
        label: str,
        default: str,
        live_replan: bool = True,
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=7, pady=2)
        ttk.Label(row, text=label, width=30, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        entry = ttk.Entry(row, textvariable=var, width=16)
        entry.pack(side=tk.RIGHT)
        if live_replan:
            var.trace_add("write", self._on_live_control_changed)
        return var

    def _add_choice(
        self,
        parent: ttk.Frame,
        *,
        key: str,
        label: str,
        default: str,
        values: list[str],
        on_change: Any | None = None,
        live_replan: bool = True,
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=7, pady=2)
        ttk.Label(row, text=label, width=30, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        combo = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=14)
        combo.pack(side=tk.RIGHT)
        if on_change is not None:
            combo.bind("<<ComboboxSelected>>", on_change)
        elif live_replan:
            combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_plan())
        return var

    def _add_check(
        self,
        parent: ttk.Frame,
        *,
        key: str,
        label: str,
        default: bool,
        live_replan: bool = True,
    ) -> tk.BooleanVar:
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        cb = ttk.Checkbutton(
            parent,
            text=label,
            variable=var,
            command=(self._schedule_plan if live_replan else None),
        )
        cb.pack(fill=tk.X, padx=7, pady=1)
        return var

    def _on_controls_configure(self, _event: tk.Event) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self.controls_canvas.winfo_exists():
            return
        if self.controls_canvas.winfo_containing(event.x_root, event.y_root) is None:
            return
        delta = -1 if event.delta < 0 else 1
        self.controls_canvas.yview_scroll(-delta, "units")

    def _connect_plot_events(self) -> None:
        self.canvas.mpl_connect("button_press_event", self._on_map_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_map_motion)
        self.canvas.mpl_connect("button_release_event", self._on_map_release)

    @contextmanager
    def _suspended_replan(self):
        prev = self._suspend_replan
        self._suspend_replan = True
        try:
            yield
        finally:
            self._suspend_replan = prev

    def _on_live_control_changed(self, *_args: object) -> None:
        self._schedule_plan()

    def _schedule_plan(self, *_args: object, immediate: bool = False) -> None:
        if self._suspend_replan:
            return
        if self._pending_plan_job is not None:
            self.root.after_cancel(self._pending_plan_job)
            self._pending_plan_job = None
        if immediate:
            delay_ms = 1
        else:
            try:
                delay_ms = self._read_int("debounce_ms", default=110, minimum=0)
            except ValueError:
                delay_ms = 110
        self._pending_plan_job = self.root.after(delay_ms, self._run_plan)

    def _run_plan(self) -> None:
        self._pending_plan_job = None
        if self._planning_now:
            self._schedule_plan()
            return

        cfg = self._build_cfg_from_controls()
        if cfg is None:
            return
        self._planning_now = True
        t0 = time.perf_counter()
        try:
            blocked = self.map_state.blocked_mask if self._read_bool("use_blocked_mask", default=True) else None
            artifacts = run_planner(
                heat=self.map_state.heat,
                start_xy=self.map_state.start_xy,
                goal_xy=self.map_state.goal_xy,
                cfg=cfg,
                blocked_mask=blocked,
                runtime=self._runtime,
                environment_id=f"{self.map_state.scenario_name}:{self._environment_revision}",
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._artifacts = artifacts
            self._current_cfg = cfg
            self._last_error = None
            self.status_var.set(
                f"Planned in {dt_ms:.2f} ms. planner={cfg.planner_mode} backend={artifacts.backend_status.get('used', 'cpu')}"
            )
            self._update_quality_warnings(artifacts)
        except Exception as exc:
            self._artifacts = None
            self._current_cfg = cfg
            self._last_error = str(exc)
            self.warning_var.set(f"Planning failed: {self._classify_failure(str(exc))}")
            self.warning_label.configure(fg="#991b1b")
            self.status_var.set("Planning error. Adjust map or parameters.")
        finally:
            self._planning_now = False
            self._refresh_plot()
            self._refresh_diagnostics()

    def _classify_failure(self, msg: str) -> str:
        text = msg.strip()
        low = text.lower()
        if "outside map bounds" in low:
            return f"invalid input ({text})"
        if "start cell is blocked" in low or "goal cell is blocked" in low:
            return text
        if "no finite path" in low:
            return "disconnected due to blocked cells/constraints"
        if "clearance" in low:
            return f"constraint conflict ({text})"
        return text

    def _update_quality_warnings(self, artifacts: PlannerArtifacts) -> None:
        smoothing_diag = artifacts.smoothing_diagnostics or {}
        smoothed = smoothing_diag.get("smoothedPathDiagnostics", {}) if isinstance(smoothing_diag, dict) else {}
        warns: list[str] = []
        if bool(smoothing_diag.get("refitTriggered", False)):
            warns.append("smoothing refit/selection triggered")
        if float(smoothed.get("terminalOvershootCount", 0.0)) > 0.0:
            warns.append("endpoint overshoot detected")
        if float(smoothed.get("terminalGoalProjectionMonotonicViolations", 0.0)) > 0.0:
            warns.append("terminal hook projection violation")
        if float(smoothed.get("terminalGoalDistanceMonotonicViolations", 0.0)) > 0.0:
            warns.append("terminal hook distance violation")
        if float(smoothed.get("endpointSelfIntersectionCount", 0.0)) > 0.0:
            warns.append("endpoint self-intersection detected")
        if warns:
            self.warning_var.set("; ".join(warns))
            self.warning_label.configure(fg="#991b1b")
        else:
            self.warning_var.set("No quality guard warnings.")
            self.warning_label.configure(fg="#166534")

    def _refresh_plot(self) -> None:
        field, cmap, label = self._current_background_field()
        self.background_im.set_data(field)
        self.background_im.set_extent(self._extent())
        self.background_im.set_cmap(cmap)
        finite = np.asarray(field)[np.isfinite(field)]
        if finite.size:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmax - vmin < 1e-9:
                vmax = vmin + 1.0
            self.background_im.set_clim(vmin, vmax)
        self.colorbar.update_normal(self.background_im)
        self.colorbar.set_label(label)

        show_blocked = self._read_bool("show_blocked_cells", default=True)
        blocked_mask = (
            np.ma.masked_where(~self.map_state.blocked_mask, self.map_state.blocked_mask.astype(float))
            if show_blocked
            else np.ma.masked_all_like(self.map_state.blocked_mask.astype(float))
        )
        self.blocked_im.set_data(blocked_mask)
        self.blocked_im.set_extent(self._extent())
        self.blocked_im.set_visible(show_blocked and np.any(self.map_state.blocked_mask))

        sx, sy = self.map_state.start_xy
        gx, gy = self.map_state.goal_xy
        self.start_marker.set_data([sx], [sy])
        self.goal_marker.set_data([gx], [gy])

        art = self._artifacts
        if art is None:
            self.raw_line.set_data([], [])
            self.smooth_line.set_data([], [])
            self.control_lines.set_segments([])
            self.control_points.set_offsets(np.empty((0, 2)))
            self.sample_points.set_offsets(np.empty((0, 2)))
            self._set_quiver("tangent_quiver", None, None, "#60a5fa", False)
            self._set_quiver("holonomic_quiver", None, None, "#eab308", False)
            self.start_arrow.set_visible(False)
            self.goal_arrow.set_visible(False)
        else:
            raw = np.asarray(art.raw_path_world, dtype=float) if art.raw_path_world else np.empty((0, 2))
            smooth = np.asarray(art.sampled_smoothed_points, dtype=float)

            if self._read_bool("show_raw_path", default=False) and len(raw) > 1:
                self.raw_line.set_data(raw[:, 0], raw[:, 1])
                self.raw_line.set_visible(True)
            else:
                self.raw_line.set_data([], [])
                self.raw_line.set_visible(False)

            if self._read_bool("show_smoothed_bezier", default=True) and len(smooth) > 1:
                self.smooth_line.set_data(smooth[:, 0], smooth[:, 1])
                self.smooth_line.set_visible(True)
            else:
                self.smooth_line.set_data([], [])
                self.smooth_line.set_visible(False)

            show_controls = self._read_bool("show_control_polygons", default=False)
            if show_controls and art.bezier_segments:
                segments: list[np.ndarray] = []
                ctrl_pts: list[np.ndarray] = []
                for seg in art.bezier_segments:
                    segments.append(np.vstack([seg.p0, seg.p1]))
                    segments.append(np.vstack([seg.p2, seg.p3]))
                    ctrl_pts.append(np.asarray(seg.p1, dtype=float))
                    ctrl_pts.append(np.asarray(seg.p2, dtype=float))
                self.control_lines.set_segments(segments)
                self.control_points.set_offsets(np.asarray(ctrl_pts, dtype=float))
                self.control_lines.set_visible(True)
                self.control_points.set_visible(True)
            else:
                self.control_lines.set_segments([])
                self.control_points.set_offsets(np.empty((0, 2)))
                self.control_lines.set_visible(False)
                self.control_points.set_visible(False)

            if self._read_bool("show_path_samples", default=False) and len(smooth) > 0:
                stride = max(1, len(smooth) // 140)
                samples = smooth[::stride]
                self.sample_points.set_offsets(samples)
                self.sample_points.set_visible(True)
            else:
                self.sample_points.set_offsets(np.empty((0, 2)))
                self.sample_points.set_visible(False)

            show_tangent = self._read_bool("show_tangent_arrows", default=False)
            self._set_quiver(
                "tangent_quiver",
                smooth,
                np.asarray(art.sampled_path_tangent_headings_rad, dtype=float),
                "#60a5fa",
                show_tangent,
            )
            show_holo = self._read_bool("show_holonomic_rotation_arrows", default=False)
            self._set_quiver(
                "holonomic_quiver",
                smooth,
                np.asarray(art.sampled_holonomic_rotations_rad, dtype=float),
                "#facc15",
                show_holo,
            )
            self._update_endpoint_arrows(art)

        x_max = self.map_state.heat.shape[1] * self.map_state.resolution_m_per_cell
        y_max = self.map_state.heat.shape[0] * self.map_state.resolution_m_per_cell
        self.ax.set_xlim(0.0, x_max)
        self.ax.set_ylim(0.0, y_max)
        self._update_overlay_text()
        self.canvas.draw_idle()

    def _set_quiver(
        self,
        attr_name: str,
        points: np.ndarray | None,
        headings: np.ndarray | None,
        color: str,
        visible: bool,
    ) -> None:
        current = getattr(self, attr_name)
        if current is not None:
            current.remove()
            setattr(self, attr_name, None)
        if not visible or points is None or headings is None or len(points) == 0 or len(headings) == 0:
            return
        n = min(len(points), len(headings))
        if n <= 0:
            return
        stride = max(1, n // 28)
        pts = points[:n:stride]
        hd = headings[:n:stride]
        if len(pts) == 0:
            return
        length = max(0.24, 3.2 * self.map_state.resolution_m_per_cell)
        u = length * np.cos(hd)
        v = length * np.sin(hd)
        q = self.ax.quiver(
            pts[:, 0],
            pts[:, 1],
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.0028,
            alpha=0.9,
        )
        setattr(self, attr_name, q)

    def _update_endpoint_arrows(self, artifacts: PlannerArtifacts) -> None:
        show = self._read_bool("show_endpoint_arrows", default=True)
        if not show:
            self.start_arrow.set_visible(False)
            self.goal_arrow.set_visible(False)
            return
        cfg = self._current_cfg
        if cfg is None:
            self.start_arrow.set_visible(False)
            self.goal_arrow.set_visible(False)
            return
        length = max(0.32, 4.0 * self.map_state.resolution_m_per_cell)

        start_heading_deg = cfg.resolved_start_approach_heading_deg
        if start_heading_deg is None and len(artifacts.sampled_path_tangent_headings_rad) > 0:
            start_theta = float(artifacts.sampled_path_tangent_headings_rad[0])
        else:
            start_theta = math.radians(float(start_heading_deg or 0.0))
        goal_heading_deg = cfg.resolved_goal_approach_heading_deg
        if goal_heading_deg is None and len(artifacts.sampled_path_tangent_headings_rad) > 0:
            goal_theta = float(artifacts.sampled_path_tangent_headings_rad[-1])
        else:
            goal_theta = math.radians(float(goal_heading_deg or 0.0))

        sx, sy = self.map_state.start_xy
        gx, gy = self.map_state.goal_xy
        self.start_arrow.set_positions((sx, sy), (sx + length * math.cos(start_theta), sy + length * math.sin(start_theta)))
        self.goal_arrow.set_positions((gx - length * math.cos(goal_theta), gy - length * math.sin(goal_theta)), (gx, gy))
        self.start_arrow.set_visible(True)
        self.goal_arrow.set_visible(True)

    def _update_overlay_text(self) -> None:
        show_overlay = self._read_bool("show_diagnostics_overlay", default=True)
        if not show_overlay:
            self.overlay_text.set_visible(False)
            self.overlay_warning.set_visible(False)
            return
        self.overlay_text.set_visible(True)
        if self._last_error is not None:
            self.overlay_text.set_text(f"Planning failed:\n{self._classify_failure(self._last_error)}")
        elif self._artifacts is not None:
            summary = self._artifacts.summary
            timing = self._artifacts.stage_timings_ms
            smoothed = self._artifacts.smoothing_diagnostics.get("smoothedPathDiagnostics", {})
            text = (
                f"total={timing.get('totalRuntimeMs', 0.0):.1f} ms  planner={summary.get('plannerMode', '-')}\n"
                f"path={summary.get('smoothedLengthM', 0.0):.2f} m  cost={summary.get('smoothedIntegratedCost', 0.0):.2f}\n"
                f"minClr={summary.get('minWallClearanceM', 0.0):.3f}/{summary.get('requiredClearanceM', 0.0):.3f} m\n"
                f"maxCurv={smoothed.get('maxCurvature', 0.0):.3f}  alignErr={summary.get('startEndpointAlignmentErrorDeg', 0.0):.1f}/{summary.get('endEndpointAlignmentErrorDeg', 0.0):.1f} deg"
            )
            self.overlay_text.set_text(text)
        else:
            self.overlay_text.set_text("No plan yet.")

        warn = self.warning_var.get().strip()
        if warn and warn.lower() != "no quality guard warnings.":
            self.overlay_warning.set_text(warn)
            self.overlay_warning.set_visible(True)
        else:
            self.overlay_warning.set_visible(False)

    def _refresh_diagnostics(self) -> None:
        if self._last_error is not None:
            self.diagnostics_var.set(f"Planning failed: {self._classify_failure(self._last_error)}")
            return
        if self._artifacts is None or self._current_cfg is None:
            self.diagnostics_var.set("No plan yet.")
            return
        a = self._artifacts
        s = a.summary
        timing = a.stage_timings_ms
        smoothed = a.smoothing_diagnostics.get("smoothedPathDiagnostics", {})
        heat_min = s.get("minHeatRegionClearanceM", -1.0)
        if heat_min is None or float(heat_min) < 0.0:
            heat_text = "n/a"
        else:
            heat_text = f"{float(heat_min):.3f} m"
        lines = [
            f"total runtime: {timing.get('totalRuntimeMs', 0.0):.2f} ms",
            (
                "stage timings [ms]: "
                f"propagation={timing.get('propagation', 0.0):.2f}, "
                f"extraction={timing.get('extraction', 0.0):.2f}, "
                f"smoothing={timing.get('smoothing', 0.0):.2f}, "
                f"export={self._last_export_ms:.2f}"
            ),
            f"path length: {s.get('smoothedLengthM', 0.0):.3f} m",
            (
                "integrated path cost: "
                f"{s.get('smoothedIntegratedCost', 0.0):.3f} "
                f"(objective={s.get('smoothedObjectiveIntegratedCost', 0.0):.3f})"
            ),
            (
                "min wall clearance / required: "
                f"{s.get('minWallClearanceM', 0.0):.3f} / {s.get('requiredClearanceM', 0.0):.3f} m"
            ),
            f"min heat-region clearance: {heat_text}",
            f"max curvature: {smoothed.get('maxCurvature', 0.0):.4f}",
            (
                "endpoint alignment error [deg]: "
                f"start={s.get('startEndpointAlignmentErrorDeg', 0.0):.2f}, "
                f"end={s.get('endEndpointAlignmentErrorDeg', 0.0):.2f}"
            ),
            f"planner/backend: {s.get('plannerMode', '-')}/{a.backend_status.get('used', 'cpu')}",
            (
                "smoothing attempts: "
                f"{a.smoothing_diagnostics.get('attemptCount', 0)} "
                f"(accepted={a.smoothing_diagnostics.get('acceptedAttempt', -1)})"
            ),
        ]
        self.diagnostics_var.set("\n".join(lines))

    def _current_background_field(self) -> tuple[np.ndarray, str, str]:
        art = self._artifacts
        if self._read_bool("show_cost_to_go", default=False) and art is not None:
            data = np.asarray(art.t_field, dtype=float).copy()
            finite = data[np.isfinite(data)]
            if finite.size:
                fill = float(np.percentile(finite, 97))
            else:
                fill = 0.0
            data[~np.isfinite(data)] = fill
            return data, "magma", "Cost-to-go"
        if self._read_bool("show_clearance_field", default=False) and art is not None:
            data = self._prepare_distance_field_for_display(art.wall_clearance_m)
            return data, "cividis", "Wall clearance (m)"
        if self._read_bool("show_heat_region_clearance_field", default=False) and art is not None:
            data = self._prepare_distance_field_for_display(art.heat_region_clearance_m)
            return data, "plasma", "Heat-region clearance (m)"
        if not self._read_bool("show_heatmap", default=True):
            blank = np.zeros_like(self.map_state.heat, dtype=float)
            return blank, "Greys", "Background off"
        return np.asarray(self.map_state.heat, dtype=float), "viridis", "Heat"

    @staticmethod
    def _prepare_distance_field_for_display(field: np.ndarray) -> np.ndarray:
        out = np.asarray(field, dtype=float).copy()
        finite = out[np.isfinite(out)]
        if finite.size == 0:
            out.fill(0.0)
            return out
        clip_hi = float(np.percentile(finite, 98))
        out[~np.isfinite(out)] = clip_hi
        np.clip(out, 0.0, clip_hi, out=out)
        return out

    def _extent(self) -> list[float]:
        h, w = self.map_state.heat.shape
        res = self.map_state.resolution_m_per_cell
        return [0.0, w * res, 0.0, h * res]

    def _on_map_press(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = self._clamp_world(event.xdata, event.ydata)

        if int(event.button) == 3:
            self.map_state.start_xy = (x, y)
            self._schedule_plan(immediate=True)
            self._refresh_plot()
            return
        if int(event.button) != 1:
            return

        mode = self._read_choice("interaction_mode", default="goal_click")
        if mode in ("paint_heat", "erase_heat", "toggle_blocked"):
            self._painting = True
            self._apply_brush(x, y, mode, start_stroke=True)
            self._refresh_plot()
            self._schedule_plan()
            return

        marker = self._closest_marker(x, y)
        if marker is not None and mode in ("goal_click", "start_then_goal", "drag_markers"):
            self._drag_target = marker
            self._set_marker(marker, x, y)
            self._refresh_plot()
            self._schedule_plan(immediate=True)
            return

        if mode == "start_then_goal":
            if self._next_click_sets_start:
                self.map_state.start_xy = (x, y)
            else:
                self.map_state.goal_xy = (x, y)
            self._next_click_sets_start = not self._next_click_sets_start
        else:
            self.map_state.goal_xy = (x, y)
        self._refresh_plot()
        self._schedule_plan(immediate=True)

    def _on_map_motion(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = self._clamp_world(event.xdata, event.ydata)
        if self._drag_target is not None:
            self._set_marker(self._drag_target, x, y)
            self._refresh_plot()
            self._schedule_plan()
            return
        if self._painting:
            mode = self._read_choice("interaction_mode", default="goal_click")
            if mode in ("paint_heat", "erase_heat", "toggle_blocked"):
                self._apply_brush(x, y, mode, start_stroke=False)
                self._refresh_plot()
                self._schedule_plan()

    def _on_map_release(self, _event: Any) -> None:
        had_drag = self._drag_target is not None
        had_paint = self._painting
        self._drag_target = None
        self._painting = False
        self._brush_block_value = None
        self._last_paint_rc = None
        if had_drag or had_paint:
            self._schedule_plan(immediate=True)

    def _set_marker(self, marker: str, x: float, y: float) -> None:
        if marker == "start":
            self.map_state.start_xy = (x, y)
        else:
            self.map_state.goal_xy = (x, y)

    def _closest_marker(self, x: float, y: float) -> str | None:
        sx, sy = self.map_state.start_xy
        gx, gy = self.map_state.goal_xy
        threshold = max(0.28, 4.0 * self.map_state.resolution_m_per_cell)
        ds = math.hypot(x - sx, y - sy)
        dg = math.hypot(x - gx, y - gy)
        if ds <= threshold and ds <= dg:
            return "start"
        if dg <= threshold:
            return "goal"
        return None

    def _apply_brush(self, x: float, y: float, mode: str, start_stroke: bool) -> None:
        r, c = self._world_to_cell(x, y)
        if not start_stroke and self._last_paint_rc == (r, c):
            return
        self._last_paint_rc = (r, c)

        radius_m = max(0.01, self._read_float("brush_radius_m", default=0.35))
        radius_cells = max(1, int(round(radius_m / max(self.map_state.resolution_m_per_cell, 1e-6))))
        r0 = max(0, r - radius_cells)
        r1 = min(self.map_state.heat.shape[0], r + radius_cells + 1)
        c0 = max(0, c - radius_cells)
        c1 = min(self.map_state.heat.shape[1], c + radius_cells + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        dist = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
        mask = dist <= float(radius_cells)
        if not np.any(mask):
            return

        self._mark_environment_dirty()
        if mode == "paint_heat":
            delta = max(0.001, self._read_float("brush_heat_delta", default=1.0))
            falloff = 1.0 - (dist[mask] / max(float(radius_cells), 1.0))
            self.map_state.heat[r0:r1, c0:c1][mask] += delta * falloff
            np.maximum(self.map_state.heat, 0.05, out=self.map_state.heat)
        elif mode == "erase_heat":
            delta = max(0.001, self._read_float("brush_heat_delta", default=1.0))
            falloff = 1.0 - (dist[mask] / max(float(radius_cells), 1.0))
            self.map_state.heat[r0:r1, c0:c1][mask] -= delta * falloff
            np.maximum(self.map_state.heat, 0.05, out=self.map_state.heat)
        elif mode == "toggle_blocked":
            local = self.map_state.blocked_mask[r0:r1, c0:c1]
            if start_stroke or self._brush_block_value is None:
                self._brush_block_value = not bool(self.map_state.blocked_mask[r, c])
            local[mask] = bool(self._brush_block_value)

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        res = self.map_state.resolution_m_per_cell
        h, w = self.map_state.heat.shape
        c = int(round(x / res))
        r = int(round(y / res))
        c = max(0, min(w - 1, c))
        r = max(0, min(h - 1, r))
        return r, c

    def _clamp_world(self, x: float, y: float) -> tuple[float, float]:
        h, w = self.map_state.heat.shape
        res = self.map_state.resolution_m_per_cell
        x = max(0.0, min(float(x), (w - 1) * res))
        y = max(0.0, min(float(y), (h - 1) * res))
        return x, y

    def _on_scenario_selected(self, _event: Any) -> None:
        name = self._read_choice("scenario_name", default=self.map_state.scenario_name)
        if name in self._scenarios:
            self._load_scenario(name, apply_scenario_endpoint_defaults=True)

    def _load_scenario(self, name: str, apply_scenario_endpoint_defaults: bool) -> None:
        scenario = self._scenarios[name]
        self.map_state = self._scenario_to_map_state(scenario)
        self._runtime.reset()
        self._mark_environment_dirty()
        self._artifacts = None
        self._last_error = None
        if apply_scenario_endpoint_defaults:
            with self._suspended_replan():
                self.vars["scenario_name"].set(name)
                self.vars["start_heading_deg"].set(_fmt(scenario.start_heading_deg))
                self.vars["goal_heading_deg"].set(_fmt(scenario.end_heading_deg))
                self.vars["start_approach_heading_deg"].set(_fmt(scenario.start_approach_heading_deg))
                self.vars["goal_approach_heading_deg"].set(_fmt(scenario.goal_approach_heading_deg))
                self.vars["start_approach_lock_distance_m"].set(_fmt(scenario.start_approach_lock_distance_m))
                self.vars["goal_approach_lock_distance_m"].set(_fmt(scenario.goal_approach_lock_distance_m))
        self.status_var.set(f"Loaded scenario: {name}")
        self._refresh_plot()
        self._schedule_plan(immediate=True)

    def _on_reset_map(self) -> None:
        scenario_name = self._read_choice("scenario_name", default=self.map_state.scenario_name)
        if scenario_name in self._scenarios:
            self._load_scenario(scenario_name, apply_scenario_endpoint_defaults=False)

    def _on_clear_blocked(self) -> None:
        self.map_state.blocked_mask.fill(False)
        self._mark_environment_dirty()
        self.status_var.set("Cleared blocked mask.")
        self._refresh_plot()
        self._schedule_plan(immediate=True)

    def _on_randomize_heat(self) -> None:
        try:
            seed = self._read_int("random_seed", default=0, minimum=0)
        except ValueError:
            seed = 0
        rng = np.random.default_rng(seed)
        h, w = self.map_state.heat.shape
        yy, xx = np.mgrid[0:h, 0:w]
        heat = np.full((h, w), rng.uniform(1.0, 4.0), dtype=float)
        island_count = max(8, int(0.05 * math.sqrt(h * w)))
        for _ in range(island_count):
            cx = rng.uniform(0.0, w - 1.0)
            cy = rng.uniform(0.0, h - 1.0)
            sigma = rng.uniform(0.04, 0.18) * min(h, w)
            amp = rng.uniform(-3.0, 16.0)
            heat += amp * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * sigma * sigma))
        heat += 0.4 * np.sin(xx / max(6.0, w / 18.0)) + 0.35 * np.cos(yy / max(6.0, h / 17.0))
        np.maximum(heat, 0.05, out=heat)
        self.map_state.heat = heat
        self._mark_environment_dirty()
        self.status_var.set(f"Randomized heat map (seed={seed}).")
        self._refresh_plot()
        self._schedule_plan(immediate=True)

    def _save_scenario_config(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save scenario config",
            defaultextension=".npz",
            filetypes=[("NumPy compressed", "*.npz"), ("All files", "*.*")],
            initialfile=f"{self.map_state.scenario_name}_scenario.npz",
        )
        if not path:
            return
        preset = self._collect_parameter_preset()
        np.savez_compressed(
            path,
            heat=self.map_state.heat,
            blocked=self.map_state.blocked_mask.astype(np.uint8),
            start=np.asarray(self.map_state.start_xy, dtype=float),
            goal=np.asarray(self.map_state.goal_xy, dtype=float),
            resolution=np.asarray([self.map_state.resolution_m_per_cell], dtype=float),
            scenario_name=np.asarray([self.map_state.scenario_name]),
            param_preset=np.asarray([json.dumps(preset)]),
        )
        self.status_var.set(f"Saved scenario config: {Path(path).name}")

    def _load_scenario_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load scenario config",
            filetypes=[("NumPy compressed", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        with np.load(path, allow_pickle=False) as data:
            heat = np.asarray(data["heat"], dtype=float)
            blocked = (
                np.asarray(data["blocked"], dtype=bool)
                if "blocked" in data
                else np.zeros_like(heat, dtype=bool)
            )
            start = np.asarray(data["start"], dtype=float)
            goal = np.asarray(data["goal"], dtype=float)
            resolution = float(data["resolution"][0]) if "resolution" in data else self.map_state.resolution_m_per_cell
            preset_json = str(data["param_preset"][0]) if "param_preset" in data else None

        self.map_state = VisualizerMapState(
            scenario_name=Path(path).stem,
            heat=heat.copy(),
            blocked_mask=blocked.copy(),
            start_xy=(float(start[0]), float(start[1])),
            goal_xy=(float(goal[0]), float(goal[1])),
            resolution_m_per_cell=float(resolution),
        )
        self._runtime.reset()
        self._mark_environment_dirty()
        self._artifacts = None
        self._last_error = None

        if preset_json:
            try:
                preset_payload = json.loads(preset_json)
                if isinstance(preset_payload, dict):
                    self._apply_parameter_preset_dict(preset_payload)
            except Exception:
                pass

        self.status_var.set(f"Loaded scenario config: {Path(path).name}")
        self._refresh_plot()
        self._schedule_plan(immediate=True)

    def _export_runtime_payload(self) -> None:
        if self._artifacts is None or self._current_cfg is None:
            self.status_var.set("No successful plan to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export runtime payload JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="runtime_payload.json",
        )
        if not path:
            return
        t0 = time.perf_counter()
        payload = build_runtime_payload_compact(
            bezier_segments=self._artifacts.bezier_segments,
            sampled_points=self._artifacts.sampled_smoothed_points,
            sampled_path_tangent_headings_rad=self._artifacts.sampled_path_tangent_headings_rad,
            sampled_holonomic_rotations_rad=self._artifacts.sampled_holonomic_rotations_rad,
            summary=self._artifacts.summary,
            required_clearance_m=self._artifacts.required_clearance_m,
            backend_status=self._artifacts.backend_status,
            cfg=self._current_cfg,
        )
        write_json(Path(path), payload, compact=False)
        self._last_export_ms = (time.perf_counter() - t0) * 1000.0
        self.status_var.set(f"Exported runtime payload: {Path(path).name}")
        self._refresh_diagnostics()

    def _export_concept_bundle(self) -> None:
        if self._artifacts is None or self._current_cfg is None:
            self.status_var.set("No successful plan to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export concept JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="pathplanner_concept.json",
        )
        if not path:
            return
        t0 = time.perf_counter()
        waypoints = beziers_to_pathplanner_waypoints(self._artifacts.bezier_segments, self._current_cfg)
        concept = build_concept_export(self._artifacts.bezier_segments, waypoints, self._current_cfg)
        out_path = Path(path)
        write_json(out_path, concept, compact=False)
        csv_path = out_path.with_name(f"{out_path.stem}_waypoints.csv")
        write_waypoints_csv(csv_path, waypoints)
        self._last_export_ms = (time.perf_counter() - t0) * 1000.0
        self.status_var.set(f"Exported concept JSON+CSV: {out_path.name}")
        self._refresh_diagnostics()

    def _save_screenshot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save screenshot",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            initialfile="planner_view.png",
        )
        if not path:
            return
        t0 = time.perf_counter()
        self.figure.savefig(path, dpi=180)
        self._last_export_ms = (time.perf_counter() - t0) * 1000.0
        self.status_var.set(f"Saved screenshot: {Path(path).name}")
        self._refresh_diagnostics()

    def _save_parameter_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save parameter preset",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="planner_preset.json",
        )
        if not path:
            return
        payload = self._collect_parameter_preset()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_var.set(f"Saved parameter preset: {Path(path).name}")

    def _load_parameter_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load parameter preset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            self.status_var.set("Preset file is invalid.")
            return
        self._apply_parameter_preset_dict(payload)
        self.status_var.set(f"Loaded parameter preset: {Path(path).name}")
        self._schedule_plan(immediate=True)

    def _collect_parameter_preset(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, var in self.vars.items():
            if isinstance(var, tk.BooleanVar):
                payload[key] = bool(var.get())
            else:
                payload[key] = str(var.get())
        return payload

    def _apply_parameter_preset_dict(self, payload: dict[str, Any]) -> None:
        with self._suspended_replan():
            for key, value in payload.items():
                if key not in self.vars:
                    continue
                var = self.vars[key]
                if isinstance(var, tk.BooleanVar):
                    if isinstance(value, bool):
                        var.set(value)
                    elif isinstance(value, str):
                        var.set(value.strip().lower() in ("1", "true", "yes", "on"))
                    else:
                        var.set(bool(value))
                else:
                    var.set("" if value is None else str(value))

    def _mark_environment_dirty(self) -> None:
        self._environment_revision += 1

    def _quality_profile(self, quality: str) -> tuple[int, int, int, int]:
        if quality == "fast":
            return 3, 1, 3, 1
        if quality == "high":
            return 10, 6, 8, 3
        return 6, 3, 6, 2

    def _build_cfg_from_controls(self) -> PlannerConfig | None:
        try:
            runtime_fast = self._read_bool("runtime_fast", default=True)
            quality = self._read_choice("smoothing_quality", default="fast")
            max_refits, runtime_fast_refits, curvature_iters, c2_iters = self._quality_profile(quality)

            sample_ds = max(0.01, self._read_float("sample_ds_m", default=self._default_cfg.sample_ds_m))
            if quality == "fast":
                sample_ds = max(sample_ds, 0.09)
            elif quality == "high":
                sample_ds = min(sample_ds, 0.06)

            cfg = PlannerConfig(
                runtime_mode="runtime_fast" if runtime_fast else "debug_diagnostics",
                compute_backend=self._read_choice("compute_backend", default="cpu"),
                cache_goal_fields=self._read_bool("cache_goal_fields", default=True),
                max_goal_field_cache_entries=self._read_int("max_goal_cache_entries", default=16, minimum=1),
                resolution_m_per_cell=self.map_state.resolution_m_per_cell,
                planner_mode=self._read_choice("planner_mode", default="fmm"),
                cost_mode=self._read_choice("cost_mode", default="density"),
                base_cost=max(1e-6, self._read_float("base_cost", default=self._default_cfg.base_cost)),
                alpha=max(0.0, self._read_float("alpha", default=self._default_cfg.alpha)),
                epsilon=max(1e-9, self._read_float("epsilon", default=self._default_cfg.epsilon)),
                enable_smoothing=self._read_bool("enable_smoothing", default=True),
                sample_ds_m=sample_ds,
                spline_smoothing=max(0.0, self._read_float("fit_tension", default=self._default_cfg.spline_smoothing)),
                max_curvature=max(0.05, self._read_float("max_curvature", default=self._default_cfg.max_curvature)),
                max_endpoint_curvature=max(
                    0.05, self._read_float("max_endpoint_curvature", default=self._default_cfg.max_endpoint_curvature)
                ),
                endpoint_zone_m=max(0.05, self._read_float("endpoint_zone_m", default=self._default_cfg.endpoint_zone_m)),
                handle_clamp_ratio=float(
                    np.clip(self._read_float("handle_clamp_ratio", default=self._default_cfg.handle_clamp_ratio), 0.05, 0.95)
                ),
                min_handle_ratio=float(
                    np.clip(self._read_float("min_handle_ratio", default=self._default_cfg.min_handle_ratio), 0.01, 0.95)
                ),
                endpoint_heading_blend_power=max(
                    1.0,
                    self._read_float("endpoint_blend_strength", default=self._default_cfg.endpoint_heading_blend_power),
                ),
                start_approach_lock_distance_m=max(
                    0.0,
                    self._read_float(
                        "start_approach_lock_distance_m",
                        default=self._default_cfg.start_approach_lock_distance_m,
                    ),
                ),
                goal_approach_lock_distance_m=max(
                    0.0,
                    self._read_float(
                        "goal_approach_lock_distance_m",
                        default=self._default_cfg.goal_approach_lock_distance_m,
                    ),
                ),
                start_approach_heading_deg=self._read_float("start_approach_heading_deg", allow_none=True),
                goal_approach_heading_deg=self._read_float("goal_approach_heading_deg", allow_none=True),
                terminal_progress_window_m=max(
                    0.01,
                    self._read_float(
                        "terminal_progress_window_m",
                        default=self._default_cfg.terminal_progress_window_m,
                    ),
                ),
                allow_terminal_overshoot=self._read_bool("allow_terminal_overshoot", default=False),
                start_heading_deg=self._read_float("start_heading_deg", allow_none=True),
                end_heading_deg=self._read_float("goal_heading_deg", allow_none=True),
                start_velocity_mps=max(
                    0.0, self._read_float("start_velocity_mps", default=self._default_cfg.start_velocity_mps)
                ),
                end_velocity_mps=max(0.0, self._read_float("end_velocity_mps", default=self._default_cfg.end_velocity_mps)),
                rotation_finish_progress=float(
                    np.clip(
                        self._read_float(
                            "rotation_finish_progress",
                            default=self._default_cfg.rotation_finish_progress,
                        ),
                        0.0,
                        1.0,
                    )
                ),
                holonomic_rotation_mode=self._read_choice(
                    "holonomic_rotation_mode",
                    default=self._default_cfg.holonomic_rotation_mode,
                ),
                object_width_m=max(0.05, self._read_float("object_width_m", default=self._default_cfg.object_width_m)),
                object_height_m=max(0.05, self._read_float("object_height_m", default=self._default_cfg.object_height_m)),
                object_shape=self._read_choice("object_shape", default=self._default_cfg.object_shape),
                safe_space_m=max(0.0, self._read_float("safe_space_m", default=self._default_cfg.safe_space_m)),
                enable_clearance_constraints=self._read_bool("enable_clearance_constraints", default=True),
                enforce_hard_clearance_if_feasible=self._read_bool("enforce_hard_clearance_if_feasible", default=True),
                heat_region_clearance_enabled=self._read_bool("heat_region_clearance_enabled", default=True),
                heat_region_threshold=self._read_float("heat_region_threshold", allow_none=True),
                max_smoothing_refits=max_refits,
                runtime_fast_max_refits=runtime_fast_refits,
                curvature_iters=curvature_iters,
                c2_regularization_iters=c2_iters,
            )
            return cfg
        except ValueError as exc:
            self.status_var.set(f"Invalid control value: {exc}")
            return None

    def _read_bool(self, key: str, default: bool) -> bool:
        var = self.vars.get(key)
        if isinstance(var, tk.BooleanVar):
            return bool(var.get())
        return bool(default)

    def _read_choice(self, key: str, default: str) -> str:
        var = self.vars.get(key)
        if isinstance(var, tk.StringVar):
            value = str(var.get()).strip()
            if value:
                return value
        return default

    def _read_float(
        self,
        key: str,
        *,
        default: float | None = None,
        allow_none: bool = False,
    ) -> float | None:
        var = self.vars.get(key)
        if not isinstance(var, tk.StringVar):
            if allow_none:
                return None
            if default is None:
                raise ValueError(f"{key} is missing.")
            return float(default)
        raw = str(var.get()).strip()
        if raw == "":
            if allow_none:
                return None
            if default is None:
                raise ValueError(f"{key} cannot be empty.")
            return float(default)
        if allow_none and raw.lower() in ("none", "null"):
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric.") from exc

    def _read_int(self, key: str, *, default: int, minimum: int) -> int:
        var = self.vars.get(key)
        if not isinstance(var, tk.StringVar):
            return max(minimum, int(default))
        raw = str(var.get()).strip()
        if raw == "":
            return max(minimum, int(default))
        try:
            value = int(float(raw))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        return max(minimum, value)

    def run(self) -> None:
        self.root.mainloop()


def launch_interactive_visualizer(initial_scenario: str = "hot_island") -> int:
    app = InteractivePlannerVisualizer(initial_scenario=initial_scenario)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_interactive_visualizer())
