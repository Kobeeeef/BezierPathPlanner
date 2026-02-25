# Heatmap FMM Path Planner (FRC / PathPlanner Concepts)

This prototype plans smooth robot paths on a **2D heat map** where:

- **Heat** = high cost but still traversable (finite values).
- **Obstacle** = blocked/untraversable (`blocked_mask` or `blocked_sentinel`).

It supports:

- True Fast Marching Method (Eikonal upwind) and Dijkstra approximation toggle.
- Gradient-descent raw path extraction from cost-to-go.
- Spline-first smoothing pipeline:
  - arc-length resampling
  - SciPy spline smoothing/fitting
  - cubic Bezier chain conversion with C1 join continuity
  - exact start/end position constraints
  - start/goal approach heading locks (`start_approach_heading`, `goal_approach_heading`)
  - endpoint blending/lock zones with automatic reject+refit on spikes/hooks
  - anti-hook checks near terminal zone (overshoot, monotonic approach, self-intersection)
- Separate clearance layers (not heat inflation):
  - wall/blocked geometric clearance field in meters
  - optional distance-to-high-heat-region field for spacing preference
  - footprint-aware required clearance: `footprint_radius + safe_space`
- Separate heading channels:
  - `pathTangentHeading*` for geometric path direction
  - `holonomicRotation*` for robot-facing direction (independent profile by default)
- Curvature-aware speed profile (supports goal stop with end velocity `0.0`).
- PathPlanner-concept waypoint export (anchors + control handles + end state).
- Plots and optional GIF animation.

## Quick Start

```powershell
python main.py --scenario all --planner fmm --export both
```

Outputs are written to `outputs/`.

## Scenarios

- `hot_island`: hot center + cooler corridors (path should route around hot zone).
- `uniform_high`: map is high heat everywhere but finite (path should still exist).
- `blocked_gap`: optional blocked wall with a single traversable gap.
- `double_hot_mid_corridor`: start at mid-left and goal at mid-right with two large hot circles and a clear middle corridor.

## Key CLI Flags

```powershell
python main.py `
  --scenario hot_island `
  --planner fmm `
  --alpha 4.0 `
  --end-heading 0 `
  --end-velocity 0 `
  --max-curvature 2.6
```

- `--planner`: `fmm` or `dijkstra_approx`
- `--cost-mode`: `density` or `inverse_speed`
- `--blocked-sentinel`: e.g. `nan` or numeric sentinel
- `--holonomic-rotation-mode`: `independent_profile` (default) or `tangent_follow`
- `--rotation-finish-progress`: fraction of path progress to finish rotating (default `0.8`)
- `--start-approach-heading`, `--goal-approach-heading`: tangent approach constraints (deg)
- `--start-approach-lock-distance-m`, `--goal-approach-lock-distance-m`: lock/blend lengths
- `--endpoint-zone-m`: endpoint blending/diagnostic distance (default `0.5`)
- `--max-endpoint-curvature`: refit threshold near endpoints
- `--object-width-m`, `--object-height-m`, `--object-shape`, `--safe-space-m`
- `--enforce-hard-clearance-if-feasible`: uses clearance feasibility as hard constraint when possible
- `--heat-region-clearance-enabled`: adds separate spacing preference from high-heat regions
- `--animation`: enable GIF generation (disabled by default)
- `--no-animation`: explicit disable (default behavior)

## Produced Artifacts

Per scenario:

- `heatmap.png`
- `cost_to_go.png`
- `path_overlay.png` (raw + smoothed)
- `curvature_overlay.png`
- `speed_overlay.png`
- `path_animation.gif` (only when `--animation` is set)
- `pathplanner_concept.json`
- `pathplanner_best_effort.path.json` (if export mode requests it)
- `pathplanner_waypoints.csv`
- `plan_summary.json`

`plan_summary.json` now includes smoothing quality diagnostics:

- max tangent jump at Bezier joins (deg)
- max/percentile curvature
- max curvature in first `0.5m` and last `0.5m`
- tangent jump near start/end zones
- endpoint approach alignment error (start/end)
- lock-zone heading error (weighted + raw)
- terminal anti-hook checks (overshoot + monotonic progress)
- self-intersection checks
- clearance stats (`requiredClearanceM`, min wall clearance, min heat-region clearance)
- segment length stats
- raw-vs-smoothed comparison metrics
- refit attempt log and selected attempt index
