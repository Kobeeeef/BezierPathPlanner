# Heatmap FMM Path Planner (FRC / PathPlanner Concepts)

<img width="815" height="413" alt="beziercurves" src="https://github.com/user-attachments/assets/9fb960c1-23cd-4730-a4b9-c3f60f7ca702" />

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
  - heat/clearance-aware terminal-zone optimization at both start and goal
  - optional start/goal approach heading hints (`start_approach_heading`, `goal_approach_heading`) with soft weighting
  - endpoint blending/lock zones with automatic reject+refit on spikes/hooks/degradation
  - anti-hook checks near terminal zone (overshoot, monotonic approach, self-intersection)
  - `geometry_mode=raw_preferred` default: raw path geometry is baseline and smoother output is rejected/fallback-to-raw if quality degrades
- Separate clearance layers (not heat inflation):
  - wall/blocked geometric clearance field in meters
  - optional distance-to-high-heat-region field for spacing preference
  - footprint-aware required clearance: `footprint_radius + safe_space`
- Separate heading channels:
  - `pathTangentHeading*` for geometric path direction
  - `holonomicRotation*` for robot-facing direction (independent profile by default)
  - holonomic rotation does not force terminal geometric curvature
- Curvature-aware speed profile (supports goal stop with end velocity `0.0`).
- PathPlanner-concept waypoint export (anchors + control handles + end state).
- Runtime engine for repeated replanning:
  - environment precompute/cache reuse
  - goal-centric cost-to-go field cache
  - reusable planner workspaces to reduce allocations
  - stage timing instrumentation (`propagation`, `extraction`, `smoothing`, `clearance_validation`, `rotation_profile_generation`, `exportSerializationMs`)
- Optional backend toggle (`cpu` / `gpu` with automatic CPU fallback).
- Plots and optional GIF animation.

## Quick Start

```powershell
python main.py --scenario all --planner fmm --export both
```

Outputs are written to `outputs/`.

For low-latency replanning mode (no plots/file writes by default):

```powershell
python main.py --scenario hot_island --mode runtime_fast --no-write-artifacts --no-compare --no-animation
```

Interactive visualizer (live click-to-plan tuning UI):

```powershell
python visualizer.py
```

Or from the existing CLI:

```powershell
python main.py --interactive-visualizer --scenario hot_island
```

## Scenarios

- `hot_island`: hot center + cooler corridors (path should route around hot zone).
- `uniform_high`: map is high heat everywhere but finite (path should still exist).
- `blocked_gap`: optional blocked wall with a single traversable gap.
- `double_hot_mid_corridor`: start at mid-left and goal at mid-right with two large hot circles and a clear middle corridor.
- `small_islands_weave`: many small hot islands that force repeated up/down turns, then a final rise into the goal.

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
- `--mode`: `runtime_fast` (low-latency, minimal overhead) or `debug_diagnostics` (full artifacts)
- `--geometry-mode`: `raw_only`, `raw_preferred` (default), `spline_then_bezier`, `bezier_optimized`
- `--compute-backend`: `cpu` (default) or `gpu` (fallbacks to CPU if unavailable)
- `--cache-goal-fields` / `--no-cache-goal-fields`: reuse propagation fields for repeated goals
- `--max-goal-cache-entries`: LRU cache size for goal fields
- `--write-artifacts` / `--no-write-artifacts`: file writes (disabled by default in `runtime_fast`)
- `--enable-plots` / `--no-plots`: plot/GIF generation
- `--cost-mode`: `density` or `inverse_speed`
- `--blocked-sentinel`: e.g. `nan` or numeric sentinel
- `--holonomic-rotation-mode`: `independent_profile` (default) or `tangent_follow`
- `--rotation-finish-progress`: fraction of path progress to finish rotating (default `0.8`)
- `--start-approach-heading`, `--goal-approach-heading`: tangent approach constraints (deg)
- `--start-approach-lock-distance-m`, `--goal-approach-lock-distance-m`: lock/blend lengths
- `--start-heading`, `--end-heading`: holonomic facing profile endpoints (independent from geometric tangent unless you also set approach headings)
- `--start-velocity`, `--end-velocity`: start/end speed targets used by speed profile generation
- `--endpoint-zone-m`: endpoint blending/diagnostic distance (default `0.5`)
- `--max-endpoint-curvature`: refit threshold near endpoints
- `--enable-smoothing` / `--no-smoothing`
- `--enable-clearance-constraints` / `--no-clearance-constraints`
- `--object-width-m`, `--object-height-m`, `--object-shape`, `--safe-space-m`
- `--enforce-hard-clearance-if-feasible`: uses clearance feasibility as hard constraint when possible
- `--heat-region-clearance-enabled`: adds separate spacing preference from high-heat regions
- `--animation`: enable GIF generation (disabled by default)
- `--no-animation`: explicit disable (default behavior)
- `--interactive-visualizer`: launch the interactive UI instead of batch CLI execution

## Interactive Visualizer

The visualizer is a separate module intended for rapid experimentation and uses
`runtime_fast` by default for responsiveness.

Main capabilities:

- click-to-plan updates (goal-click default, optional start/goal click mode, marker drag)
- live planner/smoothing/endpoint/clearance/runtime control panel
- overlays for heat, cost-to-go, clearance fields, raw path, smoothed Bezier path,
  control handles, sampled points, tangent/holonomic arrows, endpoint approach arrows
- live diagnostics panel with stage timings and quality metrics
- quality guard messages for failures/refits/overshoot-hook issues
- map editing tools (paint/erase heat, blocked brush toggle, randomize, reset)
- export buttons for runtime payload JSON, concept JSON+CSV, screenshot, and presets

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

In `runtime_fast` + `--write-artifacts`, planner writes:

- `runtime_payload.json` (compact runtime payload, no debug visuals by default)

`plan_summary.json` now includes smoothing quality diagnostics:

- max tangent jump at Bezier joins (deg)
- max/percentile curvature
- max curvature in first `0.5m` and last `0.5m`
- tangent jump near start/end zones
- endpoint approach alignment error (start/end)
- lock-zone heading error (weighted + raw)
- terminal anti-hook checks (overshoot + monotonic progress)
- terminal raw-vs-smoothed heat exposure, clearance, directness, curvature, and hook flags (start + goal)
- geometry decision report (accepted final / fallback to raw / raw-only), reasons, and final geometry source
- runtime-friendly raw path stream (`rawPathWorldResampled` / `rawSampledPath`) for direct follower integration
- self-intersection checks
- clearance stats (`requiredClearanceM`, min wall clearance, min heat-region clearance)
- segment length stats
- raw-vs-smoothed comparison metrics
- refit attempt log and selected attempt index
- stage timing summary + backend/cache stats

## Benchmark Suite

Target-validation loop (runtime-fast only, strict `<200 ms` average gate):

```powershell
.venv\Scripts\python.exe tests/benchmark_performance.py `
  --benchmark-kind target_loop `
  --planner-mode fmm `
  --replans-per-case 120 `
  --target-avg-ms 200 `
  --max-optimization-iterations 4 `
  --strict
```

This mode:

- measures per-run end-to-end latency from planning request to in-memory runtime payload + JSON serialization
- runs spammable replanning scenarios:
  - same map, same start/goal, tight loop
  - same map, changing start/goal
  - same start/goal, changing heatmap
  - changing blocked mask
- iteratively reruns with progressively faster runtime profiles until average latency passes the target or iteration limit is reached
- reports bottlenecks and next optimization recommendations if target is not met

General benchmark suite (full coverage):

```powershell
.venv\Scripts\python.exe tests/benchmark_performance.py `
  --benchmark-kind suite `
  --replans-per-case 40 `
  --mode runtime_fast `
  --planner-mode fmm
```

Outputs:

- `outputs/benchmarks/benchmark_results.json`
- `outputs/benchmarks/benchmark_results.csv`
- terminal summary table with avg/median/p95/p99/worst/throughput, quality fail ratio, memory growth

Benchmark coverage:

- end-to-end latency and stage-by-stage timing
- repeated rapid replanning (warm/cold cache)
- same map varying start/goal
- changing heat map / changing blocked mask
- scenario stability across different heat distributions
- runtime vs resolution scaling
- with/without smoothing
- with/without clearance constraints
- CPU vs GPU backend comparison (with fallback reporting)
