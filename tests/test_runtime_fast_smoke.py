from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def test_runtime_fast_smoke() -> None:
    _bootstrap_src_path()
    from path_planner.config import PlannerConfig
    from path_planner.planner import run_planner
    from path_planner.runtime import PlannerRuntime
    from path_planner.scenarios import get_scenarios

    scenario = get_scenarios()["hot_island"]
    cfg = PlannerConfig(
        runtime_mode="runtime_fast",
        planner_mode="fmm",
        resolution_m_per_cell=scenario.resolution_m_per_cell,
        start_heading_deg=scenario.start_heading_deg,
        end_heading_deg=scenario.end_heading_deg,
        start_approach_heading_deg=scenario.start_approach_heading_deg,
        goal_approach_heading_deg=scenario.goal_approach_heading_deg,
    )
    runtime = PlannerRuntime(max_goal_cache_entries=8)
    art = run_planner(
        heat=scenario.heat,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        blocked_mask=scenario.blocked_mask,
        cfg=cfg,
        runtime=runtime,
        environment_id="runtime_fast_smoke",
    )
    assert art.summary["pathExists"] is True
    timing = art.summary.get("timingMs", {})
    assert float(timing.get("totalRuntimeMs", 0.0)) > 0.0
    assert "propagation" in timing
    assert "smoothing" in timing
    assert art.backend_status["used"] in ("cpu", "gpu")

