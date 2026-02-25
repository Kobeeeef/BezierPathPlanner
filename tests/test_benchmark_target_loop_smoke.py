from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def test_target_loop_smoke() -> None:
    _bootstrap_src_path()
    from path_planner.benchmarking import run_timing_target_validation_loop

    payload = run_timing_target_validation_loop(
        target_avg_ms=200.0,
        replans_per_case=6,
        planner_mode="fmm",
        max_optimization_iterations=1,
        quality_check_stride=1,
        determinism_sample_count=3,
        deterministic_seed=0,
    )

    assert payload["validationType"] == "timing_target_loop"
    assert float(payload["overallLatencyStats"]["count"]) > 0.0
    assert "summaryTable" in payload
    assert "cases" in payload
