from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _make_hot_blob_field(size: int = 160, peak: float = 70.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = int(0.35 * size)
    cy = int(0.62 * size)
    sigma = max(2.0, 0.045 * size)
    blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * sigma * sigma))
    return np.asarray(1.0 + peak * blob, dtype=float)


def _count_alternating_sharp_turn_pairs(points: np.ndarray, sharp_deg: float = 8.0) -> int:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return 0
    seg = np.diff(pts, axis=0)
    norms = np.linalg.norm(seg, axis=1)
    turns: list[float] = []
    for i in range(1, len(seg)):
        if norms[i - 1] <= 1e-12 or norms[i] <= 1e-12:
            turns.append(0.0)
            continue
        a = seg[i - 1] / norms[i - 1]
        b = seg[i] / norms[i]
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(dot)))
        cross = float(seg[i - 1, 0] * seg[i, 1] - seg[i - 1, 1] * seg[i, 0])
        if cross > 1e-12:
            turns.append(ang)
        elif cross < -1e-12:
            turns.append(-ang)
        else:
            turns.append(0.0)
    if len(turns) < 2:
        return 0
    out = 0
    for i in range(1, len(turns)):
        if abs(turns[i - 1]) < sharp_deg or abs(turns[i]) < sharp_deg:
            continue
        if turns[i - 1] * turns[i] < 0.0:
            out += 1
    return int(out)


def test_heat_chord_guard_rejects_hot_shortcuts() -> None:
    _bootstrap_src_path()
    from path_planner.config import PlannerConfig
    from path_planner.smooth import (
        SmoothingContext,
        _apply_heat_chord_guard,
        _integrate_cost_along_polyline,
    )

    cfg = PlannerConfig(resolution_m_per_cell=0.1, sample_ds_m=0.08)
    cost = _make_hot_blob_field(size=30, peak=220.0)
    wall_clear = np.full_like(cost, 5.0, dtype=float)
    heat_clear = np.full_like(cost, 5.0, dtype=float)

    context = SmoothingContext(
        start_xy=np.array([0.0, 0.0], dtype=float),
        goal_xy=np.array([1.0, 1.0], dtype=float),
        wall_clearance_field=wall_clear,
        heat_region_clearance_field=heat_clear,
        cost_density_field=cost,
        required_clearance_m=0.0,
        hard_clearance_feasible=True,
    )

    # Raw L-shape goes around the hot center while a direct chord cuts through it.
    raw_pts = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.2],
            [1.2, 1.2],
        ],
        dtype=float,
    )
    raw_u = np.asarray([0.0, 0.5, 1.0], dtype=float)
    chord_pts = np.asarray([[0.0, 0.0], [1.2, 1.2]], dtype=float)
    chord_u = np.asarray([0.0, 1.0], dtype=float)

    guarded_pts, guarded_u, rejected = _apply_heat_chord_guard(
        candidate_points=chord_pts,
        candidate_u=chord_u,
        raw_points=raw_pts,
        raw_u=raw_u,
        cfg=cfg,
        context=context,
    )

    assert rejected >= 1
    assert len(guarded_pts) >= 3
    assert np.all(np.diff(guarded_u) >= -1e-9)
    chord_cost = _integrate_cost_along_polyline(chord_pts, cost, cfg.resolution_m_per_cell)
    guarded_cost = _integrate_cost_along_polyline(guarded_pts, cost, cfg.resolution_m_per_cell)
    assert guarded_cost <= chord_cost


def test_terminal_dense_fallback_is_dense_and_non_degrading() -> None:
    _bootstrap_src_path()
    from path_planner.config import PlannerConfig
    from path_planner.smooth import SmoothingContext, smooth_path_to_beziers

    cfg = PlannerConfig(
        resolution_m_per_cell=0.1,
        sample_ds_m=0.08,
        # Force terminal guard rejection in smoothing attempts so fallback path is used.
        min_terminal_progress_ratio=1.02,
        max_smoothing_refits=2,
    )
    cost = _make_hot_blob_field(size=180, peak=120.0)
    wall_clear = np.full_like(cost, 6.0, dtype=float)
    heat_clear = np.full_like(cost, 4.0, dtype=float)

    raw_path = [
        (1.0, 1.0),
        (1.0, 7.0),
        (1.0, 12.0),
        (8.0, 12.0),
        (12.0, 12.0),
    ]
    context = SmoothingContext(
        start_xy=np.asarray(raw_path[0], dtype=float),
        goal_xy=np.asarray(raw_path[-1], dtype=float),
        wall_clearance_field=wall_clear,
        heat_region_clearance_field=heat_clear,
        cost_density_field=cost,
        required_clearance_m=0.0,
        hard_clearance_feasible=True,
    )

    result = smooth_path_to_beziers(raw_path, cfg, context=context)
    diag = result.diagnostics
    dense_fallback = diag.get("terminalSafeDenseFallbackDiagnostics", {})

    assert bool(diag.get("terminalSafeDenseFallbackUsed", False))
    assert isinstance(dense_fallback, dict)
    assert dense_fallback.get("fallbackReason", "") == "terminal_degradation_guard_triggered_no_accepted_candidate"

    raw_heat = float(dense_fallback.get("rawIntegratedHeatCost", 0.0))
    final_heat = float(dense_fallback.get("finalIntegratedHeatCost", 0.0))
    assert final_heat <= raw_heat * 1.001 + 1e-4

    raw_wall = float(dense_fallback.get("rawMinWallClearanceM", 0.0))
    final_wall = float(dense_fallback.get("finalMinWallClearanceM", 0.0))
    assert final_wall >= raw_wall - 1e-6

    raw_heat_clear = float(dense_fallback.get("rawMinHeatRegionClearanceM", -1.0))
    final_heat_clear = float(dense_fallback.get("finalMinHeatRegionClearanceM", -1.0))
    if raw_heat_clear >= 0.0 and final_heat_clear >= 0.0:
        assert final_heat_clear >= raw_heat_clear - 1e-6

    anchors = np.asarray(result.anchors, dtype=float)
    seg_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    assert seg_lengths.size > 0
    assert float(np.max(seg_lengths)) <= 0.08


def test_terminal_dense_fallback_averages_alternating_sharp_turn_artifacts() -> None:
    _bootstrap_src_path()
    from path_planner.config import PlannerConfig
    from path_planner.smooth import (
        SmoothingContext,
        _terminal_safe_dense_fallback_bezier_chain,
    )

    cfg = PlannerConfig(
        resolution_m_per_cell=0.1,
        sample_ds_m=0.05,
    )

    size = 80
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((xx - 40) ** 2) + ((yy - 40) ** 2)) / (2.0 * 6.5**2))
    cost = np.asarray(1.0 + 260.0 * blob, dtype=float)
    context = SmoothingContext(
        start_xy=np.array([0.0, 0.0], dtype=float),
        goal_xy=np.array([0.0, 0.0], dtype=float),
        wall_clearance_field=np.full_like(cost, 6.0, dtype=float),
        heat_region_clearance_field=np.full_like(cost, 4.0, dtype=float),
        cost_density_field=cost,
        required_clearance_m=0.0,
        hard_clearance_feasible=True,
    )

    raw_points = np.asarray(
        [
            [0.2, 0.2],
            [0.2, 1.0],
            [0.2, 1.8],
            [0.2, 2.6],
            [0.6, 3.0],
            [1.2, 3.2],
            [1.8, 3.2],
            [2.4, 3.2],
            [3.0, 3.2],
            [3.4, 3.4],
            [3.6, 4.0],
            [3.6, 4.8],
            [3.6, 5.6],
            [3.6, 6.4],
            [4.0, 6.8],
            [4.8, 7.0],
            [5.6, 7.0],
        ],
        dtype=float,
    )

    segments, anchors, _, diag = _terminal_safe_dense_fallback_bezier_chain(
        raw_points,
        cfg,
        context=context,
        trigger_reasons=["terminal_progress_ratio_low"],
    )
    final_pairs = _count_alternating_sharp_turn_pairs(np.asarray(anchors, dtype=float), sharp_deg=8.0)

    assert len(segments) > 0
    assert bool(diag.get("zigzagAveragingApplied", False))
    assert int(diag.get("zigzagAveragingBeforePairs", 0)) >= 2
    assert int(diag.get("zigzagAveragingAfterPairs", 99)) < int(diag.get("zigzagAveragingBeforePairs", 0))
    assert int(diag.get("finalAlternatingSharpTurnPairs", 99)) <= int(diag.get("zigzagAveragingAfterPairs", 0))
    assert int(diag.get("finalAlternatingSharpTurnPairs", 99)) == final_pairs
    assert final_pairs <= 1
