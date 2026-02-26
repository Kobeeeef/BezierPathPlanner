from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def test_wall_clearance_is_always_hard_blocked() -> None:
    _bootstrap_src_path()
    from path_planner.clearance import clearance_layers_from_base, precompute_clearance_base
    from path_planner.config import PlannerConfig

    heat = np.zeros((7, 7), dtype=float)
    blocked = np.zeros_like(heat, dtype=bool)
    start_rc = (3, 1)
    goal_rc = (3, 5)
    probe_rc = (3, 3)

    cfg_small = PlannerConfig(
        resolution_m_per_cell=1.0,
        object_width_m=0.6,
        object_height_m=0.6,
        safe_space_m=0.0,
        enforce_hard_clearance_if_feasible=False,
    )
    cfg_large = cfg_small.with_updates(
        object_width_m=4.0,
        object_height_m=4.0,
        safe_space_m=1.0,
    )

    base_small = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg_small)
    base_large = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg_large)
    layers_small = clearance_layers_from_base(
        base=base_small,
        blocked=blocked,
        cfg=cfg_small,
        start_rc=start_rc,
        goal_rc=goal_rc,
    )
    layers_large = clearance_layers_from_base(
        base=base_large,
        blocked=blocked,
        cfg=cfg_large,
        start_rc=start_rc,
        goal_rc=goal_rc,
    )

    assert layers_large.required_clearance_m > layers_small.required_clearance_m
    assert bool(layers_small.planning_blocked[probe_rc]) is False
    assert bool(layers_large.hard_clearance_feasible) is False
    assert bool(layers_large.planning_blocked[probe_rc]) is True

