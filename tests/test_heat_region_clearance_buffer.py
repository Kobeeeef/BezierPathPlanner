from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def test_heat_region_clearance_uses_required_clearance_buffer() -> None:
    _bootstrap_src_path()
    from path_planner.clearance import precompute_clearance_base
    from path_planner.config import PlannerConfig

    heat = np.zeros((7, 7), dtype=float)
    heat[3, 3] = 10.0
    blocked = np.zeros_like(heat, dtype=bool)
    probe = (3, 6)  # 3m from the hot cell at 1m/cell resolution.

    cfg_small = PlannerConfig(
        resolution_m_per_cell=1.0,
        heat_region_threshold=5.0,
        heat_region_clearance_weight=1.0,
        heat_region_clearance_decay_m=1.0,
        object_width_m=0.6,
        object_height_m=0.6,
        safe_space_m=0.0,
    )
    cfg_large = cfg_small.with_updates(
        object_width_m=4.0,
        object_height_m=4.0,
        safe_space_m=1.0,
    )

    base_small = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg_small)
    base_large = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg_large)

    assert base_large.required_clearance_m > base_small.required_clearance_m
    assert np.array_equal(base_small.heat_region_mask, base_large.heat_region_mask)
    assert base_large.heat_region_clearance_m[probe] < base_small.heat_region_clearance_m[probe]
    assert base_large.heat_region_clearance_m[probe] == 0.0
    assert base_large.heat_region_penalty[probe] > base_small.heat_region_penalty[probe]
