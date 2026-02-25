from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    heat: np.ndarray
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    resolution_m_per_cell: float
    start_heading_deg: float | None = 0.0
    end_heading_deg: float | None = 0.0
    start_approach_heading_deg: float | None = None
    goal_approach_heading_deg: float | None = None
    start_approach_lock_distance_m: float = 0.5
    goal_approach_lock_distance_m: float = 0.5
    blocked_mask: np.ndarray | None = None


def make_hot_island_scenario() -> Scenario:
    h, w = 90, 140
    res = 0.09
    yy, xx = np.mgrid[0:h, 0:w]

    heat = np.full((h, w), 4.5, dtype=float)

    cx = 0.52 * w
    cy = 0.52 * h
    sx = 0.11 * w
    sy = 0.16 * h
    heat += 34.0 * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))

    corridor_top = 0.28 * h + 0.08 * h * np.sin((xx / w) * np.pi)
    corridor_bot = 0.73 * h - 0.06 * h * np.sin((xx / w) * np.pi)
    heat -= 3.2 * np.exp(-((yy - corridor_top) ** 2) / (2 * (0.05 * h) ** 2))
    heat -= 1.8 * np.exp(-((yy - corridor_bot) ** 2) / (2 * (0.05 * h) ** 2))

    heat = np.clip(heat, 0.35, None)

    start_xy = (8 * res, 0.50 * h * res)
    goal_xy = ((w - 9) * res, 0.50 * h * res)
    return Scenario(
        name="hot_island",
        description="Very hot center with cooler corridors around it.",
        heat=heat,
        start_xy=start_xy,
        goal_xy=goal_xy,
        resolution_m_per_cell=res,
        start_heading_deg=0.0,
        end_heading_deg=0.0,
        start_approach_heading_deg=0.0,
        goal_approach_heading_deg=0.0,
    )


def make_uniform_high_heat_scenario() -> Scenario:
    h, w = 80, 120
    res = 0.10
    yy, xx = np.mgrid[0:h, 0:w]

    heat = np.full((h, w), 13.0, dtype=float)
    heat += 0.4 * np.sin(xx / 18.0) + 0.3 * np.cos(yy / 15.0)
    heat = np.clip(heat, 10.0, None)

    start_xy = (6 * res, 0.35 * h * res)
    goal_xy = ((w - 8) * res, 0.64 * h * res)
    return Scenario(
        name="uniform_high",
        description="All cells are high heat but finite/traversable.",
        heat=heat,
        start_xy=start_xy,
        goal_xy=goal_xy,
        resolution_m_per_cell=res,
        start_heading_deg=8.0,
        end_heading_deg=-3.0,
        start_approach_heading_deg=8.0,
        goal_approach_heading_deg=-3.0,
    )


def make_blocked_gap_scenario() -> Scenario:
    h, w = 90, 140
    res = 0.09
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.full((h, w), 3.0, dtype=float)
    heat += 1.6 * np.exp(-((yy - 0.35 * h) ** 2) / (2 * (0.11 * h) ** 2))
    heat += 2.4 * np.exp(-((yy - 0.73 * h) ** 2) / (2 * (0.10 * h) ** 2))

    blocked = np.zeros((h, w), dtype=bool)
    wall_col = int(0.55 * w)
    blocked[:, wall_col] = True
    gap_low = int(0.43 * h)
    gap_high = int(0.58 * h)
    blocked[gap_low:gap_high, wall_col] = False

    start_xy = (9 * res, 0.50 * h * res)
    goal_xy = ((w - 10) * res, 0.50 * h * res)

    return Scenario(
        name="blocked_gap",
        description="Optional blocked-wall map with one legal gap.",
        heat=heat,
        start_xy=start_xy,
        goal_xy=goal_xy,
        resolution_m_per_cell=res,
        start_heading_deg=0.0,
        end_heading_deg=0.0,
        start_approach_heading_deg=0.0,
        goal_approach_heading_deg=0.0,
        blocked_mask=blocked,
    )


def make_double_hot_mid_corridor_scenario() -> Scenario:
    h, w = 96, 160
    res = 0.08
    yy, xx = np.mgrid[0:h, 0:w]

    heat = np.full((h, w), 3.6, dtype=float)
    heat += 0.20 * np.sin(xx / 19.0) + 0.18 * np.cos(yy / 16.0)

    sigma = 0.135 * min(h, w)
    cx1, cy1 = 0.39 * w, 0.50 * h
    cx2, cy2 = 0.61 * w, 0.50 * h
    heat += 30.0 * np.exp(-(((xx - cx1) ** 2) + ((yy - cy1) ** 2)) / (2 * sigma**2))
    heat += 30.0 * np.exp(-(((xx - cx2) ** 2) + ((yy - cy2) ** 2)) / (2 * sigma**2))

    mid_clear = np.exp(-((yy - 0.50 * h) ** 2) / (2 * (0.09 * h) ** 2))
    mid_clear *= np.exp(-((xx - 0.50 * w) ** 2) / (2 * (0.12 * w) ** 2))
    heat -= 4.2 * mid_clear
    heat = np.clip(heat, 0.35, None)

    start_xy = (6 * res, 0.50 * h * res)
    goal_xy = ((w - 7) * res, 0.50 * h * res)
    return Scenario(
        name="double_hot_mid_corridor",
        description=(
            "Start mid-left to mid-right with two large hot circles and a clear "
            "middle corridor between them."
        ),
        heat=heat,
        start_xy=start_xy,
        goal_xy=goal_xy,
        resolution_m_per_cell=res,
        start_heading_deg=0.0,
        end_heading_deg=145.0,
        start_approach_heading_deg=0.0,
        goal_approach_heading_deg=0.0,
    )


def make_small_islands_weave_scenario() -> Scenario:
    h, w = 104, 180
    res = 0.08
    yy, xx = np.mgrid[0:h, 0:w]

    heat = np.full((h, w), 3.1, dtype=float)
    heat += 0.24 * np.sin(xx / 21.0) + 0.20 * np.cos(yy / 15.0)

    # A chain of small hot islands with alternating vertical placement to force
    # repeated up/down turns, then a final rise toward the goal.
    islands: list[tuple[float, float, float, float]] = [
        (0.31, 0.61, 0.065, 19.5),
        (0.49, 0.46, 0.064, 20.0),

        (0.77, 0.66, 0.060, 18.5),
        (0.77, 0.26, 0.090, 18.5),
        (0.87, 0.56, 0.080, 18.5),
        (0.77, 0.16, 0.090, 18.5),
    ]
    scale = float(min(h, w))
    for x_frac, y_frac, sigma_frac, amp in islands:
        cx = x_frac * w
        cy = y_frac * h
        sigma = sigma_frac * scale
        heat += amp * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma**2))

    # Keep an overall rightward corridor so endpoint approach can stay clean.
    ribbon_mid = 0.50 * h - 0.10 * h * np.sin((xx / w) * 2.2 * np.pi)
    heat -= 1.2 * np.exp(-((yy - ribbon_mid) ** 2) / (2 * (0.075 * h) ** 2))
    heat = np.clip(heat, 0.35, None)

    start_xy = (7 * res, 0.48 * h * res)
    goal_xy = ((w - 8) * res, 0.34 * h * res)
    return Scenario(
        name="small_islands_weave",
        description=(
            "Multiple small hot islands arranged to induce repeated up/down "
            "curving with a final rise into the goal."
        ),
        heat=heat,
        start_xy=start_xy,
        goal_xy=goal_xy,
        resolution_m_per_cell=res,
        start_heading_deg=0.0,
        end_heading_deg=0.0,
        start_approach_heading_deg=0.0,
        goal_approach_heading_deg=0.0,
    )


def get_scenarios() -> dict[str, Scenario]:
    return {
        "hot_island": make_hot_island_scenario(),
        "uniform_high": make_uniform_high_heat_scenario(),
        "blocked_gap": make_blocked_gap_scenario(),
        "double_hot_mid_corridor": make_double_hot_mid_corridor_scenario(),
        "small_islands_weave": make_small_islands_weave_scenario(),
    }
