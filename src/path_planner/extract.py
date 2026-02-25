from __future__ import annotations

import math

import numpy as np

from .config import PlannerConfig
from .geometry import bilinear_sample_grid, grid_to_world, normalize, world_to_grid


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    ]


def _inside_grid(shape: tuple[int, int], x_idx: float, y_idx: float) -> bool:
    h, w = shape
    return 0.0 <= x_idx <= w - 1 and 0.0 <= y_idx <= h - 1


def _is_blocked(blocked: np.ndarray, x_idx: float, y_idx: float) -> bool:
    r = int(round(y_idx))
    c = int(round(x_idx))
    h, w = blocked.shape
    if r < 0 or c < 0 or r >= h or c >= w:
        return True
    return bool(blocked[r, c])


def _best_neighbor_step(
    t_field: np.ndarray,
    blocked: np.ndarray,
    x_idx: float,
    y_idx: float,
) -> tuple[float, float] | None:
    r = int(round(y_idx))
    c = int(round(x_idx))
    h, w = t_field.shape
    if r < 0 or c < 0 or r >= h or c >= w:
        return None

    t_here = float(t_field[r, c])
    best_val = t_here
    best: tuple[int, int] | None = None
    for rr, cc in _neighbors8(r, c):
        if rr < 0 or cc < 0 or rr >= h or cc >= w:
            continue
        if blocked[rr, cc]:
            continue
        cand = float(t_field[rr, cc])
        if cand < best_val - 1e-9:
            best_val = cand
            best = (rr, cc)

    if best is None:
        return None
    rr, cc = best
    return float(cc), float(rr)


def _heading_unit(heading_deg: float | None) -> np.ndarray | None:
    if heading_deg is None:
        return None
    theta = math.radians(float(heading_deg))
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def _lock_weight(distance: float, lock_distance: float) -> float:
    if lock_distance <= 1e-9:
        return 0.0
    t = 1.0 - distance / lock_distance
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def extract_path_from_cost_to_go(
    t_field: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    blocked: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    res = cfg.resolution_m_per_cell
    x_idx, y_idx = world_to_grid(start_xy[0], start_xy[1], res)
    gx_idx, gy_idx = world_to_grid(goal_xy[0], goal_xy[1], res)

    if not _inside_grid(t_field.shape, x_idx, y_idx):
        raise ValueError("Start is out of map.")
    if not _inside_grid(t_field.shape, gx_idx, gy_idx):
        raise ValueError("Goal is out of map.")
    if _is_blocked(blocked, x_idx, y_idx):
        raise ValueError("Start lies on a blocked cell.")
    if _is_blocked(blocked, gx_idx, gy_idx):
        raise ValueError("Goal lies on a blocked cell.")

    finite = t_field[np.isfinite(t_field)]
    if finite.size == 0:
        raise RuntimeError("Cost-to-go field is fully infinite.")

    t_work = np.asarray(t_field, dtype=float).copy()
    inf_fill = float(np.max(finite)) + 1e6
    t_work[~np.isfinite(t_work)] = inf_fill

    grad_y, grad_x = np.gradient(t_work, res, res)

    raw_world: list[tuple[float, float]] = []
    raw_cells: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()

    cur_x_idx = x_idx
    cur_y_idx = y_idx
    goal_tol_idx = cfg.goal_tolerance_m / res
    step_m = max(cfg.path_step_m, 1e-5)
    start_heading = _heading_unit(cfg.resolved_start_approach_heading_deg)
    goal_heading = _heading_unit(cfg.resolved_goal_approach_heading_deg)
    start_lock_m = max(0.0, float(cfg.start_approach_lock_distance_m))
    goal_lock_m = max(0.0, float(cfg.goal_approach_lock_distance_m))
    overshoot_tol_m = max(0.0, float(cfg.endpoint_overshoot_tolerance_m))

    for _ in range(cfg.max_extract_steps):
        x_m, y_m = grid_to_world(cur_x_idx, cur_y_idx, res)
        raw_world.append((x_m, y_m))
        raw_cells.append((int(round(cur_x_idx)), int(round(cur_y_idx))))

        if math.dist((cur_x_idx, cur_y_idx), (gx_idx, gy_idx)) <= goal_tol_idx:
            raw_world.append(goal_xy)
            raw_cells.append((int(round(gx_idx)), int(round(gy_idx))))
            return raw_world, raw_cells

        q = (
            int(round(cur_x_idx * cfg.loop_quantization)),
            int(round(cur_y_idx * cfg.loop_quantization)),
        )
        if q in visited:
            neighbor = _best_neighbor_step(t_work, blocked, cur_x_idx, cur_y_idx)
            if neighbor is None:
                break
            next_x_idx, next_y_idx = neighbor
        else:
            visited.add(q)

            t_here = bilinear_sample_grid(t_work, cur_x_idx, cur_y_idx)
            gx = bilinear_sample_grid(grad_x, cur_x_idx, cur_y_idx)
            gy = bilinear_sample_grid(grad_y, cur_x_idx, cur_y_idx)
            valid_grad = (
                np.isfinite(gx)
                and np.isfinite(gy)
                and math.hypot(gx, gy) > 1e-12
            )
            if not valid_grad:
                neighbor = _best_neighbor_step(t_work, blocked, cur_x_idx, cur_y_idx)
                if neighbor is None:
                    break
                next_x_idx, next_y_idx = neighbor
            else:
                norm = math.hypot(gx, gy)
                dir_x = -gx / norm
                dir_y = -gy / norm

                cur_world = np.array([x_m, y_m], dtype=float)
                if start_heading is not None and start_lock_m > 1e-9:
                    dist_start = float(np.linalg.norm(cur_world - np.asarray(start_xy, dtype=float)))
                    w_start = _lock_weight(dist_start, start_lock_m)
                    if w_start > 0.0:
                        blended = normalize(
                            (1.0 - w_start) * np.array([dir_x, dir_y], dtype=float)
                            + w_start * start_heading
                        )
                        if np.linalg.norm(blended) > 1e-12:
                            dir_x, dir_y = float(blended[0]), float(blended[1])

                if goal_heading is not None and goal_lock_m > 1e-9:
                    dist_goal = float(np.linalg.norm(np.asarray(goal_xy, dtype=float) - cur_world))
                    w_goal = _lock_weight(dist_goal, goal_lock_m)
                    if w_goal > 0.0:
                        to_goal = normalize(np.asarray(goal_xy, dtype=float) - cur_world)
                        preferred = normalize(0.35 * to_goal + 0.65 * goal_heading)
                        blended = normalize(
                            (1.0 - w_goal) * np.array([dir_x, dir_y], dtype=float) + w_goal * preferred
                        )
                        if np.linalg.norm(blended) > 1e-12:
                            dir_x, dir_y = float(blended[0]), float(blended[1])

                step_idx = step_m / res
                cand_x = cur_x_idx + dir_x * step_idx
                cand_y = cur_y_idx + dir_y * step_idx
                if (not _inside_grid(t_work.shape, cand_x, cand_y)) or _is_blocked(
                    blocked, cand_x, cand_y
                ):
                    neighbor = _best_neighbor_step(t_work, blocked, cur_x_idx, cur_y_idx)
                    if neighbor is None:
                        break
                    next_x_idx, next_y_idx = neighbor
                else:
                    t_next = bilinear_sample_grid(t_work, cand_x, cand_y)
                    cand_world = np.asarray(grid_to_world(cand_x, cand_y, res), dtype=float)
                    dist_goal_cur = float(
                        np.linalg.norm(np.asarray(goal_xy, dtype=float) - np.asarray([x_m, y_m], dtype=float))
                    )
                    dist_goal_next = float(np.linalg.norm(np.asarray(goal_xy, dtype=float) - cand_world))
                    terminal_lock = goal_heading is not None and goal_lock_m > 1e-9 and (
                        dist_goal_cur <= goal_lock_m + cfg.goal_tolerance_m
                    )
                    terminal_invalid = False
                    if terminal_lock:
                        cur_to_goal = np.asarray(goal_xy, dtype=float) - np.asarray([x_m, y_m], dtype=float)
                        nxt_to_goal = np.asarray(goal_xy, dtype=float) - cand_world
                        cur_proj = float(np.dot(cur_to_goal, goal_heading))
                        nxt_proj = float(np.dot(nxt_to_goal, goal_heading))
                        if (not cfg.allow_terminal_overshoot) and (nxt_proj < -overshoot_tol_m):
                            terminal_invalid = True
                        if nxt_proj > cur_proj + max(0.01, 0.15 * step_m):
                            terminal_invalid = True
                        if dist_goal_next > dist_goal_cur + max(0.01, 0.35 * step_m):
                            terminal_invalid = True

                    if (not np.isfinite(t_next)) or (t_next >= t_here - 1e-8) or terminal_invalid:
                        neighbor = _best_neighbor_step(t_work, blocked, cur_x_idx, cur_y_idx)
                        if neighbor is None:
                            break
                        next_x_idx, next_y_idx = neighbor
                    else:
                        next_x_idx, next_y_idx = cand_x, cand_y

        if math.dist((next_x_idx, next_y_idx), (cur_x_idx, cur_y_idx)) < 1e-8:
            break
        cur_x_idx, cur_y_idx = next_x_idx, next_y_idx

    raise RuntimeError(
        "Path extraction failed to reach goal. This usually means start and goal are disconnected by blocked cells."
    )
