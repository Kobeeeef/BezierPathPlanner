from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from .models import BezierSegment


def _extent(shape: tuple[int, int], resolution_m: float) -> list[float]:
    h, w = shape
    return [0.0, w * resolution_m, 0.0, h * resolution_m]


def _draw_heat_background(
    ax: plt.Axes,
    heat: np.ndarray,
    resolution_m: float,
    blocked: np.ndarray | None = None,
) -> None:
    im = ax.imshow(
        heat,
        origin="lower",
        cmap="viridis",
        extent=_extent(heat.shape, resolution_m),
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Heat (cost bias)")
    if blocked is not None and np.any(blocked):
        mask = np.ma.masked_where(~blocked, blocked.astype(float))
        ax.imshow(
            mask,
            origin="lower",
            cmap="gray",
            alpha=0.45,
            extent=_extent(heat.shape, resolution_m),
            aspect="equal",
        )


def save_heatmap_plot(
    out_path: Path,
    heat: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    resolution_m: float,
    blocked: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_heat_background(ax, heat, resolution_m, blocked)
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=90, marker="o", edgecolors="black", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=90, marker="X", edgecolors="black", label="Goal")
    ax.set_title("Heat Map (low is good, high is bad)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_cost_to_go_plot(
    out_path: Path,
    t_field: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    resolution_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    data = np.asarray(t_field, dtype=float).copy()
    finite = np.isfinite(data)
    if np.any(finite):
        high = float(np.nanmax(data[finite]))
        data[~finite] = high * 1.05
    im = ax.imshow(
        data,
        origin="lower",
        cmap="magma",
        extent=_extent(data.shape, resolution_m),
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cost-to-go / arrival time")
    ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", s=80, marker="o", edgecolors="black", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="white", s=85, marker="X", edgecolors="black", label="Goal")
    ax.set_title("Propagation Result (Cost-to-Go Field)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_bezier_segments(ax: plt.Axes, segments: list[BezierSegment], color: str, label: str) -> None:
    if not segments:
        return
    for i, seg in enumerate(segments):
        t = np.linspace(0.0, 1.0, 120)
        pts = (
            ((1 - t) ** 3)[:, None] * seg.p0
            + (3 * (1 - t) ** 2 * t)[:, None] * seg.p1
            + (3 * (1 - t) * t**2)[:, None] * seg.p2
            + (t**3)[:, None] * seg.p3
        )
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2, label=label if i == 0 else None)
        ax.plot([seg.p0[0], seg.p1[0]], [seg.p0[1], seg.p1[1]], color=color, alpha=0.18, linewidth=1.0)
        ax.plot([seg.p2[0], seg.p3[0]], [seg.p2[1], seg.p3[1]], color=color, alpha=0.18, linewidth=1.0)


def save_path_overlay_plot(
    out_path: Path,
    heat: np.ndarray,
    resolution_m: float,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    raw_path_world: list[tuple[float, float]],
    bezier_segments: list[BezierSegment],
    blocked: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_heat_background(ax, heat, resolution_m, blocked)

    if raw_path_world:
        raw = np.asarray(raw_path_world, dtype=float)
        ax.plot(raw[:, 0], raw[:, 1], color="#ff7f0e", linewidth=1.8, label="Raw extracted path")

    _draw_bezier_segments(ax, bezier_segments, color="#00e676", label="Smoothed Bezier path")
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=95, marker="o", edgecolors="black", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=95, marker="X", edgecolors="black", label="Goal")
    ax.set_title("Raw Path + Smoothed Bezier Overlay")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_scalar_overlay_plot(
    out_path: Path,
    heat: np.ndarray,
    resolution_m: float,
    points: np.ndarray,
    scalar: np.ndarray,
    label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_heat_background(ax, heat, resolution_m, None)
    if len(points) >= 2:
        segs = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap="plasma", linewidths=2.5)
        lc.set_array(scalar[:-1] if len(scalar) == len(points) else scalar)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, heat.shape[1] * resolution_m)
    ax.set_ylim(0, heat.shape[0] * resolution_m)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_comparison_plot(
    out_path: Path,
    heat: np.ndarray,
    resolution_m: float,
    shortest_path: list[tuple[float, float]] | None,
    low_heat_path: list[tuple[float, float]] | None,
    smoothed_points: np.ndarray | None,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_heat_background(ax, heat, resolution_m, None)
    if shortest_path:
        p = np.asarray(shortest_path, dtype=float)
        ax.plot(p[:, 0], p[:, 1], color="#3f51b5", linewidth=1.8, label="Raw shortest-like path")
    if low_heat_path:
        p = np.asarray(low_heat_path, dtype=float)
        ax.plot(p[:, 0], p[:, 1], color="#ff9800", linewidth=2.0, label="Low-heat raw path")
    if smoothed_points is not None and len(smoothed_points) > 1:
        ax.plot(
            smoothed_points[:, 0],
            smoothed_points[:, 1],
            color="#00e676",
            linewidth=2.5,
            label="Smoothed Bezier path",
        )
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=90, edgecolors="black", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=90, marker="X", edgecolors="black", label="Goal")
    ax.set_title("Path Comparison: shortest-like vs low-heat vs smoothed")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_path_animation_gif(
    out_path: Path,
    heat: np.ndarray,
    resolution_m: float,
    sampled_points: np.ndarray,
    robot_rotations_rad: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> str | None:
    if len(sampled_points) < 2:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    _draw_heat_background(ax, heat, resolution_m, None)
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=85, edgecolors="black", label="Start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=85, marker="X", edgecolors="black", label="Goal")
    ax.plot(sampled_points[:, 0], sampled_points[:, 1], color="white", alpha=0.5, linewidth=1.2)
    ax.set_title("Robot Traversal Animation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    robot_dot = ax.scatter(
        [sampled_points[0, 0]],
        [sampled_points[0, 1]],
        c="#03a9f4",
        s=100,
        edgecolors="black",
        zorder=5,
    )
    heading_line, = ax.plot([], [], color="#03a9f4", linewidth=2.2, zorder=4)

    arrow_len = 0.35

    def update(frame: int):
        x = sampled_points[frame, 0]
        y = sampled_points[frame, 1]
        theta = float(robot_rotations_rad[min(frame, len(robot_rotations_rad) - 1)])
        robot_dot.set_offsets(np.array([[x, y]]))
        hx = x + arrow_len * np.cos(theta)
        hy = y + arrow_len * np.sin(theta)
        heading_line.set_data([x, hx], [y, hy])
        return robot_dot, heading_line

    anim = FuncAnimation(fig, update, frames=len(sampled_points), interval=40, blit=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_path, writer=PillowWriter(fps=25))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return str(out_path)
