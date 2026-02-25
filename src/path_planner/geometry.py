from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .models import BezierSegment


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    s = np.zeros(len(points), dtype=float)
    if len(points) == 1:
        return s
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s[1:] = np.cumsum(ds)
    return s


def resample_polyline(points: np.ndarray, ds: float) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    if ds <= 0:
        return points.copy()

    s = cumulative_arc_length(points)
    total = float(s[-1])
    if total <= 1e-9:
        return np.array([points[0], points[-1]], dtype=float)

    targets = np.arange(0.0, total, ds, dtype=float)
    if targets.size == 0 or targets[-1] < total:
        targets = np.append(targets, total)

    out = np.empty((targets.size, 2), dtype=float)
    out[:, 0] = np.interp(targets, s, points[:, 0])
    out[:, 1] = np.interp(targets, s, points[:, 1])
    return out


def _rdp_recursive(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return points

    start = points[0]
    end = points[-1]
    seg = end - start
    seg_norm = np.linalg.norm(seg)

    if seg_norm <= 1e-12:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        rel = points - start
        proj = np.dot(rel, seg) / (seg_norm ** 2)
        proj_pts = start + proj[:, None] * seg
        dists = np.linalg.norm(points - proj_pts, axis=1)

    idx = int(np.argmax(dists))
    dmax = float(dists[idx])
    if dmax <= epsilon:
        return np.vstack([start, end])

    left = _rdp_recursive(points[: idx + 1], epsilon)
    right = _rdp_recursive(points[idx:], epsilon)
    return np.vstack([left[:-1], right])


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2 or epsilon <= 0:
        return points.copy()
    return _rdp_recursive(points, epsilon)


def dedupe_consecutive(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > tol:
            keep.append(i)
    return points[np.array(keep, dtype=int)]


def heading_to_unit(heading_deg: float) -> np.ndarray:
    theta = math.radians(heading_deg)
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def angle_deg(vec: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def angle_rad(vec: np.ndarray) -> float:
    return math.atan2(float(vec[1]), float(vec[0]))


def bezier_point(segment: BezierSegment, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        (u ** 3) * segment.p0
        + 3.0 * (u ** 2) * t * segment.p1
        + 3.0 * u * (t ** 2) * segment.p2
        + (t ** 3) * segment.p3
    )


def bezier_first_derivative(segment: BezierSegment, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        3.0 * (u ** 2) * (segment.p1 - segment.p0)
        + 6.0 * u * t * (segment.p2 - segment.p1)
        + 3.0 * (t ** 2) * (segment.p3 - segment.p2)
    )


def bezier_second_derivative(segment: BezierSegment, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        6.0 * u * (segment.p2 - 2.0 * segment.p1 + segment.p0)
        + 6.0 * t * (segment.p3 - 2.0 * segment.p2 + segment.p1)
    )


def curvature(segment: BezierSegment, t: float) -> float:
    d1 = bezier_first_derivative(segment, t)
    d2 = bezier_second_derivative(segment, t)
    num = abs(float(d1[0] * d2[1] - d1[1] * d2[0]))
    den = float(np.linalg.norm(d1) ** 3)
    if den <= 1e-12:
        return 0.0
    return num / den


def max_curvature(segment: BezierSegment, samples: int = 40) -> float:
    n = max(samples, 2)
    ts = np.linspace(0.0, 1.0, n, dtype=float)
    u = 1.0 - ts
    d1 = (
        3.0 * (u**2)[:, None] * (segment.p1 - segment.p0)[None, :]
        + 6.0 * (u * ts)[:, None] * (segment.p2 - segment.p1)[None, :]
        + 3.0 * (ts**2)[:, None] * (segment.p3 - segment.p2)[None, :]
    )
    d2 = (
        6.0 * u[:, None] * (segment.p2 - 2.0 * segment.p1 + segment.p0)[None, :]
        + 6.0 * ts[:, None] * (segment.p3 - 2.0 * segment.p2 + segment.p1)[None, :]
    )
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(np.linalg.norm(d1, axis=1), 3.0)
    vals = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(np.max(vals))


def sample_bezier_chain(
    segments: Iterable[BezierSegment],
    sample_per_segment: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    headings: list[float] = []
    curvatures: list[float] = []

    seg_list = list(segments)
    if not seg_list:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )

    n = max(2, int(sample_per_segment))
    base_ts = np.linspace(0.0, 1.0, n, dtype=float)
    for i, seg in enumerate(seg_list):
        ts = base_ts if i == 0 else base_ts[1:]
        u = 1.0 - ts
        p = (
            (u**3)[:, None] * seg.p0[None, :]
            + 3.0 * ((u**2) * ts)[:, None] * seg.p1[None, :]
            + 3.0 * (u * (ts**2))[:, None] * seg.p2[None, :]
            + (ts**3)[:, None] * seg.p3[None, :]
        )
        d1 = (
            3.0 * (u**2)[:, None] * (seg.p1 - seg.p0)[None, :]
            + 6.0 * (u * ts)[:, None] * (seg.p2 - seg.p1)[None, :]
            + 3.0 * (ts**2)[:, None] * (seg.p3 - seg.p2)[None, :]
        )
        d2 = (
            6.0 * u[:, None] * (seg.p2 - 2.0 * seg.p1 + seg.p0)[None, :]
            + 6.0 * ts[:, None] * (seg.p3 - 2.0 * seg.p2 + seg.p1)[None, :]
        )
        num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        den = np.power(np.linalg.norm(d1, axis=1), 3.0)
        curv = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        head = np.arctan2(d1[:, 1], d1[:, 0])
        points.extend(p)
        headings.extend(head.tolist())
        curvatures.extend(curv.tolist())

    return (
        np.asarray(points, dtype=float),
        np.asarray(headings, dtype=float),
        np.asarray(curvatures, dtype=float),
    )


def bilinear_sample_grid(arr: np.ndarray, x_idx: float, y_idx: float) -> float:
    h, w = arr.shape
    if x_idx < 0 or y_idx < 0 or x_idx > w - 1 or y_idx > h - 1:
        return float("nan")

    x0 = int(np.floor(x_idx))
    y0 = int(np.floor(y_idx))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x_idx - x0
    ty = y_idx - y0

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])

    if any(math.isnan(v) for v in (v00, v10, v01, v11)):
        return float("nan")

    top = (1.0 - tx) * v00 + tx * v10
    bot = (1.0 - tx) * v01 + tx * v11
    return (1.0 - ty) * top + ty * bot


def bilinear_sample_grid_vectorized(
    arr: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> np.ndarray:
    if x_idx.size == 0:
        return np.empty((0,), dtype=float)
    h, w = arr.shape
    x = np.asarray(x_idx, dtype=float)
    y = np.asarray(y_idx, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    inside = (x >= 0.0) & (y >= 0.0) & (x <= (w - 1)) & (y <= (h - 1))
    if not np.any(inside):
        return out

    xi = x[inside]
    yi = y[inside]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = xi - x0
    ty = yi - y0

    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]
    vals = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v10) + ty * ((1.0 - tx) * v01 + tx * v11)
    invalid = np.isnan(v00) | np.isnan(v10) | np.isnan(v01) | np.isnan(v11)
    vals = np.asarray(vals, dtype=float)
    vals[invalid] = np.nan
    out[inside] = vals
    return out


def world_to_grid(x_m: float, y_m: float, resolution_m: float) -> tuple[float, float]:
    return x_m / resolution_m, y_m / resolution_m


def grid_to_world(x_idx: float, y_idx: float, resolution_m: float) -> tuple[float, float]:
    return x_idx * resolution_m, y_idx * resolution_m
