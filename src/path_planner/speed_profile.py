from __future__ import annotations

import math

import numpy as np


def compute_speed_profile(
    points: np.ndarray,
    curvatures: np.ndarray,
    max_speed_mps: float,
    max_accel_mps2: float,
    max_centripetal_accel_mps2: float,
    end_velocity_mps: float,
) -> list[dict[str, float]]:
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [{"s": 0.0, "v": max(0.0, float(end_velocity_mps))}]

    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.zeros(len(points), dtype=float)
    s[1:] = np.cumsum(ds)

    k = np.abs(curvatures)
    k = np.maximum(k, 1e-6)
    v_lim = np.sqrt(max_centripetal_accel_mps2 / k)
    v_lim = np.minimum(v_lim, max_speed_mps)
    v_lim = np.maximum(v_lim, 0.0)

    v = v_lim.copy()
    v[-1] = min(v[-1], max(0.0, end_velocity_mps))

    for i in range(1, len(v)):
        step = max(float(ds[i - 1]), 1e-9)
        vmax = math.sqrt(max(v[i - 1] ** 2 + 2.0 * max_accel_mps2 * step, 0.0))
        v[i] = min(v[i], vmax)

    for i in range(len(v) - 2, -1, -1):
        step = max(float(ds[i]), 1e-9)
        vmax = math.sqrt(max(v[i + 1] ** 2 + 2.0 * max_accel_mps2 * step, 0.0))
        v[i] = min(v[i], vmax)

    out: list[dict[str, float]] = []
    for ss, vv in zip(s, v):
        out.append({"s": float(ss), "v": float(vv)})
    return out

