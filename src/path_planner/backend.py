from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np


ComputeBackend = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class BackendStatus:
    requested: ComputeBackend
    used: ComputeBackend
    reason: str


@lru_cache(maxsize=1)
def _cupy_handle() -> object | None:
    try:
        import cupy as cp  # type: ignore

        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return None
        _ = cp.zeros((1,), dtype=cp.float32)
        return cp
    except Exception:
        return None


def resolve_backend(requested: ComputeBackend) -> BackendStatus:
    if requested == "cpu":
        return BackendStatus(requested="cpu", used="cpu", reason="cpu_requested")
    cp = _cupy_handle()
    if cp is None:
        return BackendStatus(
            requested="gpu",
            used="cpu",
            reason="gpu_unavailable_fallback_cpu",
        )
    return BackendStatus(requested="gpu", used="gpu", reason="gpu_available")


def gpu_available() -> bool:
    return _cupy_handle() is not None


def cost_density_on_backend(
    heat: np.ndarray,
    base_cost: float,
    alpha: float,
    epsilon: float,
    cost_mode: str,
    requested_backend: ComputeBackend,
) -> tuple[np.ndarray, BackendStatus]:
    status = resolve_backend(requested_backend)
    heat_f = np.asarray(heat, dtype=float)
    if status.used == "gpu":
        cp = _cupy_handle()
        if cp is None:
            status = BackendStatus(requested=requested_backend, used="cpu", reason="gpu_handle_failed")
        else:
            heat_gpu = cp.asarray(heat_f)
            if cost_mode == "density":
                w_gpu = base_cost + alpha * heat_gpu
            elif cost_mode == "inverse_speed":
                speed_gpu = 1.0 / (epsilon + heat_gpu)
                w_gpu = base_cost + alpha * (1.0 / speed_gpu)
            else:
                raise ValueError(f"Unsupported cost_mode: {cost_mode}")
            return cp.asnumpy(w_gpu), status

    if cost_mode == "density":
        w = base_cost + alpha * heat_f
    elif cost_mode == "inverse_speed":
        speed = 1.0 / (epsilon + heat_f)
        w = base_cost + alpha * (1.0 / speed)
    else:
        raise ValueError(f"Unsupported cost_mode: {cost_mode}")
    return np.asarray(w, dtype=float), status

