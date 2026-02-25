from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import BackendStatus
from .clearance import (
    ClearanceBase,
    ClearanceLayers,
    clearance_layers_from_base,
    precompute_clearance_base,
)
from .config import PlannerConfig
from .dijkstra_approx import DijkstraWorkspace, compute_cost_to_go_dijkstra
from .fmm import FmmWorkspace, compute_cost_to_go_fmm
from .heatmap import build_cost_density_with_backend


@dataclass
class PreparedEnvironment:
    key: tuple[Any, ...]
    cost_density: np.ndarray
    blocked: np.ndarray
    objective_density_base: np.ndarray
    clearance_base: ClearanceBase | None
    backend_status: BackendStatus


@dataclass
class PlannerRuntimeStats:
    environment_cache_hits: int = 0
    environment_cache_misses: int = 0
    goal_field_cache_hits: int = 0
    goal_field_cache_misses: int = 0
    environment_resets: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "environmentCacheHits": int(self.environment_cache_hits),
            "environmentCacheMisses": int(self.environment_cache_misses),
            "goalFieldCacheHits": int(self.goal_field_cache_hits),
            "goalFieldCacheMisses": int(self.goal_field_cache_misses),
            "environmentResets": int(self.environment_resets),
        }


class PlannerRuntime:
    def __init__(self, max_goal_cache_entries: int = 12) -> None:
        self._max_goal_cache_entries = max(1, int(max_goal_cache_entries))
        self._environment: PreparedEnvironment | None = None
        self._goal_field_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._planning_density_buffer: np.ndarray | None = None
        self._fmm_workspace = FmmWorkspace()
        self._dijkstra_workspace = DijkstraWorkspace()
        self._stats = PlannerRuntimeStats()

    @property
    def stats(self) -> PlannerRuntimeStats:
        return self._stats

    def _cfg_environment_key(self, cfg: PlannerConfig) -> tuple[Any, ...]:
        return (
            cfg.compute_backend,
            cfg.cost_mode,
            float(cfg.base_cost),
            float(cfg.alpha),
            float(cfg.epsilon),
            cfg.blocked_sentinel,
            bool(cfg.enable_clearance_constraints),
            float(cfg.resolution_m_per_cell),
            float(cfg.wall_clearance_weight),
            float(cfg.wall_clearance_power),
            float(cfg.wall_clearance_soft_ratio),
            bool(cfg.heat_region_clearance_enabled),
            cfg.heat_region_threshold,
            float(cfg.heat_region_quantile),
            float(cfg.heat_region_clearance_weight),
            float(cfg.heat_region_clearance_decay_m),
            bool(cfg.enforce_hard_clearance_if_feasible),
            float(cfg.object_width_m),
            float(cfg.object_height_m),
            cfg.object_shape,
            float(cfg.safe_space_m),
        )

    def _environment_key(
        self,
        heat: np.ndarray,
        blocked_mask: np.ndarray | None,
        cfg: PlannerConfig,
        environment_id: str | int | None,
    ) -> tuple[Any, ...]:
        if environment_id is not None:
            return ("id", environment_id, self._cfg_environment_key(cfg))
        return (
            "obj",
            id(heat),
            id(blocked_mask) if blocked_mask is not None else -1,
            heat.shape,
            heat.dtype.str,
            self._cfg_environment_key(cfg),
        )

    def reset(self) -> None:
        self._environment = None
        self._goal_field_cache.clear()
        self._planning_density_buffer = None
        self._stats.environment_resets += 1

    def prepare_environment(
        self,
        heat: np.ndarray,
        cfg: PlannerConfig,
        blocked_mask: np.ndarray | None = None,
        environment_id: str | int | None = None,
    ) -> PreparedEnvironment:
        key = self._environment_key(heat, blocked_mask, cfg, environment_id)
        if self._environment is not None and self._environment.key == key:
            self._stats.environment_cache_hits += 1
            return self._environment

        self._stats.environment_cache_misses += 1
        cost_density, blocked, backend_status = build_cost_density_with_backend(
            heat=heat,
            cfg=cfg,
            blocked_mask=blocked_mask,
            compute_backend=cfg.compute_backend,
        )

        clearance_base: ClearanceBase | None = None
        if cfg.enable_clearance_constraints:
            clearance_base = precompute_clearance_base(heat=heat, blocked=blocked, cfg=cfg)
            objective_density_base = cost_density + clearance_base.wall_penalty + clearance_base.heat_region_penalty
        else:
            objective_density_base = np.asarray(cost_density, dtype=float).copy()

        self._environment = PreparedEnvironment(
            key=key,
            cost_density=cost_density,
            blocked=blocked,
            objective_density_base=np.asarray(objective_density_base, dtype=float),
            clearance_base=clearance_base,
            backend_status=backend_status,
        )
        self._goal_field_cache.clear()
        if self._planning_density_buffer is None or self._planning_density_buffer.shape != heat.shape:
            self._planning_density_buffer = np.empty_like(cost_density, dtype=float)
        return self._environment

    def build_clearance_layers(
        self,
        env: PreparedEnvironment,
        cfg: PlannerConfig,
        start_rc: tuple[int, int],
        goal_rc: tuple[int, int],
    ) -> ClearanceLayers:
        if cfg.enable_clearance_constraints and env.clearance_base is not None:
            return clearance_layers_from_base(
                base=env.clearance_base,
                blocked=env.blocked,
                cfg=cfg,
                start_rc=start_rc,
                goal_rc=goal_rc,
            )

        zeros = np.zeros_like(env.cost_density, dtype=float)
        inf_clear = np.full_like(env.cost_density, np.inf, dtype=float)
        return ClearanceLayers(
            required_clearance_m=float(cfg.required_clearance_m),
            wall_clearance_m=inf_clear,
            heat_region_clearance_m=inf_clear,
            heat_region_mask=np.zeros_like(env.blocked, dtype=bool),
            heat_region_threshold=None,
            wall_penalty=zeros,
            heat_region_penalty=zeros,
            combined_penalty=zeros,
            planning_blocked=env.blocked.copy(),
            hard_clearance_feasible=True,
        )

    def planning_density(
        self,
        env: PreparedEnvironment,
        planning_blocked: np.ndarray,
    ) -> np.ndarray:
        if self._planning_density_buffer is None or self._planning_density_buffer.shape != env.cost_density.shape:
            self._planning_density_buffer = np.empty_like(env.cost_density, dtype=float)
        np.copyto(self._planning_density_buffer, env.objective_density_base)
        self._planning_density_buffer[planning_blocked] = np.inf
        return self._planning_density_buffer

    def _goal_cache_key(
        self,
        goal_rc: tuple[int, int],
        cfg: PlannerConfig,
        blocked_key: tuple[Any, ...] | None,
    ) -> tuple[Any, ...]:
        return (
            cfg.planner_mode,
            goal_rc,
            blocked_key,
        )

    def get_cost_to_go(
        self,
        planning_density: np.ndarray,
        planning_blocked: np.ndarray,
        goal_rc: tuple[int, int],
        cfg: PlannerConfig,
        blocked_key: tuple[Any, ...] | None = None,
    ) -> np.ndarray:
        cache_enabled = bool(cfg.cache_goal_fields)
        key = self._goal_cache_key(goal_rc, cfg, blocked_key)
        if cache_enabled:
            cached = self._goal_field_cache.get(key)
            if cached is not None:
                self._stats.goal_field_cache_hits += 1
                self._goal_field_cache.move_to_end(key)
                return cached

        self._stats.goal_field_cache_misses += 1
        if cfg.planner_mode == "fmm":
            t_field = compute_cost_to_go_fmm(
                cost_density=planning_density,
                goal_rc=goal_rc,
                blocked=planning_blocked,
                resolution_m=cfg.resolution_m_per_cell,
                workspace=self._fmm_workspace,
            )
        elif cfg.planner_mode == "dijkstra_approx":
            t_field = compute_cost_to_go_dijkstra(
                cost_density=planning_density,
                goal_rc=goal_rc,
                blocked=planning_blocked,
                resolution_m=cfg.resolution_m_per_cell,
                workspace=self._dijkstra_workspace,
            )
        else:
            raise ValueError(f"Unsupported planner mode: {cfg.planner_mode}")

        if not cache_enabled:
            return np.array(t_field, copy=True)

        cached = np.array(t_field, copy=True)
        self._goal_field_cache[key] = cached
        self._goal_field_cache.move_to_end(key)
        while len(self._goal_field_cache) > max(1, int(cfg.max_goal_field_cache_entries)):
            self._goal_field_cache.popitem(last=False)
        return cached

