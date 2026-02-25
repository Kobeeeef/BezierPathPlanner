from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PlannerConfig
from .export_pathplanner import build_runtime_payload_compact
from .performance import aggregate_stage_stats, latency_stats
from .planner import PlannerArtifacts, run_planner
from .runtime import PlannerRuntime
from .scenarios import Scenario, get_scenarios


@dataclass(frozen=True)
class BenchmarkThresholds:
    max_avg_ms: float | None = None
    max_p95_ms: float = 120.0
    max_p99_ms: float = 180.0
    max_worst_ms: float = 260.0
    max_quality_fail_ratio: float = 0.08
    max_memory_growth_mb: float = 35.0


@dataclass
class BenchmarkCaseResult:
    name: str
    mode: str
    replans: int
    latency_stats_ms: dict[str, float]
    stage_stats_ms: dict[str, dict[str, float]]
    quality_failures: int
    quality_failure_ratio: float
    pass_fail: dict[str, bool]
    memory_growth_mb: float
    metadata: dict[str, Any]
    latency_source: str = "planner_total_ms"
    per_run_latencies_ms: list[float] = field(default_factory=list)
    per_run_records: list[dict[str, Any]] = field(default_factory=list)
    determinism: dict[str, Any] = field(default_factory=dict)
    per_run_stage_rows: list[dict[str, float]] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "replans": self.replans,
            "latencySource": self.latency_source,
            "latencyStatsMs": self.latency_stats_ms,
            "stageStatsMs": self.stage_stats_ms,
            "perRunLatenciesMs": self.per_run_latencies_ms,
            "perRunRecords": self.per_run_records,
            "qualityFailures": self.quality_failures,
            "qualityFailureRatio": self.quality_failure_ratio,
            "passFail": self.pass_fail,
            "determinism": self.determinism,
            "memoryGrowthMb": self.memory_growth_mb,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BenchmarkOptimizationProfile:
    name: str
    rationale: str
    cfg_updates: dict[str, Any]


def _scenario_with_resolution(s: Scenario, step: int) -> Scenario:
    if step <= 1:
        return s
    heat = np.asarray(s.heat[::step, ::step], dtype=float)
    blocked = None if s.blocked_mask is None else np.asarray(s.blocked_mask[::step, ::step], dtype=bool)
    res = float(s.resolution_m_per_cell * step)

    def remap_point(pt: tuple[float, float], shape: tuple[int, int]) -> tuple[float, float]:
        x_idx = int(round(pt[0] / res))
        y_idx = int(round(pt[1] / res))
        x_idx = int(np.clip(x_idx, 0, shape[1] - 1))
        y_idx = int(np.clip(y_idx, 0, shape[0] - 1))
        return (x_idx * res, y_idx * res)

    start = remap_point(s.start_xy, heat.shape)
    goal = remap_point(s.goal_xy, heat.shape)
    return Scenario(
        name=f"{s.name}_res_step_{step}",
        description=f"{s.description} (resolution step {step})",
        heat=heat,
        start_xy=start,
        goal_xy=goal,
        resolution_m_per_cell=res,
        start_heading_deg=s.start_heading_deg,
        end_heading_deg=s.end_heading_deg,
        start_approach_heading_deg=s.start_approach_heading_deg,
        goal_approach_heading_deg=s.goal_approach_heading_deg,
        start_approach_lock_distance_m=s.start_approach_lock_distance_m,
        goal_approach_lock_distance_m=s.goal_approach_lock_distance_m,
        blocked_mask=blocked,
    )


def _quality_ok(art: PlannerArtifacts, cfg: PlannerConfig) -> bool:
    summary = art.summary
    if not bool(summary.get("pathExists", False)):
        return False
    if summary.get("startEndpointAlignmentErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg * 1.8:
        return False
    if summary.get("endEndpointAlignmentErrorDeg", 0.0) > cfg.endpoint_alignment_tolerance_deg * 1.8:
        return False
    if cfg.enable_clearance_constraints and bool(summary.get("hardClearanceFeasible", False)):
        req = float(summary.get("requiredClearanceM", 0.0))
        if float(summary.get("minWallClearanceM", req)) < req - 0.06:
            return False
    diag = art.smoothing_diagnostics.get("smoothedPathDiagnostics", {})
    if diag.get("selfIntersectionCount", 0.0) > 0.0:
        return False
    if diag.get("terminalOvershootCount", 0.0) > 0.0:
        return False
    return True


def _spiral_offset(i: int, amp: float) -> tuple[float, float]:
    theta = 0.47 * i
    r = amp * (0.35 + 0.65 * ((i % 11) / 10.0))
    return (r * math.cos(theta), r * math.sin(theta))


def _bounded_point(
    pt: tuple[float, float],
    shape: tuple[int, int],
    res: float,
) -> tuple[float, float]:
    max_x = (shape[1] - 1) * res
    max_y = (shape[0] - 1) * res
    return (float(np.clip(pt[0], 0.0, max_x)), float(np.clip(pt[1], 0.0, max_y)))


def _runtime_payload_in_memory(art: PlannerArtifacts, cfg: PlannerConfig) -> dict[str, Any]:
    return build_runtime_payload_compact(
        bezier_segments=art.bezier_segments,
        sampled_points=art.sampled_smoothed_points,
        sampled_path_tangent_headings_rad=art.sampled_path_tangent_headings_rad,
        sampled_holonomic_rotations_rad=art.sampled_holonomic_rotations_rad,
        summary=art.summary,
        required_clearance_m=art.required_clearance_m,
        backend_status=art.backend_status,
        cfg=cfg,
    )


def _stable_path_signature(art: PlannerArtifacts) -> str:
    bezier_points: list[list[float]] = []
    for seg in art.bezier_segments:
        flat = np.concatenate((seg.p0, seg.p1, seg.p2, seg.p3))
        bezier_points.append([float(np.round(v, 6)) for v in flat.tolist()])
    payload = {
        "rawPathCells": [[int(r), int(c)] for r, c in art.raw_path_cells],
        "bezier": bezier_points,
        "smoothedLengthM": float(np.round(float(art.summary.get("smoothedLengthM", 0.0)), 6)),
        "rawLengthM": float(np.round(float(art.summary.get("rawLengthM", 0.0)), 6)),
        "objectiveCost": float(np.round(float(art.summary.get("smoothedObjectiveIntegratedCost", 0.0)), 6)),
        "bezierSegmentCount": int(art.summary.get("bezierSegmentCount", len(art.bezier_segments))),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _all_case_latencies(cases: list[BenchmarkCaseResult]) -> list[float]:
    latencies: list[float] = []
    for c in cases:
        latencies.extend(float(v) for v in c.per_run_latencies_ms)
    return latencies


def _all_case_stage_rows(cases: list[BenchmarkCaseResult]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for c in cases:
        rows.extend(c.per_run_stage_rows)
    return rows


def _top_stage_bottlenecks(
    stage_stats: dict[str, dict[str, float]],
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    items: list[tuple[str, float, float]] = []
    excluded = {"totalRuntimeMs", "totalRuntimeWithExportMs", "endToEndRuntimeMs"}
    for name, stats in stage_stats.items():
        if name in excluded:
            continue
        avg_ms = float(stats.get("avgMs", 0.0))
        p95_ms = float(stats.get("p95Ms", 0.0))
        if not math.isfinite(avg_ms) or avg_ms <= 0.0:
            continue
        items.append((name, avg_ms, p95_ms))
    items.sort(key=lambda x: x[1], reverse=True)
    out: list[dict[str, float | str]] = []
    for name, avg_ms, p95_ms in items[: max(1, int(top_k))]:
        out.append({"stage": name, "avgMs": avg_ms, "p95Ms": p95_ms})
    return out


def _recommended_optimizations(
    bottlenecks: list[dict[str, float | str]],
    planner_mode: str,
) -> list[str]:
    if not bottlenecks:
        return [
            "Collect per-stage profiles with a larger run count and include CPU affinity/power-state controls.",
            "Reduce variance by pinning benchmark process priority and disabling background heavy workloads.",
        ]
    recs: list[str] = []
    top = str(bottlenecks[0].get("stage", ""))
    if top == "smoothing":
        recs.append(
            "Lower smoothing refit budget further (`runtime_fast_max_refits=1`, smaller `max_smoothing_refits`) and increase `bezier_target_segment_length_m`."
        )
    elif top == "propagation":
        if planner_mode != "dijkstra_approx":
            recs.append(
                "For strict latency mode, switch propagation to `dijkstra_approx` or multi-resolution warm-started propagation."
            )
        recs.append("Reuse fixed-size propagation front buffers and avoid planner-mode toggles across runs.")
    elif top == "clearance_validation":
        recs.append("Cache and reuse immutable clearance fields per environment and reduce repeated feasibility recomputes.")
    elif top in ("exportObjectBuildMs", "exportJsonSerializeMs"):
        recs.append("Reduce sampled-path export density in runtime payload and keep compact JSON serialization only.")
    else:
        recs.append("Focus optimization on the top stage and re-run target validation with the same deterministic workload.")
    recs.append("If latency spikes persist, reduce map resolution or use adaptive multi-resolution for difficult maps.")
    return recs


def _optimization_profiles(planner_mode: str) -> list[BenchmarkOptimizationProfile]:
    profiles: list[BenchmarkOptimizationProfile] = [
        BenchmarkOptimizationProfile(
            name="baseline_runtime_fast",
            rationale="Reference runtime_fast config with full core behavior enabled.",
            cfg_updates={},
        ),
        BenchmarkOptimizationProfile(
            name="trim_refits",
            rationale="Reduce smoothing refit work while preserving smoothing + clearance.",
            cfg_updates={
                "runtime_fast_max_refits": 2,
                "max_smoothing_refits": 4,
                "sample_ds_m": 0.09,
                "path_step_m": 0.09,
            },
        ),
        BenchmarkOptimizationProfile(
            name="lighter_sampling",
            rationale="Lower per-plan sampling density and segment pressure in runtime mode.",
            cfg_updates={
                "runtime_fast_max_refits": 1,
                "max_smoothing_refits": 3,
                "sample_ds_m": 0.11,
                "path_step_m": 0.10,
                "bezier_target_segment_length_m": 2.10,
            },
        ),
    ]
    if planner_mode != "dijkstra_approx":
        profiles.append(
            BenchmarkOptimizationProfile(
                name="fast_propagation_fallback",
                rationale="Use faster propagation fallback when propagation dominates latency budget.",
                cfg_updates={
                    "planner_mode": "dijkstra_approx",
                    "runtime_fast_max_refits": 1,
                    "max_smoothing_refits": 2,
                    "sample_ds_m": 0.12,
                    "path_step_m": 0.11,
                    "bezier_target_segment_length_m": 2.25,
                },
            )
        )
    return profiles


def _run_case(
    case_name: str,
    scenario: Scenario,
    cfg: PlannerConfig,
    replans: int,
    vary_start_goal: bool = False,
    vary_goal: bool = True,
    mutate_heat: bool = False,
    mutate_blocked: bool = False,
    warm_start: bool = True,
    thresholds: BenchmarkThresholds | None = None,
    latency_source: str = "planner_total_ms",
    include_runtime_payload: bool = False,
    include_json_serialize: bool = False,
    quality_check_stride: int = 1,
    determinism_sample_count: int = 0,
) -> BenchmarkCaseResult:
    th = thresholds or BenchmarkThresholds()
    runtime = PlannerRuntime(max_goal_cache_entries=max(1, int(cfg.max_goal_field_cache_entries)))
    latencies_ms: list[float] = []
    stage_rows: list[dict[str, float]] = []
    per_run_records: list[dict[str, Any]] = []
    quality_failures = 0
    quality_checks = 0
    replans = max(1, int(replans))
    check_stride = max(1, int(quality_check_stride))
    determinism_signatures: list[str] = []
    run_determinism_check = (
        determinism_sample_count > 1 and not vary_start_goal and not mutate_heat and not mutate_blocked
    )

    tracemalloc.start()
    start_current, _ = tracemalloc.get_traced_memory()
    for i in range(replans):
        if not warm_start:
            runtime.reset()

        heat = scenario.heat
        blocked = scenario.blocked_mask
        env_id: str | int | None = case_name
        if mutate_heat:
            yy, xx = np.mgrid[0 : heat.shape[0], 0 : heat.shape[1]]
            delta = 0.22 * np.sin((xx + 0.9 * i) / 14.0) + 0.18 * np.cos((yy - 0.7 * i) / 17.0)
            heat = np.clip(np.asarray(heat, dtype=float) + delta, 0.25, None)
            env_id = f"{case_name}_heat_{i}"
        if mutate_blocked:
            if blocked is None:
                blocked = np.zeros_like(scenario.heat, dtype=bool)
            blocked = np.asarray(blocked, dtype=bool).copy()
            rr = (5 + 3 * i) % max(6, blocked.shape[0] - 6)
            cc = (7 + 5 * i) % max(8, blocked.shape[1] - 8)
            blocked[rr : rr + 2, cc : cc + 2] = True
            env_id = f"{case_name}_blocked_{i}"

        start_xy = scenario.start_xy
        goal_xy = scenario.goal_xy
        if vary_start_goal:
            off_s = _spiral_offset(i, amp=0.45)
            start_xy = _bounded_point(
                (scenario.start_xy[0] + off_s[0], scenario.start_xy[1] + off_s[1]),
                heat.shape,
                scenario.resolution_m_per_cell,
            )
            if vary_goal:
                off_g = _spiral_offset(i + 7, amp=0.55)
                goal_xy = _bounded_point(
                    (scenario.goal_xy[0] + off_g[0], scenario.goal_xy[1] + off_g[1]),
                    heat.shape,
                    scenario.resolution_m_per_cell,
                )

        e2e_start_ns = time.perf_counter_ns()
        art = run_planner(
            heat=np.asarray(heat, dtype=float),
            start_xy=start_xy,
            goal_xy=goal_xy,
            cfg=cfg,
            blocked_mask=blocked,
            runtime=runtime,
            environment_id=env_id,
        )
        export_obj_ms = 0.0
        export_json_ms = 0.0
        if include_runtime_payload:
            export_start_ns = time.perf_counter_ns()
            runtime_payload = _runtime_payload_in_memory(art, cfg)
            export_obj_ms = (time.perf_counter_ns() - export_start_ns) / 1_000_000.0
            if include_json_serialize:
                serialize_start_ns = time.perf_counter_ns()
                _ = json.dumps(runtime_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                export_json_ms = (time.perf_counter_ns() - serialize_start_ns) / 1_000_000.0
        end_to_end_ms = (time.perf_counter_ns() - e2e_start_ns) / 1_000_000.0

        planner_total_ms = float(art.stage_timings_ms.get("totalRuntimeMs", 0.0))
        if latency_source == "end_to_end_runtime_payload_ms":
            observed_latency = float(end_to_end_ms)
        else:
            observed_latency = planner_total_ms
        latencies_ms.append(observed_latency)

        stage_row = dict(art.stage_timings_ms)
        stage_row["exportObjectBuildMs"] = float(export_obj_ms)
        stage_row["exportJsonSerializeMs"] = float(export_json_ms)
        stage_row["endToEndRuntimeMs"] = float(end_to_end_ms)
        stage_rows.append(stage_row)

        quality_checked = (i % check_stride) == 0
        quality_ok = True
        if quality_checked:
            quality_checks += 1
            quality_ok = _quality_ok(art, cfg)
            if not quality_ok:
                quality_failures += 1

        if run_determinism_check and i < int(determinism_sample_count):
            determinism_signatures.append(_stable_path_signature(art))

        per_run_records.append(
            {
                "runIndex": i,
                "scenario": scenario.name,
                "latencyMs": float(observed_latency),
                "plannerTotalMs": planner_total_ms,
                "endToEndRuntimeMs": float(end_to_end_ms),
                "exportObjectBuildMs": float(export_obj_ms),
                "exportJsonSerializeMs": float(export_json_ms),
                "startXY": [float(start_xy[0]), float(start_xy[1])],
                "goalXY": [float(goal_xy[0]), float(goal_xy[1])],
                "environmentId": str(env_id),
                "qualityChecked": bool(quality_checked),
                "qualityOk": bool(quality_ok) if quality_checked else None,
            }
        )

    end_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_growth_mb = float(max(0, end_current - start_current) / (1024.0 * 1024.0))
    peak_mb = float(peak / (1024.0 * 1024.0))

    lat = latency_stats(latencies_ms)
    checks = max(1, int(quality_checks))
    q_ratio = float(quality_failures / checks)
    avg_ok = True
    if th.max_avg_ms is not None:
        avg_ok = float(lat.get("avgMs", 0.0)) < float(th.max_avg_ms)
    determinism_ok = True
    distinct_signatures = len(set(determinism_signatures))
    if run_determinism_check and len(determinism_signatures) > 1:
        determinism_ok = distinct_signatures == 1
    pass_fail = {
        "avg": bool(avg_ok),
        "p95": float(lat.get("p95Ms", 0.0)) <= th.max_p95_ms,
        "p99": float(lat.get("p99Ms", 0.0)) <= th.max_p99_ms,
        "worst": float(lat.get("worstMs", 0.0)) <= th.max_worst_ms,
        "quality": q_ratio <= th.max_quality_fail_ratio,
        "memory_growth": mem_growth_mb <= th.max_memory_growth_mb,
    }
    if run_determinism_check:
        pass_fail["determinism"] = bool(determinism_ok)

    determinism = {
        "enabled": bool(run_determinism_check),
        "sampleCount": int(min(replans, int(max(0, determinism_sample_count)))),
        "distinctSignatures": int(distinct_signatures),
        "deterministic": bool(determinism_ok),
    }

    return BenchmarkCaseResult(
        name=case_name,
        mode=cfg.runtime_mode,
        replans=replans,
        latency_source=latency_source,
        latency_stats_ms=lat,
        stage_stats_ms=aggregate_stage_stats(stage_rows),
        per_run_latencies_ms=[float(v) for v in latencies_ms],
        per_run_records=per_run_records,
        per_run_stage_rows=stage_rows,
        determinism=determinism,
        quality_failures=quality_failures,
        quality_failure_ratio=q_ratio,
        pass_fail=pass_fail,
        memory_growth_mb=mem_growth_mb,
        metadata={
            "scenario": scenario.name,
            "peakMemoryMb": peak_mb,
            "cacheStats": runtime.stats.as_dict(),
            "computeBackend": cfg.compute_backend,
            "plannerMode": cfg.planner_mode,
            "smoothingEnabled": bool(cfg.enable_smoothing),
            "clearanceEnabled": bool(cfg.enable_clearance_constraints),
            "warmStart": bool(warm_start),
            "qualityCheckStride": int(check_stride),
            "qualityChecks": int(quality_checks),
            "deterministicSeed": int(cfg.deterministic_seed),
        },
    )


def _case_rows(cases: list[BenchmarkCaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in cases:
        lat = c.latency_stats_ms
        rows.append(
            {
                "case": c.name,
                "mode": c.mode,
                "latency_source": c.latency_source,
                "replans": c.replans,
                "avg_ms": float(lat.get("avgMs", 0.0)),
                "median_ms": float(lat.get("medianMs", 0.0)),
                "p95_ms": float(lat.get("p95Ms", 0.0)),
                "p99_ms": float(lat.get("p99Ms", 0.0)),
                "worst_ms": float(lat.get("worstMs", 0.0)),
                "throughput_hz": float(lat.get("throughputHz", 0.0)),
                "quality_failures": int(c.quality_failures),
                "quality_fail_ratio": float(c.quality_failure_ratio),
                "memory_growth_mb": float(c.memory_growth_mb),
                "avg_target_pass": bool(c.pass_fail.get("avg", True)),
                "pass_all": bool(all(c.pass_fail.values())),
            }
        )
    return rows


def render_summary_table(cases: list[BenchmarkCaseResult]) -> str:
    rows = _case_rows(cases)
    if not rows:
        return "No benchmark rows."
    headers = [
        "case",
        "avg_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "worst_ms",
        "throughput_hz",
        "quality_fail_ratio",
        "memory_growth_mb",
        "avg_target_pass",
        "pass_all",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(f"{row[h]:.3f}" if isinstance(row[h], float) else str(row[h])))
    lines: list[str] = []
    lines.append(" | ".join(h.ljust(widths[h]) for h in headers))
    lines.append("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        parts: list[str] = []
        for h in headers:
            v = row[h]
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            parts.append(txt.ljust(widths[h]))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_results_csv(path: Path, cases: list[BenchmarkCaseResult]) -> None:
    rows = _case_rows(cases)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark_suite(
    replans_per_case: int = 40,
    mode: str = "runtime_fast",
    planner_mode: str = "fmm",
    thresholds: BenchmarkThresholds | None = None,
    latency_source: str = "planner_total_ms",
    include_runtime_payload: bool = False,
    include_json_serialize: bool = False,
    quality_check_stride: int = 1,
) -> dict[str, Any]:
    scenarios = get_scenarios()
    base = PlannerConfig(
        runtime_mode=mode,  # type: ignore[arg-type]
        planner_mode=planner_mode,  # type: ignore[arg-type]
        stage_timing_enabled=True,
        cache_goal_fields=True,
        enable_smoothing=True,
        enable_clearance_constraints=True,
    )
    th = thresholds or BenchmarkThresholds()
    cases: list[BenchmarkCaseResult] = []

    t0 = time.perf_counter()
    cases.append(
        _run_case(
            case_name="spam_same_map_vary_start_goal_warm",
            scenario=scenarios["hot_island"],
            cfg=base,
            replans=replans_per_case,
            vary_start_goal=True,
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="spam_same_map_vary_start_same_goal_warm",
            scenario=scenarios["hot_island"],
            cfg=base,
            replans=replans_per_case,
            vary_start_goal=True,
            vary_goal=False,
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="spam_same_map_vary_start_goal_cold",
            scenario=scenarios["hot_island"],
            cfg=base,
            replans=replans_per_case,
            vary_start_goal=True,
            warm_start=False,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="spam_same_start_goal_changing_heat",
            scenario=scenarios["small_islands_weave"],
            cfg=base,
            replans=replans_per_case,
            mutate_heat=True,
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="spam_changing_blocked_mask",
            scenario=scenarios["blocked_gap"],
            cfg=base,
            replans=max(20, replans_per_case // 2),
            mutate_blocked=True,
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )

    for name in ("hot_island", "uniform_high", "double_hot_mid_corridor", "small_islands_weave"):
        cases.append(
            _run_case(
                case_name=f"stability_{name}",
                scenario=scenarios[name],
                cfg=base,
                replans=max(14, replans_per_case // 3),
                warm_start=True,
                thresholds=th,
                latency_source=latency_source,
                include_runtime_payload=include_runtime_payload,
                include_json_serialize=include_json_serialize,
                quality_check_stride=quality_check_stride,
            )
        )

    for step in (1, 2, 3):
        sc = _scenario_with_resolution(scenarios["hot_island"], step=step)
        cfg = base.with_updates(resolution_m_per_cell=sc.resolution_m_per_cell)
        cases.append(
            _run_case(
                case_name=f"resolution_scaling_step_{step}",
                scenario=sc,
                cfg=cfg,
                replans=max(14, replans_per_case // 3),
                warm_start=True,
                thresholds=th,
                latency_source=latency_source,
                include_runtime_payload=include_runtime_payload,
                include_json_serialize=include_json_serialize,
                quality_check_stride=quality_check_stride,
            )
        )

    cases.append(
        _run_case(
            case_name="without_smoothing",
            scenario=scenarios["hot_island"],
            cfg=base.with_updates(enable_smoothing=False),
            replans=max(18, replans_per_case // 2),
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="without_clearance_constraints",
            scenario=scenarios["hot_island"],
            cfg=base.with_updates(enable_clearance_constraints=False),
            replans=max(18, replans_per_case // 2),
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )

    cases.append(
        _run_case(
            case_name="cpu_backend_baseline",
            scenario=scenarios["hot_island"],
            cfg=base.with_updates(compute_backend="cpu"),
            replans=max(18, replans_per_case // 2),
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )
    cases.append(
        _run_case(
            case_name="gpu_backend_compare",
            scenario=scenarios["hot_island"],
            cfg=base.with_updates(compute_backend="gpu"),
            replans=max(18, replans_per_case // 2),
            warm_start=True,
            thresholds=th,
            latency_source=latency_source,
            include_runtime_payload=include_runtime_payload,
            include_json_serialize=include_json_serialize,
            quality_check_stride=quality_check_stride,
        )
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    overall_lat = latency_stats(_all_case_latencies(cases))
    payload = {
        "generatedAtMs": elapsed_ms,
        "config": {
            "mode": mode,
            "plannerMode": planner_mode,
            "replansPerCase": replans_per_case,
            "latencySource": latency_source,
            "runtimePayloadMeasured": bool(include_runtime_payload),
            "jsonSerializationMeasured": bool(include_json_serialize),
            "qualityCheckStride": int(max(1, quality_check_stride)),
            "thresholds": {
                "maxAvgMs": th.max_avg_ms,
                "maxP95Ms": th.max_p95_ms,
                "maxP99Ms": th.max_p99_ms,
                "maxWorstMs": th.max_worst_ms,
                "maxQualityFailRatio": th.max_quality_fail_ratio,
                "maxMemoryGrowthMb": th.max_memory_growth_mb,
            },
        },
        "overallLatencyStats": overall_lat,
        "allPassed": bool(all(all(c.pass_fail.values()) for c in cases)),
        "cases": [c.as_dict() for c in cases],
        "summaryTable": render_summary_table(cases),
    }
    return payload


def run_timing_target_validation_loop(
    target_avg_ms: float = 200.0,
    replans_per_case: int = 120,
    planner_mode: str = "fmm",
    thresholds: BenchmarkThresholds | None = None,
    max_optimization_iterations: int = 4,
    quality_check_stride: int = 1,
    determinism_sample_count: int = 12,
    deterministic_seed: int = 0,
) -> dict[str, Any]:
    scenarios = get_scenarios()
    base = PlannerConfig(
        runtime_mode="runtime_fast",
        planner_mode=planner_mode,  # type: ignore[arg-type]
        stage_timing_enabled=True,
        cache_goal_fields=True,
        enable_smoothing=True,
        enable_clearance_constraints=True,
        deterministic_seed=int(deterministic_seed),
    )
    max_iters = max(1, int(max_optimization_iterations))
    base_th = thresholds or BenchmarkThresholds()
    th = BenchmarkThresholds(
        max_avg_ms=float(target_avg_ms),
        max_p95_ms=float(base_th.max_p95_ms),
        max_p99_ms=float(base_th.max_p99_ms),
        max_worst_ms=float(base_th.max_worst_ms),
        max_quality_fail_ratio=float(base_th.max_quality_fail_ratio),
        max_memory_growth_mb=float(base_th.max_memory_growth_mb),
    )

    profiles = _optimization_profiles(planner_mode=str(planner_mode))
    iterations: list[dict[str, Any]] = []
    final_cases: list[BenchmarkCaseResult] = []
    final_overall_lat: dict[str, float] = latency_stats([])
    final_stage_stats: dict[str, dict[str, float]] = {}
    final_bottlenecks: list[dict[str, float | str]] = []
    target_pass = False
    quality_guard_pass = False
    determinism_pass = False

    t0 = time.perf_counter()
    for idx in range(max_iters):
        profile = profiles[min(idx, len(profiles) - 1)]
        cfg = base.with_updates(**profile.cfg_updates)
        cases: list[BenchmarkCaseResult] = []

        cases.append(
            _run_case(
                case_name="tight_loop_same_map_same_start_goal",
                scenario=scenarios["hot_island"],
                cfg=cfg,
                replans=replans_per_case,
                warm_start=True,
                thresholds=th,
                latency_source="end_to_end_runtime_payload_ms",
                include_runtime_payload=True,
                include_json_serialize=True,
                quality_check_stride=quality_check_stride,
                determinism_sample_count=determinism_sample_count,
            )
        )
        cases.append(
            _run_case(
                case_name="spam_same_map_changing_start_goal",
                scenario=scenarios["hot_island"],
                cfg=cfg,
                replans=replans_per_case,
                vary_start_goal=True,
                warm_start=True,
                thresholds=th,
                latency_source="end_to_end_runtime_payload_ms",
                include_runtime_payload=True,
                include_json_serialize=True,
                quality_check_stride=quality_check_stride,
            )
        )
        cases.append(
            _run_case(
                case_name="spam_same_start_goal_changing_heatmap",
                scenario=scenarios["small_islands_weave"],
                cfg=cfg,
                replans=replans_per_case,
                mutate_heat=True,
                warm_start=True,
                thresholds=th,
                latency_source="end_to_end_runtime_payload_ms",
                include_runtime_payload=True,
                include_json_serialize=True,
                quality_check_stride=quality_check_stride,
            )
        )
        cases.append(
            _run_case(
                case_name="spam_changing_blocked_mask",
                scenario=scenarios["blocked_gap"],
                cfg=cfg,
                replans=max(24, replans_per_case // 2),
                mutate_blocked=True,
                warm_start=True,
                thresholds=th,
                latency_source="end_to_end_runtime_payload_ms",
                include_runtime_payload=True,
                include_json_serialize=True,
                quality_check_stride=quality_check_stride,
            )
        )

        all_lat = _all_case_latencies(cases)
        overall_lat = latency_stats(all_lat)
        stage_rows = _all_case_stage_rows(cases)
        stage_stats = aggregate_stage_stats(stage_rows)
        bottlenecks = _top_stage_bottlenecks(stage_stats, top_k=6)
        summary_table = render_summary_table(cases)

        target_pass = float(overall_lat.get("avgMs", 0.0)) < float(target_avg_ms)
        quality_guard_pass = bool(all(c.pass_fail.get("quality", True) for c in cases))
        determinism_pass = bool(all(c.pass_fail.get("determinism", True) for c in cases))
        all_checks_pass = bool(target_pass and quality_guard_pass and determinism_pass)

        iterations.append(
            {
                "iteration": idx + 1,
                "profileName": profile.name,
                "profileRationale": profile.rationale,
                "profileConfigUpdates": profile.cfg_updates,
                "targetPass": bool(target_pass),
                "qualityGuardPass": bool(quality_guard_pass),
                "determinismPass": bool(determinism_pass),
                "allChecksPass": bool(all_checks_pass),
                "overallLatencyStatsMs": overall_lat,
                "overallStageStatsMs": stage_stats,
                "bottlenecks": bottlenecks,
                "summaryTable": summary_table,
                "cases": [c.as_dict() for c in cases],
            }
        )

        final_cases = cases
        final_overall_lat = overall_lat
        final_stage_stats = stage_stats
        final_bottlenecks = bottlenecks

        if target_pass:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    reached_iteration_limit = len(iterations) >= max_iters and not target_pass
    recommendations = []
    if reached_iteration_limit:
        recommendations = _recommended_optimizations(final_bottlenecks, planner_mode=str(planner_mode))

    payload = {
        "validationType": "timing_target_loop",
        "generatedAtMs": elapsed_ms,
        "config": {
            "mode": "runtime_fast",
            "plannerMode": planner_mode,
            "replansPerCase": int(replans_per_case),
            "targetAvgLatencyMs": float(target_avg_ms),
            "maxOptimizationIterations": int(max_iters),
            "qualityCheckStride": int(max(1, quality_check_stride)),
            "determinismSampleCount": int(max(0, determinism_sample_count)),
            "deterministicSeed": int(deterministic_seed),
            "latencySource": "end_to_end_runtime_payload_ms",
            "runtimePayloadMeasured": True,
            "jsonSerializationMeasured": True,
            "thresholds": {
                "maxAvgMs": th.max_avg_ms,
                "maxP95Ms": th.max_p95_ms,
                "maxP99Ms": th.max_p99_ms,
                "maxWorstMs": th.max_worst_ms,
                "maxQualityFailRatio": th.max_quality_fail_ratio,
                "maxMemoryGrowthMb": th.max_memory_growth_mb,
            },
        },
        "targetRule": f"PASS if average latency < {float(target_avg_ms):.3f} ms (strict less-than).",
        "targetPass": bool(target_pass),
        "qualityGuardPass": bool(quality_guard_pass),
        "determinismPass": bool(determinism_pass),
        "allChecksPass": bool(target_pass and quality_guard_pass and determinism_pass),
        "allPassed": bool(target_pass),
        "iterationsExecuted": len(iterations),
        "reachedIterationLimit": bool(reached_iteration_limit),
        "overallLatencyStats": final_overall_lat,
        "overallStageStatsMs": final_stage_stats,
        "bottlenecks": final_bottlenecks,
        "nextRecommendedOptimizations": recommendations,
        "cases": [c.as_dict() for c in final_cases],
        "summaryTable": render_summary_table(final_cases),
        "iterations": iterations,
    }
    return payload
