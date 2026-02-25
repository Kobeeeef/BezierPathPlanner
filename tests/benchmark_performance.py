from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Performance benchmark suite for rapid replanning.")
    p.add_argument("--benchmark-kind", choices=["suite", "target_loop"], default="target_loop")
    p.add_argument("--replans-per-case", type=int, default=120)
    p.add_argument("--mode", choices=["runtime_fast", "debug_diagnostics"], default="runtime_fast")
    p.add_argument("--planner-mode", choices=["fmm", "dijkstra_approx"], default="fmm")
    p.add_argument("--out-dir", default="outputs/benchmarks")
    p.add_argument("--target-avg-ms", type=float, default=200.0)
    p.add_argument("--max-optimization-iterations", type=int, default=4)
    p.add_argument("--quality-check-stride", type=int, default=1)
    p.add_argument("--determinism-sample-count", type=int, default=12)
    p.add_argument("--deterministic-seed", type=int, default=0)
    p.add_argument("--max-p95-ms", type=float, default=120.0)
    p.add_argument("--max-p99-ms", type=float, default=180.0)
    p.add_argument("--max-worst-ms", type=float, default=260.0)
    p.add_argument("--max-quality-fail-ratio", type=float, default=0.08)
    p.add_argument("--max-memory-growth-mb", type=float, default=35.0)
    p.add_argument("--strict", action="store_true", help="Return non-zero exit code when thresholds fail.")
    return p


def main(argv: list[str] | None = None) -> int:
    _bootstrap_src_path()
    from path_planner.benchmarking import (
        BenchmarkCaseResult,
        BenchmarkThresholds,
        run_benchmark_suite,
        run_timing_target_validation_loop,
        write_results_csv,
        write_results_json,
    )

    args = build_arg_parser().parse_args(argv)
    thresholds = BenchmarkThresholds(
        max_p95_ms=float(args.max_p95_ms),
        max_p99_ms=float(args.max_p99_ms),
        max_worst_ms=float(args.max_worst_ms),
        max_quality_fail_ratio=float(args.max_quality_fail_ratio),
        max_memory_growth_mb=float(args.max_memory_growth_mb),
    )

    if args.benchmark_kind == "target_loop":
        payload = run_timing_target_validation_loop(
            target_avg_ms=float(args.target_avg_ms),
            replans_per_case=int(args.replans_per_case),
            planner_mode=args.planner_mode,
            thresholds=thresholds,
            max_optimization_iterations=int(args.max_optimization_iterations),
            quality_check_stride=int(args.quality_check_stride),
            determinism_sample_count=int(args.determinism_sample_count),
            deterministic_seed=int(args.deterministic_seed),
        )
    else:
        payload = run_benchmark_suite(
            replans_per_case=int(args.replans_per_case),
            mode=args.mode,
            planner_mode=args.planner_mode,
            thresholds=thresholds,
            latency_source="planner_total_ms",
            include_runtime_payload=False,
            include_json_serialize=False,
            quality_check_stride=int(args.quality_check_stride),
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    csv_path = out_dir / "benchmark_results.csv"
    write_results_json(json_path, payload)
    cases = payload.get("cases", [])

    case_objs = []
    for c in cases:
        case_objs.append(
            BenchmarkCaseResult(
                name=c["name"],
                mode=c["mode"],
                replans=int(c["replans"]),
                latency_stats_ms=dict(c["latencyStatsMs"]),
                stage_stats_ms=dict(c["stageStatsMs"]),
                quality_failures=int(c["qualityFailures"]),
                quality_failure_ratio=float(c["qualityFailureRatio"]),
                pass_fail=dict(c["passFail"]),
                memory_growth_mb=float(c["memoryGrowthMb"]),
                metadata=dict(c["metadata"]),
                latency_source=str(c.get("latencySource", "planner_total_ms")),
                per_run_latencies_ms=[float(v) for v in c.get("perRunLatenciesMs", [])],
                per_run_records=list(c.get("perRunRecords", [])),
                determinism=dict(c.get("determinism", {})),
            )
        )
    write_results_csv(csv_path, case_objs)

    print(payload.get("summaryTable", ""))
    overall = payload.get("overallLatencyStats", {})
    if isinstance(overall, dict) and overall:
        avg = float(overall.get("avgMs", 0.0))
        med = float(overall.get("medianMs", 0.0))
        p95 = float(overall.get("p95Ms", 0.0))
        p99 = float(overall.get("p99Ms", 0.0))
        worst = float(overall.get("worstMs", 0.0))
        hz = float(overall.get("throughputHz", 0.0))
        print()
        print("Overall latency stats:")
        print(
            f"  avg={avg:.3f} ms median={med:.3f} ms p95={p95:.3f} ms "
            f"p99={p99:.3f} ms worst={worst:.3f} ms throughput={hz:.3f} plans/s"
        )

    target_pass = bool(payload.get("targetPass", payload.get("allPassed", False)))
    if args.benchmark_kind == "target_loop":
        print()
        status = "PASS" if target_pass else "FAIL"
        print(f"Target gate (< {float(args.target_avg_ms):.3f} ms average): {status}")
        if not target_pass:
            bottlenecks = payload.get("bottlenecks", [])
            if isinstance(bottlenecks, list) and bottlenecks:
                print("Top stage bottlenecks:")
                for b in bottlenecks[:6]:
                    if isinstance(b, dict):
                        stage = b.get("stage", "?")
                        avg_ms = float(b.get("avgMs", 0.0))
                        p95_ms = float(b.get("p95Ms", 0.0))
                        print(f"  - {stage}: avg={avg_ms:.3f} ms, p95={p95_ms:.3f} ms")
            recs = payload.get("nextRecommendedOptimizations", [])
            if isinstance(recs, list) and recs:
                print("Next recommended optimizations:")
                for r in recs:
                    print(f"  - {r}")

    print()
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV : {csv_path.resolve()}")
    if args.strict and not target_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
