"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    ALGO=fla-recurrent modal run -m scripts.run_modal
    ALGO=pt-reference modal run -m scripts.run_modal
"""

import os

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

from .modal_config import TRACE_SET_PATH, image, trace_volume
from .pack_solution import pack_solution
from .shared import ALGO_ENTRY_POINTS, ALGO_LANGUAGES

app = modal.App("flashinfer-bench")

LOG_PATH = "/data/logs"


@app.function(
    image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume}
)
def run_benchmark(
    solution: Solution, config: BenchmarkConfig = None, num_workloads: int = 0
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(
            warmup_runs=3,
            iterations=100,
            num_trials=5,
            log_dir=LOG_PATH,
        )

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if num_workloads > 0:
        workloads = workloads[:num_workloads]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = (
                    trace.evaluation.performance.reference_latency_ms
                )
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms'] * 1000:.3f} µs", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


algo = os.getenv("ALGO", "fla-recurrent")
num_workloads = int(os.getenv("NUM_WORKLOADS", "0"))


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    entry_point = ALGO_ENTRY_POINTS[algo]
    print(f"Algorithm: {algo} (entry_point: {entry_point})")

    print("Packing solution from source files...")
    language = ALGO_LANGUAGES.get(algo)
    solution_path = pack_solution(entry_point=entry_point, name=algo, language=language)

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print(
        f"\nRunning benchmark on Modal B200 (num_workloads={num_workloads or 'all'})..."
    )
    results = run_benchmark.remote(solution, num_workloads=num_workloads)

    if not results:
        print("No results returned!")
        return

    print_results(results)
