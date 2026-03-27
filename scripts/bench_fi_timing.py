"""
Benchmark GDN kernels using FlashInfer's bench_gpu_time.

Uses the same timing methodology as FlashInfer's own developers (PR #2498),
giving an apples-to-apples comparison with their published numbers.
Default: CUDA events with L2 cache flushing, auto-computed warmup/iters
from 25ms/100ms targets.

Usage:
    python -m scripts.bench_fi_timing --algo=fla-recurrent
    python -m scripts.bench_fi_timing --algo=fla-recurrent,cuda-v4
    python -m scripts.bench_fi_timing --algo=all
    python -m scripts.bench_fi_timing --algo=prefill-fla-chunk --definition=gdn_prefill_qk4_v8_d128_k_last
"""

import statistics

import pandas as pd
from flashinfer.testing import bench_gpu_time

from .profile_proton import load_workload_tensors
from .shared import DEFS, DefinitionName, load_algo_functions, resolve_algo_names


def run_benchmark(
    algo_names: list[str],
    def_name: DefinitionName,
    iters: int = None,
    cupti: bool = True,
):
    """Run bench_gpu_time for the given algo names and print results."""
    algos = load_algo_functions()

    for name in algo_names:
        if name not in algos:
            raise ValueError(f"Unknown algo '{name}'. Available: {list(algos)}")

    rows = []
    for name in algo_names:
        tensors = load_workload_tensors(def_name)
        kernel_fn = algos[name]
        times_ms = bench_gpu_time(
            fn=lambda kf=kernel_fn, t=tensors: kf(**t),
            cold_l2_cache=True,
            enable_cupti=cupti,
            repeat_iters=iters,
        )
        times_us = [t * 1000.0 for t in times_ms]
        med = statistics.median(times_us)
        rows.append(
            {
                "algo": name,
                "median_us": med,
                "min_us": min(times_us),
                "max_us": max(times_us),
                "iters": len(times_us),
            }
        )
        print(f"{name}: median={med:.2f} us  (n={len(times_us)})")

    df = pd.DataFrame(rows).sort_values("median_us").reset_index(drop=True)
    print("\n" + df.to_string(index=False, float_format="%.2f"))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark GDN kernels using FlashInfer bench_gpu_time"
    )
    parser.add_argument(
        "--algo",
        default="all",
        help="Algorithm(s) to benchmark. Comma-separated or 'all' (default: all)",
    )
    parser.add_argument(
        "--definition",
        default=DEFS.DECODE,
        help="Definition name (default: gdn_decode_qk4_v8_d128_k_last)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of repeat iterations (default: auto from 100ms target)",
    )
    parser.add_argument(
        "--use-cupti",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
        help="Use CUPTI timing (default: True, set False when running under nsys)",
    )
    args = parser.parse_args()

    run_benchmark(
        resolve_algo_names(args.algo),
        def_name=args.definition,
        iters=args.iters,
        cupti=args.use_cupti,
    )


if __name__ == "__main__":
    main()
