"""
Benchmark GDN decode kernels using NVBench (NVIDIA's official kernel benchmarking tool).

Validates our timing methodology against an independent, NVIDIA-standard benchmark.
NVBench uses CUDA events on a dedicated stream with statistical convergence criteria.

Usage:
    python scripts/bench_nvbench.py --algo=fla-recurrent
    python scripts/bench_nvbench.py --algo=fi-baseline
    python scripts/bench_nvbench.py --algo=all
"""

import sys
from pathlib import Path

import cuda.bench as bench
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from profile_proton import load_workload_tensors
from solution.triton.kernel import (
    kernel_fi_baseline,
    kernel_fla_recurrent,
    kernel_fla_tma,
)


def as_torch_stream(cs: bench.CudaStream) -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(cs.addressof())


ALGOS = {
    "fla-recurrent": kernel_fla_recurrent,
    "fi-baseline": kernel_fi_baseline,
    "fla-tma": kernel_fla_tma,
}


def gdn_decode(state: bench.State):
    """Benchmark GDN decode kernels."""
    algo_name = state.get_string("Algo")
    kernel_fn = ALGOS[algo_name]

    tensors = load_workload_tensors()

    # Total bytes read/written for reporting bandwidth
    h_state = tensors["state"]
    state_bytes = h_state.nelement() * h_state.element_size()
    qkv_bytes = sum(
        tensors[k].nelement() * tensors[k].element_size() for k in ("q", "k", "v")
    )
    output_bytes = tensors["output"].nelement() * tensors["output"].element_size()

    state.add_global_memory_reads(state_bytes + qkv_bytes, column_name="Read")
    state.add_global_memory_writes(state_bytes + output_bytes, column_name="Write")

    def launcher(launch: bench.Launch):
        stream = as_torch_stream(launch.get_stream())
        with torch.cuda.stream(stream):
            kernel_fn(**tensors)

    # batched=False enables L2 cache flushing between iterations
    state.exec(launcher, batched=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NVBench GDN decode benchmark")
    parser.add_argument(
        "--algo",
        choices=list(ALGOS) + ["all"],
        default="all",
    )
    args, remaining = parser.parse_known_args()

    algo_names = list(ALGOS) if args.algo == "all" else [args.algo]

    b = bench.register(gdn_decode)
    b.add_string_axis("Algo", algo_names)
    bench.run_all_benchmarks(["bench_nvbench"] + remaining)


if __name__ == "__main__":
    main()
