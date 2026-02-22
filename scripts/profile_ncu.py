"""
Profile GDN decode kernel with NVIDIA Nsight Compute (ncu).

This script is meant to be launched by ncu, not run directly:
    sudo ncu --set full ... python -m scripts.profile_ncu

See `make ncu-fla` for the full command.
"""

import torch

from .profile_proton import load_workload_tensors
from .shared import load_algo_functions


def main():
    import argparse

    algo_fns = load_algo_functions()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="fla-recurrent",
        choices=list(algo_fns),
    )
    args = parser.parse_args()

    kernel_fn = algo_fns[args.algo]
    tensors = load_workload_tensors()

    print("Warming up...")
    for _ in range(3):
        kernel_fn(**tensors)
    torch.cuda.synchronize()

    # Flush L2 cache before profiled run (matches flashinfer_bench.bench.timing)
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    cache.zero_()
    torch.cuda.synchronize()

    print("Profiling...")
    kernel_fn(**tensors)
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
