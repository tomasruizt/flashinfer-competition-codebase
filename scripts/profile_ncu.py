"""
Profile GDN decode kernel with NVIDIA Nsight Compute (ncu).

This script is meant to be launched by ncu, not run directly:
    sudo ncu --set full ... python -m scripts.profile_ncu

See `make ncu-fla` for the full command.
"""

import torch

from solution.triton.kernel import (
    kernel_fla_recurrent,
    kernel_fla_tma,
    kernel_fi_baseline,
)
from .profile_proton import load_workload_tensors


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="fla-recurrent",
        choices=["fla-recurrent", "fla-tma", "fi-baseline"],
    )
    args = parser.parse_args()

    kernel_fn = {
        "fla-recurrent": kernel_fla_recurrent,
        "fla-tma": kernel_fla_tma,
        "fi-baseline": kernel_fi_baseline,
    }[args.algo]
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
