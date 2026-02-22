"""
Profile GDN decode kernel with NVIDIA Nsight Compute (ncu).

This script is meant to be launched by ncu, not run directly:
    sudo ncu --set full ... python -m scripts.profile_ncu

See `make ncu-fla` for the full command.
"""

import torch

from solution.cuda.binding import kernel_cuda, kernel_cuda_v1b, kernel_cuda_v2, kernel_cuda_v2b, kernel_cuda_v2c, kernel_cuda_v3
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
        choices=["fla-recurrent", "fla-tma", "fi-baseline", "cuda-v1", "cuda-v1b", "cuda-v2", "cuda-v2b", "cuda-v2c", "cuda-v3"],
    )
    args = parser.parse_args()

    kernel_fn = {
        "fla-recurrent": kernel_fla_recurrent,
        "fla-tma": kernel_fla_tma,
        "fi-baseline": kernel_fi_baseline,
        "cuda-v1": kernel_cuda,
        "cuda-v1b": kernel_cuda_v1b,
        "cuda-v2": kernel_cuda_v2,
        "cuda-v2b": kernel_cuda_v2b,
        "cuda-v2c": kernel_cuda_v2c,
        "cuda-v3": kernel_cuda_v3,
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
