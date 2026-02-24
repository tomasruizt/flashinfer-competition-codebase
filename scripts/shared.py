"""Shared constants and utilities used across scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

ALGO_ENTRY_POINTS = {
    "fla-recurrent": "kernel.py::kernel_fla_recurrent",
    "fla-tma": "kernel.py::kernel_fla_tma",
    "pt-reference": "kernel.py::kernel_pt_reference",
    "pt-compiled": "kernel.py::kernel_pt_compiled",
    "fi-baseline": "kernel.py::kernel_fi_baseline",
    "cuda-v1": "kernel.cu::kernel_cuda",
    "cuda-v4": "kernel.cu::kernel_cuda_v4",
}

ALGO_LANGUAGES = {
    "cuda-v1": "cuda",
    "cuda-v4": "cuda",
}

ALGO_NO_DPS = {"pt-reference", "pt-compiled"}


def load_algo_functions() -> dict:
    """Lazily import and return a mapping from algo name to callable kernel function.

    Imports are deferred because they pull in torch, triton, and CUDA bindings,
    which not every script needs.
    """
    from solution.cuda.binding import (
        kernel_cuda,
        kernel_cuda_v4,
    )
    from solution.triton.kernel import (
        kernel_fi_baseline,
        kernel_fla_recurrent,
        kernel_fla_tma,
        kernel_pt_compiled,
        kernel_pt_reference,
    )

    return {
        "pt-reference": kernel_pt_reference,
        "pt-compiled": kernel_pt_compiled,
        "fi-baseline": kernel_fi_baseline,
        "fla-recurrent": kernel_fla_recurrent,
        "fla-tma": kernel_fla_tma,
        "cuda-v1": kernel_cuda,
        "cuda-v4": kernel_cuda_v4,
    }


def parse_args():
    """Parse command-line arguments common to benchmark scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack and benchmark GDN kernel")
    parser.add_argument(
        "--algo",
        choices=list(ALGO_ENTRY_POINTS.keys()),
        default="fla-recurrent",
        help="Algorithm to benchmark (default: fla-recurrent)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)",
    )
    parser.add_argument(
        "-n",
        "--num-workloads",
        type=int,
        default=0,
        help="Number of workloads to run (default: 0 = all)",
    )
    return parser.parse_args()
