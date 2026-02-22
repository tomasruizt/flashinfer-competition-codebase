"""Shared constants and utilities used across scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ALGO_ENTRY_POINTS = {
    "fla-recurrent": "kernel.py::kernel_fla_recurrent",
    "fla-tma": "kernel.py::kernel_fla_tma",
    "pt-reference": "kernel.py::kernel_pt_reference",
    "fi-baseline": "kernel.py::kernel_fi_baseline",
    "cuda-v1": "kernel.cu::kernel_cuda",
    "cuda-v1b": "kernel.cu::kernel_cuda_v1b",
    "cuda-v2": "kernel.cu::kernel_cuda_v2",
    "cuda-v2b": "kernel.cu::kernel_cuda_v2b",
    "cuda-v2c": "kernel.cu::kernel_cuda_v2c",
    "cuda-v3": "kernel.cu::kernel_cuda_v3",
}

ALGO_LANGUAGES = {
    "cuda-v1": "cuda",
    "cuda-v1b": "cuda",
    "cuda-v2": "cuda",
    "cuda-v2b": "cuda",
    "cuda-v2c": "cuda",
    "cuda-v3": "cuda",
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
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    parser.add_argument(
        "-n", "--num-workloads",
        type=int,
        default=0,
        help="Number of workloads to run (default: 0 = all)",
    )
    return parser.parse_args()
