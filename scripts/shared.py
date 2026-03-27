"""Shared constants and utilities used across scripts."""

from pathlib import Path
from typing import Literal

DefinitionName = Literal[
    "gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill_qk4_v8_d128_k_last",
]


class DEFS:
    DECODE = "gdn_decode_qk4_v8_d128_k_last"
    PREFILL = "gdn_prefill_qk4_v8_d128_k_last"


PROJECT_ROOT = Path(__file__).parent.parent

ALGO_ENTRY_POINTS = {
    "fla-recurrent": "kernel.py::kernel_fla_recurrent",
    "fla-tma": "kernel.py::kernel_fla_tma",
    "pt-reference": "kernel.py::kernel_pt_reference",
    "pt-compiled": "kernel.py::kernel_pt_compiled",
    "fi-baseline": "kernel.py::kernel_fi_baseline",
    "cuda-v1": "kernel.cu::kernel_cuda",
    "cuda-v4": "kernel.cu::kernel_cuda_v4",
    "prefill-reference": "prefill_kernel.py::kernel_prefill_reference",
    "prefill-fla-chunk": "prefill_kernel.py::kernel_prefill_fla_chunk",
}

ALGO_LANGUAGES = {
    "cuda-v1": "cuda",
    "cuda-v4": "cuda",
}

ALGO_NO_DPS = {"pt-reference", "pt-compiled", "prefill-reference"}

CUDA_ALGOS = list(ALGO_LANGUAGES.keys())
NON_CUDA_ALGOS = [a for a in ALGO_ENTRY_POINTS if a not in ALGO_LANGUAGES]
ALL_ALGOS = NON_CUDA_ALGOS + CUDA_ALGOS


def resolve_algo_names(algo_str: str) -> list[str]:
    """Resolve algo string to list of algo names.

    Accepts "all", "cuda-all", or a comma-separated list of algo names.
    """
    if algo_str == "all":
        return ALL_ALGOS
    if algo_str == "cuda-all":
        return CUDA_ALGOS
    return [a.strip() for a in algo_str.split(",")]


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
    from solution.triton.prefill_kernel import (
        kernel_prefill_fla_chunk,
        kernel_prefill_reference,
    )

    return {
        "pt-reference": kernel_pt_reference,
        "pt-compiled": kernel_pt_compiled,
        "fi-baseline": kernel_fi_baseline,
        "fla-recurrent": kernel_fla_recurrent,
        "fla-tma": kernel_fla_tma,
        "cuda-v1": kernel_cuda,
        "cuda-v4": kernel_cuda_v4,
        "prefill-reference": kernel_prefill_reference,
        "prefill-fla-chunk": kernel_prefill_fla_chunk,
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
    parser.add_argument(
        "--workload-id",
        type=str,
        default=None,
        help="UUID prefix to filter a specific workload (e.g. 'a5714b69')",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default=DEFS.DECODE,
        help="Definition name (default: gdn_decode_qk4_v8_d128_k_last)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timing iterations (default: 100)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials (default: 5)",
    )
    return parser.parse_args()
