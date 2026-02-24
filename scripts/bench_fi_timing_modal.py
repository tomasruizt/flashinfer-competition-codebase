"""
FlashInfer bench_gpu_time benchmark on Modal B200 GPUs.

Reuses the shared image from modal_config and the benchmark logic
from bench_fi_timing. Mounts solution/ and scripts/ so imports work
on the remote.

Usage:
    ALGO=fla-recurrent modal run -m scripts.bench_fi_timing_modal
    ALGO=all modal run -m scripts.bench_fi_timing_modal
    ALGO=fla-recurrent,cuda-v4 modal run -m scripts.bench_fi_timing_modal
"""

import os
import sys

import modal

from .modal_config import TRACE_SET_PATH, image, trace_volume
from .shared import resolve_algo_names

app = modal.App("fi-timing-gdn")

algo = os.getenv("ALGO", "all")


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    retries=2,
)
def run_fi_timing(algo_names: list[str]):
    sys.path.insert(0, "/root")

    from scripts.bench_fi_timing import run_benchmark

    run_benchmark(algo_names)


@app.local_entrypoint()
def main():
    algo_names = resolve_algo_names(algo)
    print(f"Running FlashInfer bench_gpu_time on Modal B200: {', '.join(algo_names)}")
    run_fi_timing.remote(algo_names)
