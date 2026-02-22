"""
NVBench benchmark on Modal B200 GPUs.

Reuses the shared image from modal_config and the benchmark logic
from bench_nvbench. Mounts solution/ and scripts/ so imports work
on the remote.

Usage:
    ALGO=fla-recurrent modal run -m scripts.bench_nvbench_modal
    ALGO=all modal run -m scripts.bench_nvbench_modal
"""

import os
import sys

import modal

from .modal_config import TRACE_SET_PATH, image as base_image, trace_volume

app = modal.App("nvbench-gdn")

image = (
    base_image
    .pip_install("cuda-bench")
    .env({"FIB_DATASET_PATH": TRACE_SET_PATH})
    .add_local_dir("solution", remote_path="/root/solution")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

algo = os.getenv("ALGO", "all")

ALGOS_AVAILABLE = ["fla-recurrent", "fi-baseline", "fla-tma", "cuda-v1"]


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    retries=2,
)
def run_nvbench(algo_names: list[str]):
    # Inside the Modal container, scripts/ and solution/ are mounted at /root/.
    # Add /root to sys.path so `scripts` and `solution` packages are importable.
    sys.path.insert(0, "/root")

    import cuda.bench as bench

    from scripts.bench_nvbench import gdn_decode

    b = bench.register(gdn_decode)
    b.add_string_axis("Algo", algo_names)
    bench.run_all_benchmarks(["nvbench_modal"])


@app.local_entrypoint()
def main():
    algo_names = ALGOS_AVAILABLE if algo == "all" else [algo]
    print(f"Running NVBench on Modal B200: {', '.join(algo_names)}")
    run_nvbench.remote(algo_names)
