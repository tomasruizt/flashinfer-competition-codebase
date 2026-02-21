"""
NVBench benchmark on Modal B200 GPUs.

Reuses the shared image from modal_config.py and the benchmark logic
from bench_nvbench.py. Mounts solution/ and scripts/ so imports work
on the remote.

Usage:
    ALGO=fla-recurrent modal run scripts/bench_nvbench_modal.py
    ALGO=all modal run scripts/bench_nvbench_modal.py
"""

import os
import sys
from pathlib import Path

# Make sibling scripts importable (locally and on Modal remote)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/root/scripts")

import modal

from modal_config import TRACE_SET_PATH, image as base_image, trace_volume

app = modal.App("nvbench-gdn")

image = (
    base_image
    .pip_install("cuda-bench")
    .env({"FIB_DATASET_PATH": TRACE_SET_PATH})
    .add_local_dir("solution", remote_path="/root/solution")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

algo = os.getenv("ALGO", "all")

ALGOS_AVAILABLE = ["fla-recurrent", "fi-baseline", "fla-tma"]


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
    retries=2,
)
def run_nvbench(algo_names: list[str]):
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    import cuda.bench as bench

    from bench_nvbench import gdn_decode

    b = bench.register(gdn_decode)
    b.add_string_axis("Algo", algo_names)
    bench.run_all_benchmarks(["nvbench_modal"])


@app.local_entrypoint()
def main():
    algo_names = ALGOS_AVAILABLE if algo == "all" else [algo]
    print(f"Running NVBench on Modal B200: {', '.join(algo_names)}")
    run_nvbench.remote(algo_names)
