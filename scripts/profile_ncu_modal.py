"""
Run NCU profiling on Modal B200.

Usage:
    ALGO=fla-recurrent WORKLOAD_IDX=53 make ncu-modal
    ALGO=fla-recurrent WORKLOAD_IDX=53 NCU_MODE=profile make ncu-modal
"""

import os
import subprocess
import sys

import modal

from .modal_config import (
    TRACE_SET_PATH,
    image as base_image,
    set_triton_cache,
    trace_volume,
)

app = modal.App("ncu-gdn")

algo = os.getenv("ALGO", "fla-recurrent")
workload_idx = int(os.getenv("WORKLOAD_IDX", "0"))
ncu_mode = os.getenv("NCU_MODE", "export")

KERNEL_NAMES = {
    "fla-recurrent": "fused_recurrent_gated_delta_rule_fwd_kernel",
    "fi-baseline": "regex:kernel_cutlass_gdn_decode",
    "cuda-v1": "gdn_decode_kernel",
    "cuda-v4": "gdn_decode_v4_kernel",
}


@app.function(
    image=base_image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_ncu(algo: str, workload_idx: int, mode: str):
    set_triton_cache()
    sys.path.insert(0, "/root")

    kernel_name = KERNEL_NAMES[algo]
    cmd = [
        "ncu",
        "--set",
        "full",
        "--kernel-name",
        kernel_name,
        "--launch-skip",
        "3",
        "--launch-count",
        "1",
    ]
    if mode == "profile":
        ncu_dir = f"{TRACE_SET_PATH}/ncu-rep"
        os.makedirs(ncu_dir, exist_ok=True)
        out_path = f"{ncu_dir}/{algo}-widx{workload_idx}"
        cmd += ["--import-source", "yes", "-fo", out_path]

    cmd += [
        sys.executable,
        "-m",
        "scripts.profile_ncu",
        f"--algo={algo}",
        f"--workload-idx={workload_idx}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if mode == "profile":
        trace_volume.commit()


@app.local_entrypoint()
def main():
    print(f"NCU on Modal B200: {algo} workload_idx={workload_idx} mode={ncu_mode}")
    run_ncu.remote(algo, workload_idx, ncu_mode)
