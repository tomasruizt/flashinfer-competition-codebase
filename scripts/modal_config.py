"""Shared Modal infrastructure: image, volume, and constants."""

import os

import modal

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"
TRITON_CACHE_PATH = f"{TRACE_SET_PATH}/cache/triton"


def set_triton_cache():
    """Point Triton cache to the Modal volume so autotune results persist across runs.

    Triton ignores XDG_CACHE_HOME and reads TRITON_CACHE_DIR instead.
    """
    os.environ["TRITON_CACHE_DIR"] = TRITON_CACHE_PATH


base_image = (
    # image name taken from EVALUATION.md
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .uv_pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
        "flashinfer-python",
        "flash-linear-attention",
        "pandas",
        "cupti-python",
    )
    .env({"TRITON_PRINT_AUTOTUNING": "1"})
)

# Extended image with env vars and local mounts for benchmark scripts.
image = (
    base_image.env({"FIB_DATASET_PATH": TRACE_SET_PATH})
    .add_local_dir("solution", remote_path="/root/solution")
    .add_local_dir("scripts", remote_path="/root/scripts")
)
