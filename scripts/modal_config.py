"""Shared Modal infrastructure: image, volume, and constants."""

import modal

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "flashinfer-bench",
        "torch",
        "triton",
        "numpy",
        "flash-linear-attention",
        "flashinfer-python",
    )
    .env({"TRITON_PRINT_AUTOTUNING": "1"})
)
