"""Shared Modal infrastructure: image, volume, and constants."""

import modal

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
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
