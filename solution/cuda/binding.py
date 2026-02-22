"""Python binding for CUDA GDN decode kernel via TVM FFI.

Used by local scripts (bench_nvbench, profile_ncu) to call the kernel directly.
The benchmark framework compiles kernel.cu itself via TVMFFIBuilder.
"""

import math
from pathlib import Path

import torch
import tvm_ffi
from tvm_ffi import register_global_func

_fn = None

_CUDA_SRC = str(Path(__file__).parent / "kernel.cu")


def _get_fn():
    global _fn
    if _fn is None:
        lib_path = tvm_ffi.cpp.build(
            name="gdn_decode_cuda_local",
            cuda_files=[_CUDA_SRC],
        )
        mod = tvm_ffi.load_module(lib_path)
        _fn = getattr(mod, "kernel_cuda")
    return _fn


@torch.no_grad()
def kernel_cuda(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entry point for CUDA GDN decode kernel."""
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    fn = _get_fn()
    fn(q, k, v, state, A_log, a, dt_bias, b, float(scale), output, new_state)


# Register in TVM global table (non-decorator form preserves kwargs support)
register_global_func("flashinfer.kernel_cuda", kernel_cuda)
