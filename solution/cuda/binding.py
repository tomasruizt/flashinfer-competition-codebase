"""Python binding for CUDA GDN decode kernel via TVM FFI.

Used by local scripts (bench_nvbench, profile_ncu) to call the kernel directly.
The benchmark framework compiles kernel.cu itself via TVMFFIBuilder.
"""

import math
from pathlib import Path

import torch
import tvm_ffi
from tvm_ffi import register_global_func

_mod = None

_CUDA_SRC = str(Path(__file__).parent / "kernel.cu")


def _get_mod():
    global _mod
    if _mod is None:
        lib_path = tvm_ffi.cpp.build(
            name="gdn_decode_cuda_local",
            cuda_files=[_CUDA_SRC],
        )
        _mod = tvm_ffi.load_module(lib_path)
    return _mod


def _get_fn():
    return getattr(_get_mod(), "kernel_cuda")


@torch.no_grad()
def kernel_cuda(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entry point for CUDA GDN decode kernel."""
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    fn = _get_fn()
    fn(q, k, v, state, A_log, a, dt_bias, b, float(scale), output, new_state)


def _make_cuda_entry(symbol):
    """Create a DPS entry point that calls the given TVM FFI symbol."""
    @torch.no_grad()
    def entry(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
        if scale is None or scale == 0.0:
            scale_ = 1.0 / math.sqrt(q.shape[-1])
        else:
            scale_ = float(scale)
        fn = getattr(_get_mod(), symbol)
        fn(q, k, v, state, A_log, a, dt_bias, b, scale_, output, new_state)
    entry.__name__ = symbol
    return entry


kernel_cuda_v1b = _make_cuda_entry("kernel_cuda_v1b")
kernel_cuda_v1c = _make_cuda_entry("kernel_cuda_v1c")
kernel_cuda_v1d = _make_cuda_entry("kernel_cuda_v1d")
kernel_cuda_v2 = _make_cuda_entry("kernel_cuda_v2")
kernel_cuda_v2b = _make_cuda_entry("kernel_cuda_v2b")
kernel_cuda_v2c = _make_cuda_entry("kernel_cuda_v2c")
kernel_cuda_v3 = _make_cuda_entry("kernel_cuda_v3")

# Register in TVM global table (non-decorator form preserves kwargs support)
register_global_func("flashinfer.kernel_cuda", kernel_cuda)
register_global_func("flashinfer.kernel_cuda_v1b", kernel_cuda_v1b)
register_global_func("flashinfer.kernel_cuda_v1c", kernel_cuda_v1c)
register_global_func("flashinfer.kernel_cuda_v1d", kernel_cuda_v1d)
register_global_func("flashinfer.kernel_cuda_v2", kernel_cuda_v2)
register_global_func("flashinfer.kernel_cuda_v2b", kernel_cuda_v2b)
register_global_func("flashinfer.kernel_cuda_v2c", kernel_cuda_v2c)
register_global_func("flashinfer.kernel_cuda_v3", kernel_cuda_v3)
