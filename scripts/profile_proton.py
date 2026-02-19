"""
Profile GDN decode kernel with Proton intra-kernel instrumentation.

Usage:
    .venv/bin/python scripts/profile_proton.py                  # timeline trace
    .venv/bin/python scripts/profile_proton.py --op-measure     # op measurement (hatchet)

View results:
    Timeline:  chrome://tracing  -> load gdn_decode.chrome_trace
    Op measure: proton-viewer -m normalized_cycles gdn_decode.hatchet
"""

import argparse
import sys
from pathlib import Path

import torch
import torch._dynamo
import triton.profiler as proton
import triton.profiler.language as pl
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import TraceSet

# Disable torch.compile so Triton kernels run directly (proton scopes need this —
# torch.compile serializes kernel source and loses the `pl` import)
torch._dynamo.config.disable = True

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solution.triton.kernel import kernel_fla_recurrent  # noqa: E402

DEF_NAME = "gdn_decode_qk4_v8_d128_k_last"


def main():
    parser = argparse.ArgumentParser(description="Profile GDN decode with Proton")
    parser.add_argument("--op-measure", action="store_true", help="Op measurement mode (default: timeline trace)")
    parser.add_argument("--iters", type=int, default=1, help="Number of profiled iterations")
    args = parser.parse_args()

    print("Loading workload from dataset...")
    tensors = load_workload_tensors()
    print(f"Decode tensors: B={tensors['q'].shape[0]}, T=1, HV=8, K=V=128")

    pl.enable_semantic("triton")

    # Warmup outside profiling (triggers torch.compile + Triton compilation)
    print("Warming up...")
    for _ in range(3):
        kernel_fla_recurrent(**tensors)
    torch.cuda.synchronize()

    # Profile
    profiles_dir = PROJECT_ROOT / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    name = str(profiles_dir / "gdn_decode")
    mode = proton.mode.Default()

    if args.op_measure:
        print(f"Starting op measurement -> {name}.hatchet")
        proton.start(name, backend="instrumentation", mode=mode)
    else:
        print(f"Starting timeline trace -> {name}.chrome_trace")
        proton.start(name, data="trace", backend="instrumentation", mode=mode)

    for _ in range(args.iters):
        kernel_fla_recurrent(**tensors)
    torch.cuda.synchronize()

    proton.finalize()
    print("Done. Profiling output written.")


def load_workload_tensors(device="cuda"):
    trace_set = TraceSet.from_path()
    definition = trace_set.definitions[DEF_NAME]
    workload = trace_set.workloads[DEF_NAME][0].workload

    safe_tensors = load_safetensors(definition, workload, trace_set.root)
    input_list = gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)
    input_names = list(definition.inputs.keys())
    inputs = dict(zip(input_names, input_list))

    # Add DPS output buffers
    B, T, _, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]
    inputs["output"] = torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=device)
    inputs["new_state"] = torch.empty(B, HV, V, K, dtype=torch.float32, device=device)

    return inputs


if __name__ == "__main__":
    main()
