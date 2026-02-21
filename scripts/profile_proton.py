"""
Profile GDN decode kernel with Proton.

Usage:
    python scripts/profile_proton.py                  # timeline trace (instrumentation)
    python scripts/profile_proton.py --op-measure     # op measurement (instrumentation)
    python scripts/profile_proton.py --pcsampling     # line-by-line PC sampling

View results:
    Timeline:    chrome://tracing -> load profiles/gdn_decode.chrome_trace
    Op measure:  proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet
    PC sampling: proton-viewer -m num_samples/% profiles/gdn_decode_lines.hatchet -i profile --print-sorted
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch._dynamo
import triton.profiler as proton
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import TraceSet

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEF_NAME = "gdn_decode_qk4_v8_d128_k_last"


def main():
    parser = argparse.ArgumentParser(description="Profile GDN decode with Proton")
    parser.add_argument("--op-measure", action="store_true", help="Op measurement mode (instrumentation)")
    parser.add_argument("--pcsampling", action="store_true", help="Line-by-line PC sampling (CUPTI)")
    parser.add_argument("--iters", type=int, default=1, help="Number of profiled iterations")
    args = parser.parse_args()

    torch._dynamo.config.disable = True
    profiles_dir = PROJECT_ROOT / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    if args.pcsampling:
        # hook="triton" must be called BEFORE kernel compilation (first JIT)
        name = str(profiles_dir / "gdn_decode_lines")
        proton.start(name, hook="triton", backend="cupti", mode="pcsampling")
    else:
        # Enable pl.scope() annotations in the Triton kernel
        os.environ["PROTON_PROFILE"] = "1"
        import triton.profiler.language as pl
        pl.enable_semantic("triton")

    from solution.triton.kernel import kernel_fla_recurrent

    tensors = load_workload_tensors()

    print("Warming up...")
    for _ in range(3):
        kernel_fla_recurrent(**tensors)
    torch.cuda.synchronize()

    # Instrumentation modes start after warmup (pcsampling already started above)
    if not args.pcsampling:
        name = str(profiles_dir / "gdn_decode")
        data = "tree" if args.op_measure else "trace"
        proton.start(name, data=data, backend="instrumentation", mode=proton.mode.Default())

    print(f"Profiling ({args.iters} iters)...")
    with proton.scope("profile"):
        for _ in range(args.iters):
            kernel_fla_recurrent(**tensors)
        torch.cuda.synchronize()

    proton.finalize()
    print("Done.")


def load_workload_tensors(device="cuda", trace_set_path=None):
    trace_set = TraceSet.from_path(trace_set_path)
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
