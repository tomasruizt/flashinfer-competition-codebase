"""
Profile GDN decode kernel with Proton.

Usage:
    python -m scripts.profile_proton                  # timeline trace (instrumentation)
    python -m scripts.profile_proton --op-measure     # op measurement (instrumentation)
    python -m scripts.profile_proton --pcsampling     # line-by-line PC sampling

View results:
    Timeline:    chrome://tracing -> load profiles/gdn_decode.chrome_trace
    Op measure:  proton-viewer -m normalized_cycles profiles/gdn_decode.hatchet
    PC sampling: proton-viewer -m num_samples/% profiles/gdn_decode_lines.hatchet -i profile --print-sorted
"""

import argparse
import os

import torch
import torch._dynamo
import triton.profiler as proton
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import TraceSet

from .shared import DEFS, PROJECT_ROOT, DefinitionName


def main():
    parser = argparse.ArgumentParser(description="Profile GDN decode with Proton")
    parser.add_argument(
        "--op-measure",
        action="store_true",
        help="Op measurement mode (instrumentation)",
    )
    parser.add_argument(
        "--pcsampling", action="store_true", help="Line-by-line PC sampling (CUPTI)"
    )
    parser.add_argument(
        "--iters", type=int, default=1, help="Number of profiled iterations"
    )
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

    tensors = load_workload_tensors(DEFS.DECODE)

    print("Warming up...")
    for _ in range(3):
        kernel_fla_recurrent(**tensors)
    torch.cuda.synchronize()

    # Instrumentation modes start after warmup (pcsampling already started above)
    if not args.pcsampling:
        name = str(profiles_dir / "gdn_decode")
        data = "tree" if args.op_measure else "trace"
        proton.start(
            name, data=data, backend="instrumentation", mode=proton.mode.Default()
        )

    print(f"Profiling ({args.iters} iters)...")
    with proton.scope("profile"):
        for _ in range(args.iters):
            kernel_fla_recurrent(**tensors)
        torch.cuda.synchronize()

    proton.finalize()
    print("Done.")


def load_workload_tensors(
    def_name: DefinitionName, device="cuda", trace_set_path=None, workload_idx=0
):
    trace_set = TraceSet.from_path(trace_set_path)
    definition = trace_set.definitions[def_name]
    workload = trace_set.workloads[def_name][workload_idx].workload

    safe_tensors = load_safetensors(definition, workload, trace_set.root)
    input_list = gen_inputs(
        definition, workload, device=device, safe_tensors=safe_tensors
    )
    input_names = list(definition.inputs.keys())
    inputs = dict(zip(input_names, input_list))

    if "decode" in def_name:
        _add_dps_decode(inputs, device)
    else:
        _add_dps_prefill(inputs, device)

    return inputs


def _add_dps_decode(inputs, device):
    B, T, _, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]
    inputs["output"] = torch.empty(B, T, HV, V, dtype=torch.bfloat16, device=device)
    inputs["new_state"] = torch.empty(B, HV, V, K, dtype=torch.float32, device=device)


def _add_dps_prefill(inputs, device):
    total_seq_len, HV, V = inputs["v"].shape
    K = inputs["q"].shape[2]
    num_seqs = inputs["cu_seqlens"].shape[0] - 1
    inputs["output"] = torch.empty(
        total_seq_len, HV, V, dtype=torch.bfloat16, device=device
    )
    inputs["new_state"] = torch.empty(
        num_seqs, HV, V, K, dtype=torch.float32, device=device
    )


if __name__ == "__main__":
    main()
