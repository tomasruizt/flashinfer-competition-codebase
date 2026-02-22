# FlashInfer AI Kernel Generation Contest - GDN Track

## Writing Style
- **Never use em dashes** (the "—" character). Use parentheses, commas, colons, semicolons, or split into separate sentences instead.

## Team
- Team name: **lmu-css**
- Track: **gated_delta_net** (Gated Delta Net)

## Project Structure
- `config.toml` — Solution metadata and build config. `definition` must match the exact definition name (e.g. `gdn_decode_qk4_v8_d128_k_last`), not the track name.
- `solution/triton/kernel.py` — Triton/Python kernel implementation. Entry point is a regular Python function (not necessarily `@triton.jit`).
- `solution/cuda/kernel.cu` — CUDA C++ kernel with TVM FFI export (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`).
- `solution/cuda/binding.py` — Local Python binding via `tvm_ffi.cpp.build()` + `@register_global_func`.
- `scripts/` — Python package (has `__init__.py`). All scripts run as modules: `python -m scripts.X`.
  - `shared.py` — Shared constants: `ALGO_ENTRY_POINTS`, `ALGO_LANGUAGES`, `PROJECT_ROOT`, `parse_args()`.
  - `pack_solution.py` — Packs solution into `solution.json`.
  - `run_local.py` — Local benchmark runner.
  - `run_modal.py` — Cloud benchmark (B200 GPUs via Modal).
  - `modal_config.py` — Shared Modal infrastructure (image, volume, `TRACE_SET_PATH`).
  - `profile_proton.py` — Proton intra-kernel profiling. Also exports `load_workload_tensors()`.
  - `profile_ncu.py` — NCU profiling (launched by `ncu`, not run directly).
  - `bench_nvbench.py` — NVBench timing validation.
  - `bench_nvbench_modal.py` — NVBench on Modal B200.
  - `log_speedups.py` — Parse bench logs into `findings/speedups.csv`.

### Import conventions
- `scripts/` and `solution/` are both Python packages (have `__init__.py`).
- All scripts are invoked as modules: `python -m scripts.run_local`, `modal run -m scripts.run_modal`.
- Within `scripts/`, use **relative imports**: `from .shared import ALGO_ENTRY_POINTS`.
- For kernel imports: `from solution.triton.kernel import ...` (works because `-m` adds CWD to sys.path).
- No `sys.path` manipulation, except inside Modal remote functions (container mounts at `/root/`).
- To add a new algo: add one entry to `ALGO_ENTRY_POINTS` in `shared.py`, add one wrapper function in `kernel.py`. For non-Triton algos, also add an entry to `ALGO_LANGUAGES`.

## Environment
- Venv: `.venv/` in project root
- Packages: `flashinfer-bench`, `modal`, `torch`, `triton`
- Dataset: `~/code/mlsys26-contest` (env var `FIB_DATASET_PATH`, set in `~/.bashrc`)

## Config Notes
- `entry_point` format: `kernel.py::kernel_fla_recurrent` (Triton) or `kernel.cu::kernel_cuda` (CUDA TVM FFI)
- `definition` must be the exact definition name from the dataset (e.g. `gdn_decode_qk4_v8_d128_k_last`), not the track name (`gated_delta_net`)
- DPS (Destination Passing Style) is default: kernel receives pre-allocated output tensors as extra args
- `language="triton"` uses PythonBuilder (imports .py, calls function). `language="cuda"` uses TVMFFIBuilder (compiles .cu, exports via TVM FFI). The `ALGO_LANGUAGES` dict in `shared.py` overrides the default from config.toml.

## What is GDN?

GDN (Gated Delta Net) is an **alternative to softmax attention** for LLMs. It replaces the O(n) per-token attention decode with an O(1) recurrent state update. Used in production: Qwen3-Next-80B (75% GDN layers), Kimi Linear-48B.

### GDN vs Attention at decode time
|                      | Compute per token                         | Memory                             |
| -------------------- | ----------------------------------------- | ---------------------------------- |
| **Causal Attention** | O(n) — dot product with all n cached keys | O(n) — KV cache grows with context |
| **GDN**              | O(1) — fixed 128x128 state matrix ops     | O(1) — constant state size         |

### Where GDN sits in the transformer
```
x = x + gdn_layer(norm(x))    # replaces attention sublayer
x = x + mlp(norm(x))          # FFN sublayer unchanged
```

### Full GDN layer (our kernel is the middle part)
```python
# --- Input projections (outside our kernel) ---
q = x @ W_q                    # [B, 1, num_q_heads, K]
k = l2_normalize(x @ W_k)     # [B, 1, num_k_heads, K]  (L2-normalized!)
v = x @ W_v                    # [B, 1, num_v_heads, V]
a = x @ W_a                    # [B, 1, num_v_heads]
b = x @ W_b                    # [B, 1, num_v_heads]

# --- Our kernel (the competition part) ---
g     = exp(-exp(A_log) * softplus(a + dt_bias))   # global decay ∈ (0,1)
beta  = sigmoid(b)                                  # update gate ∈ (0,1)
S_new = g * S - k^T @ (k @ S) + k^T @ (beta * v + (1-beta) * (k @ S))
out   = scale * q @ S_new

# --- Output projection (outside our kernel) ---
out = reshape(out) @ W_o       # back to hidden dim
```

### Decode kernel hot loop — per head operations
All ops are on 128-dim vectors and a 128x128 state matrix:
```
g_val                          # scalar         — global decay
beta_val                       # scalar         — update gate
h_state                        # [K=128, V=128] — state (transposed from [V,K] storage)

old_state = g_val * h_state    # [K, V]         — scale: decayed state
old_v = k @ old_state          # [K]@[K,V]->[V] — matvec: current value at key k
new_v = beta*v + (1-beta)*old_v # [V]           — blend: new/old value
state_remove = k^T @ old_v     # [K,1]@[1,V]->[K,V] — outer product: erase old
state_update = k^T @ new_v     # [K,1]@[1,V]->[K,V] — outer product: write new
h_state = old_state - state_remove + state_update  # [K,V] — updated state
output = scale * (q @ h_state) # [K]@[K,V]->[V] — matvec: read from state
```
Summary: **2 matvecs** (k@S, q@S), **2 outer products** (k^T@old_v, k^T@new_v), plus elementwise ops.

### Gate computation: why so complex?
```
g = exp(-exp(A_log) * softplus(a + dt_bias))
```
- Inherited from SSM literature (S4 → Mamba → GDN). This is the **exact discretization** of continuous-time decay `dS/dt = -A*S`.
- `A_log` (log-space) lets the model learn decay rates spanning orders of magnitude.
- `softplus(a + dt_bias)` is a learned "timestep" — positive, smooth, input-dependent.
- `exp(-positive * positive)` guarantees g ∈ (0,1) by construction.
- Each of the 8 heads learns its own decay rate (A_log has shape [8]).

### GVA (Grouped Value Attention)
From Qwen3-Next-80B with TP=4:
- Full model: 16 q/k heads, 32 v heads → after TP=4: **4 q/k heads, 8 v heads**
- Each q/k head is **repeat-interleaved** to serve 2 v heads:
  ```
  v_head 0,1 ← q/k head 0
  v_head 2,3 ← q/k head 1
  v_head 4,5 ← q/k head 2
  v_head 6,7 ← q/k head 3
  ```
- Analogous to GQA in standard transformers, but inverted (GQA: more q than kv; GVA: more v than qk)
- The loop runs over num_v_heads=8 because q/k are expanded before the loop

### State layout: "k-last"
- Storage: `[B, H=8, V=128, K=128]` — "k-last" means K dimension is last
- Kernel works directly in k-last layout — no transposes needed
- Pointer math for element (k, v): `offset = v * K + k` (K is contiguous/inner dim)

## GDN Track: Two Kernels
Each kernel is a separate definition, needs a separate `config.toml` definition entry:

### 1. Decode: `gdn_decode_qk4_v8_d128_k_last`
- Single-token generation (seq_len=1)
- Shapes: `q/k: [B, 1, 4, 128]` bf16, `v: [B, 1, 8, 128]` bf16, `state: [B, 8, 128, 128]` f32
- Scalar inputs: `A_log: [8]` f32, `dt_bias: [8]` f32, `a: [B, 1, 8]` bf16, `b: [B, 1, 8]` bf16, `scale: f32`
- Outputs (DPS): `output [B, 1, 8, 128]` bf16, `new_state [B, 8, 128, 128]` f32
- All 20 workloads use batch_size=1
- Memory-bound regime

### 2. Prefill: `gdn_prefill_qk4_v8_d128_k_last`
- Variable-length batched sequences (uses `cu_seqlens`)
- Shapes: `q/k: [total_seq_len, 4, 128]`, `v: [total_seq_len, 8, 128]`, `state: [num_seqs, 8, 128, 128]`
- Inputs: q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
- Outputs (DPS): `output [total_seq_len, 8, 128]` bf16, `new_state [num_seqs, 8, 128, 128]` f32
- Compute-bound regime (chunkwise parallelism, WY factorization)

## Benchmark Methodology
- `run_local.py` uses: warmup_runs=3, iterations=100, num_trials=5
- Default BenchmarkConfig: warmup=10, iterations=50, num_trials=3
- **L2 cache flushed** before every iteration (256 MB zero buffer)
- **Tensor args cloned** each iteration (clone time excluded from measurement)
- Timing via `torch.cuda.Event(enable_timing=True)` with proper synchronization
- Reported latency = mean across iterations, speedup = ref_latency / sol_latency
- Correctness: atol=1e-2, rtol=1e-2
- 20 workloads per definition, all with batch_size=1 for decode

## Baseline Performance (local GPU, PyTorch reference)
- Decode: ~1.4ms, 0.93-0.99x vs FlashInfer baseline (reference is slightly slower)

## Reference Implementations
- Located in definition JSON files as the `"reference"` field
- `~/code/mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json`
- `~/code/mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json`
- Pure PyTorch with nested for-loops (per-batch, per-head, per-timestep)

## Existing Optimized Implementations (for reference)
- **fla-org/flash-linear-attention**: Primary Triton kernel library (`fla/ops/gated_delta_rule`)
- **NVlabs/GatedDeltaNet**: Official ICLR 2025 implementation (wraps FLA kernels)
- Research-grade Triton, not tuned for Blackwell — optimization headroom exists
- Full list with links and papers: `findings/research.md` under "Existing Implementations"

## Multi-Algo Benchmarking

### Entry point dispatch via `--algo` flag
```bash
python -m scripts.run_local --algo=fla-recurrent    # default
python -m scripts.run_local --algo=pt-reference      # compiled PyTorch reference
```
- Each algo maps to a separate DPS entry point function in `kernel.py` (e.g. `kernel_fla_recurrent`, `kernel_pt_reference`)
- `run_local.py` passes the entry point string to `pack_solution(entry_point=...)`, overriding config.toml
- `pack_solution()` also accepts `name=` to set the solution name per algo

### torch.compile
- `kernel_pt_reference` uses `@torch.compile(fullgraph=True)` — compiler unrolls Python loops (B=1, num_heads=8)
- `kernel_fla_recurrent` does NOT use torch.compile — gate math is fused into the Triton kernel and state transposes are eliminated, so there's nothing left to compile

### Trace file structure
- Trace output path: `{FIB_DATASET_PATH}/traces/{op_type}/{definition_name}.jsonl`
- Path is keyed by **definition name** only — all solutions for the same definition append to the same JSONL file
- Each JSON line has a `"solution"` field to distinguish between algos
- To separate algos in the trace file, use different solution names via `pack_solution(name=...)`

### Performance (RTX 3090, after benchmark timing fix)
| Algo                    | Latency  | Speedup vs reference |
| ----------------------- | -------- | -------------------- |
| pt-reference (eager)    | ~1.4 ms  | ~1.0x                |
| pt-reference (compiled) | ~0.73 ms | ~1.8x                |
| fla-recurrent           | ~4.3 µs  | ~280x                |
| cuda-v1                 | ~4.7 µs  | ~255x                |
| fi-baseline             | ~4.8 µs  | ~250x                |
| fla-tma                 | ~7.5 µs  | ~161x                |

### Performance (B200 via Modal)
| Algo          | FI-bench  | NVBench  | Speedup vs reference |
| ------------- | --------- | -------- | -------------------- |
| fla-recurrent | ~0.037 ms | 7.1 µs   | ~32x                 |
| cuda-v1       | ~0.012 ms | 7.6 µs   | ~99x                 |
| fi-baseline   | ~0.017 ms | 8.3 µs   | ~46.5x               |
| fla-tma       | ~0.041 ms | 14.0 µs  | ~31x                 |

NVBench confirms the kernel is latency-bound on B200: <2% of 7.7 TB/s bandwidth utilized. See `findings/research.md` "NVBench on B200".

## Modal Deployment Notes
- The Modal image must install ALL Python packages that `kernel.py` imports at the top level
- `flash-linear-attention` (the `fla` package) is NOT a dependency of `flashinfer-bench` — must be explicitly added to the Modal image
- If a top-level import fails on Modal, the benchmark reports `COMPILE_ERROR` for every workload (the module can't even be loaded)
- The benchmark framework's `COMPILE_ERROR` status is opaque — it covers import errors, torch.compile failures, and any other pre-execution errors, with no error message surfaced in the output
- To debug Modal errors: check that all imports in `kernel.py` are available in the Modal image (`run_modal.py` `.pip_install(...)`)
- **CUDA kernels need nvcc**: `debian_slim` lacks the CUDA toolkit. Use `Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")` as the base image so `tvm_ffi.cpp.build()` can compile `.cu` files. The `-devel` suffix is required (runtime-only images lack nvcc and headers).

## Proton Intra-Kernel Profiling
- Triton 3.5.1 includes **Proton**, an intra-kernel profiler: `import triton.profiler as proton`
- DSL: `import triton.profiler.language as pl` — insert `pl.scope()` / `pl.enter_scope()` / `pl.exit_scope()` into `@triton.jit` kernels to annotate regions
- Gluon kernels (`@gluon.jit`) also support the same `pl.scope()` annotations
- Must call `pl.enable_semantic("triton")` before launching profiled Triton kernels
- Two profiling modes:
  - **Timeline trace**: `proton.start("name", data="trace", backend="instrumentation", mode=mode)` → outputs `.chrome_trace` file (view in Perfetto or `chrome://tracing`)
  - **Op measurement**: `proton.start("name", backend="instrumentation", mode=mode)` → outputs `.hatchet` file (view with `proton-viewer -m normalized_cycles`)
- Warp sampling: `proton.mode.Default(sampling_strategy="selective", sampling_options="0,2")` to profile only specific warps
- Example: `timeline/example_dsl.py` (vector add + Gluon matmul; matmul requires Hopper GPU)
- **Gluon** (`triton.experimental.gluon`) is also available in triton 3.5.1 — low-level Triton extension for TMA, warpgroup MMA, mbarrier, etc. (Hopper-only features)

### Profiling the GDN decode kernel
- Script: `scripts/profile_proton.py` — run via `make proton-fla`
- Output: `profiles/gdn_decode.chrome_trace` and `profiles/gdn_decode.hatchet`
- **torch.compile incompatibility**: torch.compile serializes Triton kernel source into a new scope, losing the `pl` import → `NameError`. Fix: `torch._dynamo.config.disable = True` for profiling.
- **Scope toggling**: `PROTON_PROFILE` env var read at call time in the wrapper, passed as `PROFILE: tl.constexpr` to the kernel. When `False`, Triton eliminates dead branches at AST level → no `pl` references → torch.compile works. When `True` (profiling), dynamo is disabled → `pl` resolves normally.
- Kernel scopes: `load_initial_state`, `load_qkv`, `state_update`, `store_output`, `store_final_state`

## NCU (NVIDIA Nsight Compute) Profiling
- Script: `scripts/profile_ncu.py` — run via `make ncu-fla` (requires sudo)
- Output: `profiles/ncu/gdn-decode-fla.ncu-rep` (binary, open in Nsight Compute GUI or `ncu --import`)
- Text export: `make ncu-export-fla` → `profiles/ncu-txt/gdn-decode-fla.txt`
- Uses `--set full` for all metrics (speed-of-light, memory, occupancy, warp stalls, scheduler, roofline)
- Key findings documented in `findings/research.md` under "NCU Detailed Metrics"

### Key profiling results (RTX 3090, fla-recurrent)
- **NCU kernel time**: 3.84 µs (benchmarks agree at ~4.3-5.1 µs after timing fix)
- **Bottleneck**: Latency-bound (long scoreboard stalls, 0.26 waves, 23.6% occupancy, 15.6% DRAM BW)
- **Proton scope breakdown**: `load_initial_state` ~40%, `load_qkv` ~25%, `state_update` ~15%, `store_final_state` ~15%
- Full NCU metrics and Proton analysis: `findings/research.md` under "NCU Detailed Metrics"

## Triton Autotuning
- **Avoid `@triton.autotune`** for this kernel; hardcode config at launch site (adds ~6 µs Python dispatch overhead)
- `@triton.autotune` is incompatible with `torch.compile` (compile bypasses autotuner entirely)
- BV tile size matters: BV=8/16 tied at ~4.25 µs, degrades to ~7.3 µs at BV=128 (see `findings/research.md`). Hardcoded: BV=8, num_warps=8, num_stages=2
- Detailed investigation: `findings/research.md` under "Triton Autotuning Investigation"

## Benchmark Timing Fix (torch.cuda.synchronize removal)
- Removed `torch.cuda.synchronize()` from flashinfer-bench's `do_bench()` hot loop to eliminate GPU idle bubble
- **Before**: ~51 µs (fla), ~43 µs (fi). **After**: ~4.3 µs (fla), ~4.7 µs (fi). Matches NCU within ~1 µs.
- Validated with NVBench (`cuda-bench`): ~5.1 µs (fla), ~5.4 µs (fi)
- NVBench solves the same problem with a "blocking kernel" ([talk](http://www.youtube.com/watch?v=CtrqBmYtSEk&t=838))
- Script: `scripts/bench_nvbench.py`, targets: `make nvbench-fla`, `make nvbench-fi`
- Full analysis (problem, fix, NVBench comparison): `findings/research.md` under "Benchmark Timing Fix"

## Misc Kernel Notes
- `tensor.set_()` can alias DPS output to input storage (zero-copy) since benchmark clones args each iter
- `torch.compile` cannot trace CuTe-DSL internals (`from_dlpack`, `cute.compile`, `cuda.CUstream`, TVM FFI)

## FlashInfer Baseline (fi-baseline algo)
- Uses `flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose` (CuTe-DSL kernel)
- K-last state layout `[B, HV, V, K]` f32 — matches competition exactly
- FI updates state in-place; use `new_state.set_()` to alias storage (zero-copy DPS)
- NCU kernel name regex: `kernel_cutlass_gdn_decode`
- pip package name: `flashinfer-python` (NOT `flashinfer`)
- Detailed analysis: `findings/fi-gdn-decode-kernel.md`

## CUDA Kernel (cuda-v1 algo)
- Hand-written CUDA C++ port of the FLA Triton kernel
- Grid: `(V/BV, B*HV)` = `(16, 8)` = 128 blocks, 1 warp (32 threads) per block
- Per-thread: KVEC=4 K-elements, `h[8][4]` state in registers (32 regs)
- State loads/stores: `float4` for coalesced 128-bit access
- Reductions: warp `__shfl_xor_sync` butterfly all-reduce (5 steps, no shared memory)
- TVM FFI integration: `TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda, ...)` in kernel.cu
- Local binding: `binding.py` uses `tvm_ffi.cpp.build()` + `tvm_ffi.load_module()` for bench/profile scripts
- NCU kernel name: `gdn_decode_kernel`

### TVM FFI Builder (language="cuda")
- The framework's TVMFFIBuilder compiles `.cu` files via `tvm_ffi.cpp.build()` (nvcc)
- Host function receives `tvm::ffi::TensorView` (zero-copy from torch via DLPack)
- Stream via `TVMFFIEnvGetStream(dev.device_type, dev.device_id)`
- Export: `TVM_FFI_DLL_EXPORT_TYPED_FUNC(symbol_name, cpp_function)`
- Entry point format: `kernel.cu::symbol_name`
- Supported arg types: `TensorView`, `int32_t`, `int64_t`, `float`, `double`, `bool`, `std::string`
- Headers: `<tvm/ffi/container/tensor.h>`, `<tvm/ffi/function.h>`, `<tvm/ffi/extra/c_env_api.h>`
- `tvm` package is NOT installed; use `tvm_ffi` (`from tvm_ffi import register_global_func`)
- `@register_global_func` as a decorator wraps the function as a TVM PackedFunc, which breaks `**kwargs` calls. Use non-decorator form instead: `register_global_func("name", func)`
- `pack_solution_from_files` rejects empty files (SourceFile content >= 1 char); `__init__.py` needs a comment

## Makefile — Primary Interface
The Makefile is the main way to run things. Always prefer `make` targets over raw commands, and improve them when adding new workflows. Pipe long outputs to log files.
```
make bench-fla              # local benchmark (fla-recurrent)
make bench-fi               # local benchmark (fi-baseline)
make bench-cuda             # local benchmark (cuda-v1)
make bench-pt               # local benchmark (pt-reference)
make modal-fla              # Modal B200 benchmark (fla-recurrent)
make modal-fi               # Modal B200 benchmark (fi-baseline)
make modal-pt               # Modal B200 benchmark (pt-reference)
make bench-fla-all          # local + modal, logs to logs/bench-fla-{local,modal}.log
make bench-tma-all          # local + modal, logs to logs/bench-tma-{local,modal}.log
make document-speedups      # parse bench logs → findings/speedups.csv
make modal-logs             # download Modal benchmark logs to logs/fib-bench-modal/
make proton-fla             # profile kernel with Proton
make ncu-fla                # NCU full profile → profiles/ncu/gdn-decode-fla.ncu-rep
make ncu-fi                 # NCU full profile → profiles/ncu/gdn-decode-fi.ncu-rep
make ncu-cuda               # NCU full profile → profiles/ncu/gdn-decode-cuda.ncu-rep
make ncu-export-fla         # NCU text export → profiles/ncu-txt/gdn-decode-fla.txt
make ncu-export-fi          # NCU text export → profiles/ncu-txt/gdn-decode-fi.txt
make nvbench-fla            # NVBench benchmark (fla-recurrent)
make nvbench-fi             # NVBench benchmark (fi-baseline)
make nvbench-cuda           # NVBench benchmark (cuda-v1)
make clean-triton-cache     # clear ~/.triton/cache
```
- Env var overrides: `NUM_WORKLOADS=3 make modal-fla` (limit workloads), `ALGO=... make modal-fla`
- `-n 3` flag on local scripts: `python -m scripts.run_local --algo=fla-recurrent -n 3`
- Local GPU: RTX 3090 (Ampere SM86) — cannot run Hopper-only features (TMA, warpgroup MMA, Gluon matmul examples)

### Log file naming convention
- `bench-*-all` targets write logs named `logs/bench-{short}-{local,modal}.log`
- Algo → log prefix mapping: `fla-recurrent` → `bench-fla`, `fla-tma` → `bench-tma`, `pt-reference` → `bench-pt`
- `document-speedups` reads `ALGO` env var (default `fla-recurrent`) to find the right log files
- Usage: `ALGO=fla-tma make document-speedups COMMENT="tma v1"`

## flashinfer-bench Internals
- Source: `.venv/lib/python3.11/site-packages/flashinfer_bench/` (or find via `.venv/bin/python -c "import flashinfer_bench; print(flashinfer_bench.__path__)"`)
- Builder loads solution, imports as Python module, gets entry_point via `getattr()`
- `Runnable` wraps the callable, handles DPS vs value-returning dispatch
- `PersistentRunner` spawns subprocess workers per GPU device
- Evaluator: `DefaultEvaluator` — checks correctness then profiles performance
- Timing: `do_bench()` in `bench/timing.py` (derived from Triton's benchmark utility)
