# FlashInfer AI Kernel Generation Contest - GDN Track

## Team
- Team name: **lmu-css**
- Track: **gated_delta_net** (Gated Delta Net)

## Project Structure
- `config.toml` — Solution metadata and build config. `definition` must match the exact definition name (e.g. `gdn_decode_qk4_v8_d128_k_last`), not the track name.
- `solution/triton/kernel.py` — Triton/Python kernel implementation. Entry point is a regular Python function (not necessarily `@triton.jit`).
- `solution/cuda/kernel.cu` + `binding.py` — CUDA alternative.
- `scripts/run_local.py` — Local benchmark runner.
- `scripts/run_modal.py` — Cloud benchmark (B200 GPUs via Modal).
- `scripts/pack_solution.py` — Packs solution into `solution.json`.

## Environment
- Venv: `.venv/` in project root
- Packages: `flashinfer-bench`, `modal`, `torch`, `triton`
- Dataset: `~/code/mlsys26-contest` (env var `FIB_DATASET_PATH`, set in `~/.bashrc`)

## Config Notes
- `entry_point` format must include the `.py` extension: `kernel.py::kernel` (not `kernel::kernel` or just `kernel`)
- `definition` must be the exact definition name from the dataset (e.g. `gdn_decode_qk4_v8_d128_k_last`), not the track name (`gated_delta_net`)
- DPS (Destination Passing Style) is default: kernel receives pre-allocated output tensors as extra args

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

## Multi-Algo Benchmarking

### Entry point dispatch via `--algo` flag
```bash
python scripts/run_local.py --algo=fla-recurrent    # default
python scripts/run_local.py --algo=pt-reference      # compiled PyTorch reference
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

### Performance (RTX 3090)
| Algo                     | Latency   | Speedup vs reference |
| ------------------------ | --------- | -------------------- |
| pt-reference (eager)     | ~1.4 ms   | ~1.0x                |
| pt-reference (compiled)  | ~0.73 ms  | ~1.8x                |
| fla-recurrent            | ~0.05 ms  | ~29x                 |

### Performance (B200 via Modal)
| Algo                     | Latency   | Speedup vs reference |
| ------------------------ | --------- | -------------------- |
| fla-recurrent            | ~0.037 ms | ~32x                 |

## Modal Deployment Notes
- The Modal image must install ALL Python packages that `kernel.py` imports at the top level
- `flash-linear-attention` (the `fla` package) is NOT a dependency of `flashinfer-bench` — must be explicitly added to the Modal image
- If a top-level import fails on Modal, the benchmark reports `COMPILE_ERROR` for every workload (the module can't even be loaded)
- The benchmark framework's `COMPILE_ERROR` status is opaque — it covers import errors, torch.compile failures, and any other pre-execution errors, with no error message surfaced in the output
- To debug Modal errors: check that all imports in `kernel.py` are available in the Modal image (`run_modal.py` `.pip_install(...)`)

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

### Decode kernel profiling results (RTX 3090, BV=8)
- **Memory-bound**: loads/stores dominate, compute is minor
- `load_initial_state` ~40% — loading [BK=128, BV=8] f32 state tile from GMEM (16 tiles per head)
- `load_qkv` ~25% — loading q [BK], k [BK], v [BV] vectors
- `state_update` ~15% — delta rule matvecs + outer product (the actual compute)
- `store_final_state` ~15% — writing updated state tile back to GMEM
- `store_output` negligible — just [BV=8] values per tile

## Triton Autotuning

### Observed slowdown with `@triton.autotune` on RTX 3090
- Without decorator (hardcoded config): ~0.050 ms
- With decorator (cached, same config picked): ~0.066 ms
- Cause not conclusively identified — could be Python dispatch overhead, could be interaction with the benchmark harness subprocess model
- Need more investigation before drawing conclusions

### `@triton.autotune` + `@torch.compile` incompatibility
- `torch.compile(fullgraph=True)` wrapping the outer function bypasses Triton's autotune entirely
- The autotuner cache stays empty — torch.compile traces the kernel launch and handles it internally
- To use autotune, the kernel launch must NOT be inside a torch.compiled function

### Config sweep results (RTX 3090, no decorator, all equivalent ~0.050 ms)
- BV=8 num_warps=1, BV=32 num_warps=1, BV=32 num_warps=2 all perform identically
- `num_stages` is irrelevant (no loop to pipeline)

## Running Scripts Locally
- Use the project venv to run scripts:
  ```bash
  .venv/bin/python script.py
  ```
- Local GPU: RTX 3090 (Ampere SM86) — cannot run Hopper-only features (TMA, warpgroup MMA, Gluon matmul examples)

## flashinfer-bench Internals
- Source: `.venv/lib/python3.11/site-packages/flashinfer_bench/` (or find via `.venv/bin/python -c "import flashinfer_bench; print(flashinfer_bench.__path__)"`)
- Builder loads solution, imports as Python module, gets entry_point via `getattr()`
- `Runnable` wraps the callable, handles DPS vs value-returning dispatch
- `PersistentRunner` spawns subprocess workers per GPU device
- Evaluator: `DefaultEvaluator` — checks correctness then profiles performance
- Timing: `do_bench()` in `bench/timing.py` (derived from Triton's benchmark utility)
