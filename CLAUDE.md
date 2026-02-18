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
- Conda env: `fi-bench` (Python 3.12)
- Packages: `flashinfer-bench`, `modal`, `torch 2.9.1`, `triton 3.5.1`
- Dataset: `~/code/mlsys26-contest` (env var `FIB_DATASET_PATH`, set in `~/.bashrc`)

## Config Notes
- `entry_point` format must include the `.py` extension: `kernel.py::kernel` (not `kernel::kernel` or just `kernel`)
- `definition` must be the exact definition name from the dataset (e.g. `gdn_decode_qk4_v8_d128_k_last`), not the track name (`gated_delta_net`)
- DPS (Destination Passing Style) is default: kernel receives pre-allocated output tensors as extra args

## GDN Track: Two Kernels
Each kernel is a separate definition, needs a separate `config.toml` definition entry:

### 1. Decode: `gdn_decode_qk4_v8_d128_k_last`
- Single-token generation (seq_len=1)
- Shapes: `q/k: [B, 1, 4, 128]`, `v: [B, 1, 8, 128]`, `state: [B, 8, 128, 128]` (f32)
- GVA config: 4 q/k heads, 8 v heads (v_heads = 2x q_heads, heads repeat-interleaved)
- Inputs: q, k, v, state, A_log, a, dt_bias, b, scale
- Outputs (DPS): output `[B, 1, 8, 128]` (bf16), new_state `[B, 8, 128, 128]` (f32)
- All 20 workloads use batch_size=1
- Memory-bound regime

### 2. Prefill: `gdn_prefill_qk4_v8_d128_k_last`
- Variable-length batched sequences (uses `cu_seqlens`)
- Shapes: `q/k: [total_seq_len, 4, 128]`, `v: [total_seq_len, 8, 128]`, `state: [num_seqs, 8, 128, 128]`
- Inputs: q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale
- Outputs (DPS): output `[total_seq_len, 8, 128]` (bf16), new_state `[num_seqs, 8, 128, 128]` (f32)
- Compute-bound regime (chunkwise parallelism, WY factorization)

### Core Recurrence (both kernels)
```
g = exp(-exp(A_log) * softplus(a + dt_bias))    # global decay gate
beta = sigmoid(b)                                 # update gate
state = g * state + k^T @ (beta * v + (1-beta) * k @ state) - k^T @ (k @ state)
output = scale * q @ state
```
State layout is "k-last": `[..., V, K]` in storage, transposed to `[K, V]` for computation.

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

## flashinfer-bench Internals
- Source: `/home/tomasruiz/miniforge3/envs/fi-bench/lib/python3.12/site-packages/flashinfer_bench/`
- Builder loads solution, imports as Python module, gets entry_point via `getattr()`
- `Runnable` wraps the callable, handles DPS vs value-returning dispatch
- `PersistentRunner` spawns subprocess workers per GPU device
- Evaluator: `DefaultEvaluator` — checks correctness then profiles performance
- Timing: `do_bench()` in `bench/timing.py` (derived from Triton's benchmark utility)
