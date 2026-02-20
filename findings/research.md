# Track C: Gated Delta Net -- Research Notes

## Competition Overview

- **Competition**: [FlashInfer AI Kernel Generation Contest @ MLSys 2026](https://mlsys26.flashinfer.ai/)
- **Track C**: Gated Delta Net (GDN), used in Qwen3-Next-80B-A3B-Instruct
- **Target hardware**: NVIDIA Blackwell B200 GPUs
- **Deadline**: April 24, 2026 (kernel), May 1 (writeup)
- **Evaluation**: correctness, speed, win rate against FlashInfer baselines
- **Dataset**: [flashinfer-ai/mlsys26-contest](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)
- **Starter kit**: [flashinfer-bench-starter-kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)

## What is Gated Delta Net?

GDN is a **linear-time recurrent sequence model** that replaces softmax attention. Instead of a growing KV cache, it maintains a **fixed-size memory matrix S** (d_k x d_v per head) that gets updated at each token.

**Key properties vs softmax attention:**

- O(n) time complexity (vs O(n^2))
- Constant memory (fixed-size state vs linearly growing KV cache)
- Enables 256K+ context with bounded memory

**Deployed in:**

- [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (Alibaba) -- 75% GDN layers, 25% softmax attention, hybrid 3:1 ratio
- Kimi Linear-48B-A3B (Moonshot AI) -- similar hybrid approach

## The GDN Recurrence (Decode Step)

```text
S_t = α_t * S_{t-1} * (I - β_t * k_t * k_tᵀ) + β_t * v_t * k_tᵀ
o_t = S_t * q_t
```

Three components:

1. **`α_t * S_{t-1}`** -- global gating/decay (α ∈ (0,1)). "Forget everything a little bit."
2. **`- β_t * (S_{t-1} * k_t) * k_tᵀ`** -- delta rule erasure. Retrieves old value for key k_t, subtracts it. "Surgical forgetting."
3. **`+ β_t * v_t * k_tᵀ`** -- write new value v_t for key k_t.

The delta rule (steps 2+3) is the classical Widrow-Hoff learning rule: `S += β * (v_target - v_predicted) * kᵀ`.

### Comparison with simpler linear attention variants

| Model                        | State update                  | Limitation                                        |
| ---------------------------- | ----------------------------- | ------------------------------------------------- |
| Linear Attention             | S += v * kᵀ                   | Only accumulates, never forgets. State pollution. |
| GLA (Gated Linear Attention) | S = α * S + v * kᵀ            | Global decay only, can't selectively overwrite.   |
| DeltaNet                     | S = S*(I - β*k*kᵀ) + β*v*kᵀ   | Selective erase+write, but no global decay.       |
| **Gated DeltaNet**           | S = α*S*(I - β*k*kᵀ) + β*v*kᵀ | Both global decay AND selective overwrite.        |

## The Householder Connection

The transition matrix `(I - β * k * kᵀ)` is **Householder-like**:

- A Householder matrix is `H = I - 2 * u * uᵀ` (with u unit vector) -- a reflection across the hyperplane perpendicular to u
- When β=2: exact Householder reflection
- When β=1: projection (fully erases component along k)
- When β ∈ (0,1): partial erasure (shrinks component along k by factor 1-β)

**Why this matters for parallelization:** The product of N Householder-like matrices:

```text
T_1 * T_2 * ... * T_N = (I - β₁k₁k₁ᵀ)(I - β₂k₂k₂ᵀ)...(I - βₙkₙkₙᵀ)
```

can be written in **WY compact form**: `I - W * Yᵀ` where W, Y are d×N matrices. This converts sequential matrix products into **GEMMs** (tensor-core friendly).

## Two Kernels Required

### Decode Kernel

- **What:** Single recurrent step per new token
- **Regime:** Memory-bound (small compute, large state read/write)
- **State S:** d_k × d_v matrix per head (e.g., 128×128 = 16K elements)
- **Complexity:** Moderate -- essentially a matrix-vector operation with rank-1 update
- **Key challenge:** Efficient memory access for the state matrix

### Prefill Kernel

- **What:** Process entire prompt at once
- **Algorithm:** Chunkwise parallelism (chunks of ~64-128 tokens)
  - **Within chunk:** WY representation converts recurrence into GEMMs (parallel, tensor-core friendly)
  - **Across chunks:** State S propagated recurrently
- **Regime:** Compute-bound (large matrix multiplies)
- **Complexity:** High -- WY factorization, tiling, shared memory management
- **Key challenges:** Correct WY factorization with gating, numerical stability (L2-normalized keys), chunk boundary correctness

## Existing Implementations

### Fast / Optimized

- **fla-org/flash-linear-attention** ([link](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule))
  - Primary Triton kernel library for linear attention
  - `fla/ops/common/chunk_delta_h.py` contains chunk kernel
  - Known bug on Blackwell backward pass ([issue #607](https://github.com/fla-org/flash-linear-attention/issues/607))
- **NVlabs/GatedDeltaNet** ([link](https://github.com/NVlabs/GatedDeltaNet/tree/main/lit_gpt/gated_delta_rule_ops/fla_version))
  - Official ICLR 2025 implementation (wraps FLA kernels)

### Educational / Reference

- [Sebastian Raschka walkthrough](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/)
- [DeltaNet Explained Part I](https://sustcsonglin.github.io/blog/2024/deltanet-1/) (Songlin Yang, paper author)
- [DeltaNet Explained Part II](https://sustcsonglin.github.io/blog/2024/deltanet-2/) (Songlin Yang, paper author)

### Papers

- **Gated Delta Networks** (ICLR 2025): [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- **DeltaNet** (NeurIPS 2024): [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
- **Kimi Linear** (extends GDN): [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

## Decode Kernel Optimizations Applied

### 1. Fused gate computation into Triton kernel

Moved `g = exp(-exp(A_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)` from the Python wrapper into the `@triton.jit` kernel. These are scalar ops per (batch, head) — trivial compute but each was a separate CUDA kernel launch when done in PyTorch. Fusing eliminated 4 kernel launches on tiny [B=1, 1, 8] tensors.

### 2. Eliminated state transposes (k-last layout)

The FLA kernel expected k-first `[K, V]` state layout, but the competition format stores state as k-last `[V, K]`. The wrapper was doing:
- Input: `state.float().transpose(-1, -2).contiguous()` — 512 KB allocation + copy
- Output: `new_state.copy_(ht_kfirst.transpose(-1, -2))` — 512 KB strided copy

Fix: changed the kernel's pointer math from `k * V + v` to `v * K + k`. The kernel now reads/writes k-last directly. No temp buffers, no transposes. This was the single biggest win (~2x speedup).

### 3. Removed torch.compile wrapper

With gates fused and transposes eliminated, the wrapper only does shape extraction and the kernel launch — nothing for torch.compile to optimize. Removing it avoids compilation overhead.

## Triton Autotuning Investigation

### Observed slowdown with `@triton.autotune` (RTX 3090)

| Setup                                    | Latency   | Delta     |
| ---------------------------------------- | --------- | --------- |
| Hardcoded config (no decorator)          | ~0.050 ms | —         |
| `@triton.autotune` (cached, same config) | ~0.066 ms | +0.016 ms |

The cause of the ~16 µs difference is not conclusively identified. Possible explanations:
- Python-level dispatch overhead (cache key hashing, dict lookup, kwarg merging) on every call
- Interaction with the benchmark harness subprocess model
- Something else — needs further investigation

### The config choice itself doesn't matter (for this kernel)

Tested BV={8, 32} and num_warps={1, 2} without the decorator — all perform identically at ~0.050 ms. The kernel is so small (B=1, 8 heads, no loops) that tile size and warp count don't affect performance on RTX 3090. May differ on B200.

### `@triton.autotune` is incompatible with `torch.compile`

When the Triton kernel launch is inside a `@torch.compile(fullgraph=True)` function, torch.compile captures the kernel call during tracing and handles it internally. The autotuner's `cache` dict stays empty — it never runs. torch.compile just uses whatever config it encounters.

## Optimization Opportunities / Headroom

1. **Blackwell B200 is new.** FLA kernels are research-grade Triton, not tuned for B200's tensor cores, TMA, or memory hierarchy. Known Blackwell bugs exist.
2. **Triton has a ceiling.** Hand-tuned CUDA C++ (especially with TMA, warp specialization) should have headroom over Triton for both kernels.
3. **Prefill kernel:** Not yet started. Chunk size tuning, shared memory management, pipeline stages for B200.

## Open Questions

- How much headroom exists on B200 vs current Triton decode kernel?
- Can we batch across heads to improve occupancy for the decode kernel?
- What is the optimal chunk size for prefill on B200?


# Decode Memory Traffic Analysis (B=1)

All 20 decode workloads use batch_size=1.

## Tensor sizes

| Tensor         | Shape            | Dtype | Size      |
| -------------- | ---------------- | ----- | --------- |
| state (in)     | [1, 8, 128, 128] | f32   | 512 KB    |
| state (out)    | [1, 8, 128, 128] | f32   | 512 KB    |
| q              | [1, 1, 4, 128]   | bf16  | 1 KB      |
| k              | [1, 1, 4, 128]   | bf16  | 1 KB      |
| v              | [1, 1, 8, 128]   | bf16  | 2 KB      |
| output         | [1, 1, 8, 128]   | bf16  | 2 KB      |
| a, b           | [1, 1, 8]        | bf16  | 16 B each |
| A_log, dt_bias | [8]              | f32   | 32 B each |
| **Total**      |                  |       | **~1 MB** |

State is ~99.2% of total memory traffic. The kernel is essentially a **state memcpy with math sprinkled in**.

## Proton profiling confirms this (RTX 3090, BV=8)

- `load_initial_state` ~40% of runtime
- `store_final_state` ~15% of runtime
- `load_qkv` ~25% (small tensors, but many tiles)
- `state_update` ~15% (the actual compute)
- `store_output` negligible
