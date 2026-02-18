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

## Optimization Opportunities / Headroom

1. **Blackwell B200 is new.** FLA kernels are research-grade Triton, not tuned for B200's tensor cores, TMA, or memory hierarchy. Known Blackwell bugs exist.
2. **Triton has a ceiling.** Hand-tuned CUDA C++ (especially with TMA, warp specialization) should have headroom over Triton for both kernels.
3. **Competition baseline is unknown.** FlashInfer baseline may be a naive PyTorch implementation (easy to beat) or something more optimized. Need to download dataset and measure.
4. **Decode kernel:** Memory access pattern optimization (state read/write coalescing, batching across heads).
5. **Prefill kernel:** Chunk size tuning, shared memory management, pipeline stages for B200.

## Open Questions

- What exactly is the FlashInfer baseline? (Need to download dataset and run)
- What are the exact kernel signatures / tensor shapes for the competition? (In the dataset definitions)
- What kernel names does the competition use? (e.g., `gdn_decode_qk4_v8_d128_k_last`)
- How much headroom exists on B200 vs current FLA Triton kernels?
