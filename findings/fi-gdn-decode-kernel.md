# FlashInfer GDN Decode Kernel — PR #2498

**PR**: https://github.com/flashinfer-ai/flashinfer/pull/2498
**Issue**: https://github.com/flashinfer-ai/flashinfer/issues/2493
**Status**: Merged on 2026-02-17
**Author**: @ameynaik-hub (with @kahyunnam)
**Reviewers**: @yzh119 (maintainer, approved), @kahyunnam (approved), @guangyunh-nv (NVIDIA, GDN prefill author)

## What It Is

A high-performance CuTe-DSL (NVIDIA CUTLASS Python DSL) kernel for **GDN decode** with:
- **K-last state layout** `[B, HV, V, K]` — K is the contiguous/fastest dimension
- **BF16 state** (not FP32) — halves state memory bandwidth
- **Fixed sequence lengths** T=1, 2, 3, 4 only (separate optimized codepaths)
- **BF16 tensors with FP32 compute** for numerical stability
- L2-normalized Q/K with configurable scale
- Async H-state memory loading with aggressive pipelining
- GVA (Grouped Value Attention): 4 q/k heads, 8 v heads (Qwen3-Next config)

## Files Changed

| File | Description |
|------|-------------|
| `flashinfer/gdn_kernels/gdn_decode_bf16_state.py` | **Main kernel** — 2062 lines of CuTe-DSL code |
| `flashinfer/gdn_kernels/__init__.py` | New module init |
| `flashinfer/gdn_decode.py` | Integration: dispatches to new kernel when K-last layout + T<=4 |
| `flashinfer/cute_dsl/__init__.py` | Exports `gated_delta_rule`, `GatedDeltaRuleKernel` |
| `benchmarks/bench_gdn_decode.py` | Benchmark integration (+354 lines) |
| `tests/gdn/test_decode_delta_rule.py` | Tests for T=1..4 with BF16 state |
| `tests/gdn/reference_delta_rule.py` | Added `state_dtype` parameter to reference impl |

## Benchmark Results (B200, BF16 state)

| Batch | T=1 (us) | T=2 (us) | T=3 (us) | T=4 (us) |
|-------|----------|----------|----------|----------|
| 1     | 3.62     | 5.86     | 6.94     | 7.79     |
| 4     | 4.90     | 6.62     | 7.65     | 8.64     |
| 8     | 7.04     | 8.67     | 9.95     | 11.39    |
| 16    | 9.74     | 12.90    | 15.15    | 17.54    |
| 32    | 16.06    | 21.57    | 25.79    | 28.61    |
| 64    | 27.01    | 34.88    | 40.67    | 49.76    |
| 128   | 48.90    | 60.70    | 72.26    | 89.04    |
| 256   | 91.66    | 112.54   | 134.88   | 166.32   |
| 512   | 177.06   | 214.53   | 258.75   | 320.80   |

## Speedup vs Previous FlashInfer Pretranspose Kernel (H100)

From issue #2493 benchmarks (times in μs):

| Batch | T=1 spd | T=2 spd | T=3 spd | T=4 spd |
|-------|---------|---------|---------|---------|
| 1     | 0.90x   | 1.30x   | 1.25x   | 1.25x   |
| 8     | 1.30x   | 1.58x   | 1.64x   | 1.77x   |
| 64    | 1.75x   | 1.97x   | 2.05x   | 2.22x   |
| 256   | 1.86x   | 1.99x   | 2.23x   | 2.36x   |
| 512   | 1.88x   | 1.99x   | 2.28x   | 2.38x   |

At B=1/T=1 the new kernel is slightly slower (0.9x), but at larger batch sizes it's **up to 2.4x faster**.

## FP32 State Follow-up

Author also benchmarked an FP32 state variant (not in this PR, separate follow-up):

| BS  | FI-PreTr (us) | New (us) | Speedup |
|-----|---------------|----------|---------|
| 1   | 3.74          | 3.52     | 1.06x   |
| 64  | 54.85         | 42.38    | 1.29x   |
| 128 | 92.21         | 81.39    | 1.13x   |
| 512 | 336.56        | 313.69   | 1.07x   |

## Kernel Architecture Details

### Two codepaths
1. **T=1 kernel** (`gdn_decode_kernel_seqlen1`): Specialized single-token kernel with 4 V-chunks unrolled inline, async pipelining of H-state tiles
2. **T=2,3,4 kernel** (`process_vchunk_unified_234`): Unified multi-token kernel with helper functions, processes tokens sequentially within each V-chunk

### Key optimizations
- **Async H-state pipelining**: State tiles loaded from GMEM asynchronously while compute proceeds on previous tiles
- **4 CTAs per head for low batch**: When B<=4, launches 4 CTAs per batch×head to increase parallelism
- **Cross-warp reduction**: Uses shared memory for warp-level reductions of partial dot products
- **Vectorized K-dim reads**: K-last layout enables coalesced 128-element reads along K dimension
- **Gate computation fused**: `g = exp(-exp(A_log) * softplus(a + dt_bias))` computed in-kernel

### Compile options
- Uses `--enable-tvm-ffi` to reduce host-side DLPack conversion overhead (reviewer suggestion)
- `--generate-line-info` included (reviewer noted this should be conditional for production)
- Kernel caching by `(T, B)` — reviewer noted should include `(H, HV, K, V)` too

## Key Discussion Points

### BF16 vs FP32 State
- @guangyunh-nv (NVIDIA, GDN prefill author): "FP16 state might not be a good choice due to dynamic range. BF16 maybe safer, but accuracy could be a problem. Should tolerate all state dtypes and learn from production."
- @vadiklyutiy: "In vLLM, `h.dtype=bf16`. So no need to worry."
- @xutizhou: Asked about end-to-end accuracy benchmarks (unanswered)
- **Conclusion**: BF16 state is used in production (vLLM, sglang), kernel supports it

### Why K-Last Layout?
- @guangyunh-nv: "In prefill, you need to repeatedly update State in the kernel mainloop, so put state in registers (Hopper A operand). This avoids repeated transposition. Critical for performance. Blackwell has the same constraint."
- @ameynaik-hub: "K-last allows vectorized reads along K-dim, inner products with k and q can be parallelized across rows more easily."
- @vadiklyutiy: "vLLM already moved to K-last layout, gives perf improvement by itself."

### GVA Naming
- @guangyunh-nv confirmed with GDN paper author: the pattern where `HV > HQ` is called **GVA** (Grouped Value Attention), not GQA (which is the inverse: more Q than KV heads)

### Code Organization
- @yzh119 (maintainer) requested moving code out of `flashinfer/cute_dsl/` into `flashinfer/gdn_kernels/` — modules should be organized by functionality, not source
- Final location: `flashinfer/gdn_kernels/gdn_decode_bf16_state.py`
- Integrated into existing `flashinfer/gdn_decode.py` dispatch: new kernel is default for K-last layout with T<=4

### Review Issues Flagged
- **Code duplication**: T=1 kernel has 4 nearly-identical V-chunk blocks (~300 lines repeated). Reviewers suggested extracting a helper
- **Incomplete `GatedDeltaRuleKernel` class**: Exported but never used, `_compiled_kernel` never assigned
- **Cache key too narrow**: `(T, B)` doesn't include `H, HV, K, V` — could cause incorrect kernel reuse
- **`use_qk_l2norm_in_kernel` param ignored**: Kernel always L2-normalizes Q/K, param is dead
- **Shared memory bank conflicts**: `reduce_sh` layout has 4-way bank conflicts despite "bank-conflict-free" comment
- **`--generate-line-info` in production**: Increases code size, should be conditional

### CI Status
- First CI run: 11/20 passed (some test failures during development)
- Merged after fixes with maintainer approval

## Relevance to Our Competition

This kernel targets the **exact same problem** as our competition entry (`gdn_decode_qk4_v8_d128_k_last`):
- Same K-last state layout `[B, HV, V, K]`
- Same head config: 4 qk heads, 8 v heads, d=128
- Same decode (T=1) use case
- BF16 inputs with FP32 compute

**Key differences from our setup**:
- Their kernel uses **CuTe-DSL** (NVIDIA's CUTLASS Python DSL), not Triton
- Supports T=1..4, our competition only needs T=1
- Their state is **BF16**, our competition's state is **FP32** `[B, 8, 128, 128]`
- They report **3.62 μs at B=1/T=1 on B200** (BF16 state), **3.52 μs** (FP32 state follow-up)

**Performance comparison** (B=1, T=1):
| Kernel | GPU | Pure kernel time | Measurement |
|--------|-----|-----------------|-------------|
| Our fla-recurrent | RTX 3090 | **~3.84 μs** | NCU `Duration` metric |
| Their CuTe-DSL (BF16 state) | B200 | **3.62 μs** | bench_gpu_time |
| Their CuTe-DSL (FP32 state) | B200 | **3.52 μs** | bench_gpu_time |

Our kernel is in the same ballpark (~3.84 μs on RTX 3090 vs ~3.5 μs on B200). The ~37 μs number from `run_local.py` includes Python dispatch overhead, L2 cache flushing, and tensor cloning — it is **not** the pure kernel time.

Note: the benchmark framework (`run_local.py`) measures end-to-end latency including host overhead, which is what the competition scores on. The CuTe-DSL kernel uses `--enable-tvm-ffi` to minimize DLPack conversion overhead, which could give it an edge in end-to-end benchmarks despite similar raw kernel performance.
