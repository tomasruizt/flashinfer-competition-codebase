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

See `CLAUDE.md` "GDN Track: Two Kernels" for exact tensor shapes, dtypes, and definition names.

### Decode Kernel

- **What:** Single recurrent step per new token
- **Regime:** Memory-bound (small compute, large state read/write)
- **Key challenge:** Efficient memory access for the state matrix

### Prefill Kernel

- **What:** Process entire prompt at once
- **Algorithm:** Chunkwise parallelism (chunks of ~64-128 tokens)
  - **Within chunk:** WY representation converts recurrence into GEMMs (parallel, tensor-core friendly)
  - **Across chunks:** State S propagated recurrently
- **Regime:** Compute-bound (large matrix multiplies)
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

### BV tile size matters (with accurate timing)

Previous testing at ~50 µs (inflated by `torch.cuda.synchronize`) showed no difference between configs. With accurate timing (~4 µs), BV has a clear effect. Sweep on RTX 3090, N=3 workloads, num_warps=8, num_stages=2:

| BV  | Avg Latency | Speedup | vs Best |
|-----|-------------|---------|---------|
| 4   | 4.97 µs     | ~240x   | +17%    |
| **8**   | **4.26 µs** | **~275x** | **baseline** |
| 16  | 4.25 µs     | ~283x   | tied    |
| 32  | 4.82 µs     | ~260x   | +13%    |
| 64  | 5.79 µs     | ~215x   | +36%    |
| 128 | 7.27 µs     | ~163x   | +71%    |

Sweet spot is BV=8 or BV=16. BV=4 launches too many threadblocks (not enough work per block). BV>=32 degrades monotonically, likely due to register pressure and reduced occupancy from larger tiles.

Hardcoded to BV=8 (B200 winner from earlier autotuning).

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

## Arithmetic intensity: 0.87 FLOPs/byte

Per head (B=1, K=128, V=128):

| Operation                     | FLOPs          |
| ----------------------------- | -------------- |
| Decay (`g * S`)               | 16,384         |
| k@S matvec                    | 32,768         |
| Blend (`β*(v - k@S)`)         | 256            |
| Outer product + state update  | 32,768         |
| q@S matvec                    | 32,768         |
| **Per head**                  | **~115K**      |
| **8 heads total**             | **~920K**      |

Bytes: ~1,028 KB (dominated by 2 × 512 KB state load/store). AI = 920K / 1,028 KB ≈ **0.87 FLOPs/byte**.

Roofline balance points for comparison:

| GPU      | Peak F32   | Bandwidth | Balance point   | Ratio vs kernel |
| -------- | ---------- | --------- | --------------- | --------------- |
| RTX 3090 | 35.6 TF    | 936 GB/s  | 38 FLOPs/byte   | 43× above       |
| B200     | ~180 TF    | 8 TB/s    | ~22 FLOPs/byte  | 25× above       |

The kernel is deep in the memory-bound regime. But the data volume is so small (1 MB) that it can't even saturate bandwidth — making it latency-bound in practice (see below).

## Why the kernel is latency-bound, not bandwidth-bound

Total GMEM traffic is ~1 MB (512 KB state read + 512 KB state write + negligible q/k/v/scalars). At RTX 3090's 936 GB/s, this should take ~1 µs. NCU measures the kernel at ~3.8 µs, and corrected benchmarks agree at ~4.3-5.1 µs. The kernel achieves ~15% of peak bandwidth (latency-bound, not bandwidth-bound; see NCU metrics below).

This is a single decode step; one read and one write of the state is irreducible. There are no redundant GMEM round-trips to eliminate.

## Benchmark Timing Fix: Removing torch.cuda.synchronize() from the Hot Loop

### The problem

flashinfer-bench's `do_bench()` (in `timing.py:187-206`) had a `torch.cuda.synchronize()` between the setup phase and the start CUDA event on every iteration:

```python
for i in range(rep):
    _clear_cache(cache)                  # async: zeros 256 MB L2-flush buffer
    setup_result = setup()               # async: clones tensor args
    torch.cuda.synchronize(device)       # BUG: forces GPU idle before start event
    start_events[i].record()
    fn(setup_result)
    end_events[i].record()
```

This sync drained the CUDA stream, so the GPU was idle when `start_events[i].record()` fired. The start timestamp was recorded immediately (idle GPU), then the CPU had to dispatch the kernel through Python (lambda, Runnable, Triton runtime, CUDA driver). The GPU sat idle during that entire CPU dispatch, and only started executing once the kernel arrived. The CUDA event elapsed time therefore measured: **CPU dispatch latency + kernel execution**.

For a ~3.8 µs kernel, this inflated the reported time to ~51 µs (13x).

### The fix

Remove `torch.cuda.synchronize()`. Without it, the CPU enqueues everything onto the stream:

```python
for i in range(rep):
    _clear_cache(cache)                  # enqueued on stream
    setup_result = setup()               # enqueued on stream (clone ops)
    start_events[i].record()             # enqueued on stream
    fn(setup_result)                     # enqueued on stream
    end_events[i].record()              # enqueued on stream
```

CUDA stream ordering guarantees that the start event records its timestamp only after the cache flush and clones complete on the GPU. The elapsed time between start and end events now measures only the kernel execution, with no GPU idle bubble.

### Why this is correct

The sync was unnecessary because CUDA streams are ordered. Operations enqueued on the same stream execute in FIFO order. The start event is enqueued after `_clear_cache()` and `setup()`, so its timestamp is recorded after those operations finish on the GPU. No explicit sync needed.

### Validation: three independent measurements agree

| Kernel | NCU (pure GPU) | NVBench (`cuda-bench`) | flashinfer-bench (fixed) |
|--------|---------------|----------------------|--------------------------|
| fla-recurrent | 3.84 µs | 5.14 µs | 4.2-4.6 µs |
| fi-baseline | 4.50 µs | 5.43 µs | 4.7-5.0 µs |

All three agree within ~1 µs. The old flashinfer-bench reported ~51 µs (fla) and ~43 µs (fi).

NVBench (`cuda-bench`, NVIDIA's official kernel benchmarking tool) separately reports **CPU Time** (~54-76 µs) and **GPU Time** (~5 µs), confirming the ~50 µs gap was entirely CPU dispatch overhead. NVBench ran ~97K samples with statistical convergence, L2 cache flushing (`batched=False`), and GPU throttle detection.

Script: `scripts/bench_nvbench.py`, targets: `make nvbench-fla`, `make nvbench-fi`.

### NVBench: B200 vs RTX 3090

| Kernel | RTX 3090 | B200 | B200/3090 ratio |
|--------|----------|------|-----------------|
| fla-recurrent | 5.3 µs | 7.1 µs | 1.34x slower |
| fi-baseline | 5.5 µs | 8.2 µs | 1.49x slower |
| fla-tma | 9.0 µs | 13.6 µs | 1.50x slower |

B200 specs: 148 SMs, 7672 GB/s peak bandwidth, 182 GB HBM3e.
RTX 3090 specs: 82 SMs, 936 GB/s peak bandwidth, 24 GB GDDR6X.

**B200 is 1.3-1.5x slower than RTX 3090** despite 8x the bandwidth and 1.8x the SMs. The kernel is latency-bound (~1 MB total data), so bandwidth is irrelevant.

Possible explanations (unverified, no B200 NCU profile yet):
- **DRAM latency**: Chips and Cheese reports B200 has "higher VRAM latency than the MI300X, as well as the older H100 and A100" ([source](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)). RTX 3090 GDDR6X is estimated at ~250 ns average ([NVIDIA Forums](https://forums.developer.nvidia.com/t/ddr-latency-of-rtx3090/278865)). Exact HBM3e numbers are not publicly available.
- **Worse occupancy**: 128 CTAs on 148 SMs (0.86/SM) vs 82 SMs (1.56/SM). Fewer warps to hide stalls.
- **ECC**: B200 has ECC enabled, RTX 3090 does not. Minor per-access overhead.

The fla-tma variant (TMA + warp-specialization) is ~1.9x slower than fla-recurrent on B200. TMA descriptor setup overhead likely dominates for such a small data transfer. TMA's benefits (async copies, hardware address generation) only pay off for larger data volumes.

Logs: `logs/nvbench-all-local.log`, `logs/nvbench-all-modal.log`.

### How NVBench solves the GPU bubble (blocking kernel technique)

NVBench's C++ "cold" measurement uses a more principled approach to eliminate the same GPU bubble
([talk by Georgii Evtushenko](http://www.youtube.com/watch?v=CtrqBmYtSEk&t=838), NVBench engineering manager):

```
1. flush_device_l2()         // clear L2 cache
2. sync_stream()             // drain the stream (equivalent to torch.cuda.synchronize)
3. block_stream()            // launch a spinning kernel that HOLDS the GPU busy
4. cuda_timer.start()        // record start event (queued behind blocker)
5. --- user kernel ---       // enqueued behind blocker + start event
6. cuda_timer.stop()         // record end event (queued after kernel)
7. unblock_stream()          // release blocker → GPU executes: start → kernel → end
```

The key insight: NVBench *does* sync the stream (step 2), but then launches a **blocking kernel**
(step 3) that spins on a device-side flag. The start event and benchmark kernel are enqueued behind
the blocker. When `unblock_stream()` flips the flag, the GPU immediately processes the queued work
with no idle bubble between the start event and the kernel.

Source: `nvbench/blocking_kernel.cuh` and `nvbench/detail/measure_cold.cu` in the
[NVBench repo](https://github.com/NVIDIA/nvbench).

### Our fix vs NVBench's approach

Both eliminate the GPU idle bubble, through different mechanisms:

| | NVBench (C++) | Our fix (Python) |
|---|---|---|
| Sync before measurement? | Yes (explicit `sync_stream()`) | No (removed the sync) |
| How bubble is avoided | Blocking kernel holds GPU busy while work is enqueued behind it | CPU pipelines everything onto stream; GPU transitions directly from cache flush to start event to kernel |
| Stream state at start event | Stream has pending work (blocker + start event + kernel) | Stream has pending work (cache flush + clones + start event + kernel) |
| Guarantees clean start? | Yes (sync ensures prior work is done, blocker prevents idle gap) | Mostly (stream ordering ensures prior ops complete, but no explicit fence) |

Both produce equivalent GPU timing results (~5 µs), confirming they measure the same thing.

### Remaining ~1 µs gap (NCU vs benchmarks)

NCU reports pure kernel execution (CUPTI-based). CUDA event timing includes a small residual overhead: event recording latency, and any GPU-side scheduling delay between the start event and the kernel's first instruction. This ~1 µs delta is expected and consistent across NVBench and flashinfer-bench.

### Upstream status

Filed as [flashinfer-bench#195](https://github.com/flashinfer-ai/flashinfer-bench/issues/195). As of 2026-02-21, there is no upstream fix or discussion about this.

The correct pattern (single sync outside the loop) already exists in two related codebases:

- **Triton's `do_bench`** ([`triton-lang/triton/python/triton/testing.py`](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py)): no `synchronize()` inside the benchmark loop.
- **FlashInfer's own `bench_gpu_time_with_cuda_event`** ([`flashinfer/testing/utils.py`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/testing/utils.py)): explicitly comments *"Synchronize once outside of the loop to avoid synchronization overhead"*.
- **PyTorch** has an open issue about the same pattern: [pytorch/pytorch#133053](https://github.com/pytorch/pytorch/issues/133053) ("Stop calling `torch.cuda.synchronize()` in `triton/testing.py` `do_bench`"), filed by Edward Yang.

### Impact on speedup numbers

| Kernel | Old speedup (vs ~1.2 ms reference) | New speedup |
|--------|-----------------------------------|-------------|
| fla-recurrent | ~29x | ~280x |
| fi-baseline | ~35x | ~250x |

The reference implementation time (~1.2 ms) is unaffected since it's dominated by actual compute, not launch overhead.

## NCU Detailed Metrics (RTX 3090, fla-recurrent kernel)

Profile: `profiles/ncu/gdn-decode-fla.ncu-rep`
Kernel: `fused_recurrent_gated_delta_rule_fwd_kernel`, grid=(16,8,1)=(128 blocks), block=256 threads, 31 regs/thread, 16B dynamic shared mem. Duration: **3.84 µs**.

### Warp stall reasons (the smoking gun)

| Stall reason | Ratio (per issue-active) | Meaning |
|---|---|---|
| **Long scoreboard** | **3.79** | Waiting on GMEM/L2/DRAM loads to return |
| Short scoreboard | 1.13 | Waiting on L1/shared/math results |
| Not selected | 0.61 | Eligible warp not picked by scheduler |
| Barrier | 0.38 | `__syncthreads()` / warp barriers |
| MIO throttle | 0.01 | Memory pipe backpressure (negligible) |
| LG throttle | ~0 | Load/store unit backpressure (negligible) |

**Long scoreboard dominates** — warps issue a global load, then stall waiting for the data. With only 0.26 waves (128 blocks on 82 SMs), there aren't enough warps to hide this latency. Low MIO/LG throttle confirms we're not bottlenecked on memory *throughput* — it's purely *latency* not being hidden.

### Cache hit rates

| Level | Hit rate |
|---|---|
| L1/TEX | 34.9% |
| L2 | 57.6% |

L1 hit rate is low because the 128x128 f32 state (64 KB per head) far exceeds L1 capacity. L2 captures some reuse from the q/k vectors being shared across V-tiles.

### Pipe utilization (% of peak, during active cycles)

| Pipe | Utilization | Notes |
|---|---|---|
| LSU (load/store) | 19.9% | Busiest pipe — confirms memory-dominant workload |
| ALU | 17.1% | Gate math (exp, softplus, sigmoid) |
| FMA | 9.3% | Matvec dot products, outer products |
| Tensor cores | **0%** | Not used — scalar FMA only |
| FP64 | 0% | All FP32 compute |

### Occupancy limiters

| Limiter | Max blocks/SM |
|---|---|
| Warps | 6 |
| Registers | 8 |
| Shared mem | 14 |
| SM block limit | 16 |

Warp limit (6 blocks/SM) is the theoretical bottleneck, but in practice the grid is so small (128 blocks / 82 SMs = 1.56 blocks/SM) that we never reach even this limit. Theoretical occupancy is 100%, achieved is 23.6%.

### Coalescing and shared memory

- **Global load efficiency**: ~28 bytes/sector (out of 32 max) — reasonably well-coalesced
- **Global store efficiency**: ~32 bytes/sector — perfectly coalesced
- **Shared memory bank conflicts**: **0** (both loads and stores)

### DRAM traffic breakdown

| Direction | Bytes | Bandwidth |
|---|---|---|
| Reads | 534 KB | 139 GB/s |
| Writes | 24 KB | 6 GB/s |
| **Total** | **558 KB** | **145 GB/s** (15.6% of 936 GB/s peak) |

The 534 KB read matches: 8 heads x 128x128 x 4B = 512 KB state + ~22 KB for q/k/v/scalars. The 24 KB write is much less than expected (512 KB state write) — suggesting L2 write-back coalescing or the state write being absorbed by L2 during the measurement window.

### IPC

| Metric | Value |
|---|---|
| Executed IPC (active) | 0.83 |
| Executed IPC (elapsed) | 0.47 |
| Issue slots busy | 21.6% |

IPC of 0.83 during active cycles is reasonable for a memory-bound kernel, but elapsed IPC drops to 0.47 because SMs are idle ~55% of the time (not enough blocks to fill them).

## Tile shape: [BV, BK] vs [BK, BV]

The FLA kernel originally used `b_h = [BK, BV]` register tile shape, but memory is k-last `[V, K]` — meaning K (stride 1) is the contiguous dimension. The tile shape was transposed relative to memory layout.

**Changed to `[BV, BK]`** so the last dimension (BK) has stride 1 in memory, matching the k-last layout. This ensures coalesced access regardless of Triton's compiler heuristics (default blocked layout maps threads to the last dimension first).

**Benchmark result**: No measurable difference on RTX 3090 (~0.051 ms, ~29x). Triton's coalescing pass likely handled the transposed layout already. Kept the change for correctness by convention.

## Why BV autotune shows no difference

Tested BV={8, 16, 32, 64, 128} — all perform identically at ~0.050 ms. Two opposing effects cancel out:

| BV   | CTAs | Tiles/head | q/k redundancy | Parallelism             |
| ---- | ---- | ---------- | -------------- | ----------------------- |
| 8    | 128  | 16         | 32× each       | High (1.5 CTAs/SM)      |
| 32   | 32   | 4          | 8× each        | Moderate                |
| 128  | 8    | 1          | 1× each        | Low (74/82 SMs idle)    |

- **Larger BV**: fewer CTAs → less redundant q/k loads, but q/k are only ~4-64 KB vs 1 MB state (negligible savings)
- **Smaller BV**: more CTAs → better latency hiding through interleaving, but more scheduling overhead

State traffic (the dominant cost) is identical regardless of BV. The kernel is latency-bound, not bandwidth-bound, so tiling strategy doesn't matter.

## Why BK is not tunable

BK must equal K=128. The matvec `k@S` (sum over K dimension) and outer product `k^T @ v` require the full K dimension. Tiling BK<128 would need two passes over the state (first to compute k@S, then to apply the update), doubling GMEM traffic. Register pressure at BK=128 is trivial: [BV=8, BK=128] = 1024 f32 / 256 threads = 4 registers/thread.

## CUDA Kernel Architecture Experiments (RTX 3090)

We tested three fundamentally different kernel architectures to understand what matters for this latency-bound decode kernel (B=1, 8 heads, 128x128 state per head).

### Designs tested

**v1 (baseline)**: 1 warp (32 threads) per block, K-split across threads, state in registers.
Each thread owns KVEC=4 K-elements across BV V-rows. Dot products require warp `__shfl_xor` reductions. Grid: `(V/BV, B*HV)`.

**v2 (shared memory k/q)**: 4 warps per block, same K-split as v1, but k and q loaded once into shared memory and reused by all warps. Templated on BV_PER_WARP (32, 16, 8). Grid: `(V/(BV*4), B*HV)`.

**v3 (1-thread-per-row, smem state)**: 1 thread per V-row, full K=128 dot products per thread (no warp reductions). State lives in shared memory with +1 padding for bank-conflict-free access. Grid: `(V/32, B*HV)`.

### NVBench results (RTX 3090, sequential, L2 flushed)

| Variant  | BV/thread | Blocks | Regs/thread | GPU Time | BW      | vs v1 |
| -------- | --------- | ------ | ----------- | -------- | ------- | ----- |
| cuda-v1  | 8         | 128    | 64          | 5.61 us  | 188 GB/s| 1.00x |
| cuda-v1b | 16        | 64     | ~96         | 5.79 us  | 182 GB/s| 0.97x |
| cuda-v2c | 8 (x4w)   | 32     | ~120        | 5.98 us  | 176 GB/s| 0.94x |
| cuda-v2b | 16 (x4w)  | 16     | ~170        | 6.66 us  | 158 GB/s| 0.84x |
| cuda-v3  | 1 row     | 32     | low         | 7.15 us  | 148 GB/s| 0.78x |
| cuda-v2  | 32 (x4w)  | 8      | 234         | 8.08 us  | 131 GB/s| 0.69x |

### NVBench results (B200 via Modal, L2 flushed)

| Variant  | Blocks | GPU Time | BW       | vs v1 |
| -------- | ------ | -------- | -------- | ----- |
| cuda-v1  | 128    | 7.10 us  | 149 GB/s | 1.00x |
| cuda-v1b | 64     | 8.03 us  | 131 GB/s | 0.88x |
| cuda-v2c | 32     | 8.36 us  | 126 GB/s | 0.85x |
| cuda-v2b | 16     | 8.84 us  | 119 GB/s | 0.80x |
| cuda-v2  | 8      | 10.51 us | 100 GB/s | 0.68x |
| cuda-v3  | 32     | 11.07 us | 95 GB/s  | 0.64x |

Same ranking as RTX 3090. B200 has 148 SMs (vs 82), making block count even more critical. v3 is relatively worse on B200 (0.64x vs 0.78x on RTX 3090) because 32 blocks covers an even smaller fraction of available SMs.

### Key findings

**1. Block count dominates performance for latency-bound kernels.**
With B=1 and 8 heads, the GPU is severely underutilized no matter what. But more blocks = more SMs active = lower wall-clock time. v1's 128 blocks activates 82 SMs in wave 1 (1.56 waves total). Cutting to 8 blocks (v2, BV=32) leaves 74 of 82 SMs idle. The relationship is monotonic: 128 > 64 > 32 > 16 > 8 blocks.

**2. Register pressure is the hidden killer for fewer-block designs.**
v2 (BV_PER_WARP=32) compiled to 234 registers per thread (plan estimated 188). This caps occupancy at 2 blocks/SM (register-limited). Combined with only 8 total blocks, achieved occupancy was 8.2% vs v1's 3.2% (v1 is low too, but spreads work across more SMs).

**3. Shared memory k/q deduplication does not help.**
v1 redundantly loads k and q from global memory in all 16 blocks per head. v2 loads k/q once into shared memory, shared by 4 warps. But k and q are only 256 bytes each; the redundant loads hit L1/L2 cache and are effectively free. The cost of `__syncthreads` barriers and reduced block count far outweighs the saved bandwidth.

**4. Shared memory state (v3) is slower than register state (v1/v2).**
v3 eliminates warp reductions (no `__shfl_xor`) by giving each thread a full V-row. But it pays with shared memory read-modify-write loops (2 passes of 128 iterations). Even with bank-conflict-free padding (+1 stride), smem throughput cannot match register-to-register FMA throughput.

**5. Tail effect vs SM coverage is a close tradeoff.**
v1 (128 blocks) has a tail effect: wave 2 runs 46 blocks on 82 SMs (44% waste). v1b (64 blocks) fits in 0.78 waves (no tail) but covers only 78% of SMs. Result: v1 edges out v1b by 3%. The marginal SM utilization from filling more SMs in wave 1 slightly outweighs tail waste.

**6. Block dispatch overhead is negligible.**
The GigaThread Engine dispatches all blocks in a grid nearly simultaneously. The overhead for 128 blocks is ~0.5-1 us amortized, not 128x per-block cost. This is why more blocks consistently wins.

**7. `__syncwarp()` is not required for single-warp blocks on current hardware.**
Within a single warp, SIMT lockstep execution guarantees shared memory visibility without explicit fences. Removing `__syncwarp()` from v3 maintained correctness and may allow the compiler more scheduling freedom. However, this relies on hardware behavior not guaranteed by the CUDA memory model.

### NCU comparison: v1 vs v2 (BV=32)

| Metric                  | v1 (128 blk) | v2 (8 blk) |
| ----------------------- | ------------ | ---------- |
| Duration                | 4.64 us      | 9.25 us    |
| Registers/thread        | 64           | 234        |
| Waves per SM            | 0.10         | 0.05       |
| Theoretical occupancy   | 33.3%        | 16.7%      |
| Achieved occupancy      | 3.24%        | 8.21%      |
| Memory throughput       | 120.5 GB/s   | 61.3 GB/s  |
| Executed instructions   | 78,080       | 55,232     |
| IPC (active)            | 0.26         | 0.65       |
| Elapsed cycles          | 6,398        | 12,814     |

v2 is more efficient per-warp (29% fewer instructions, 2.5x higher IPC) but the GPU can't use that efficiency because 74 of 82 SMs are idle.

### NCU comparison: v1 vs v3 (1-thread-per-row, smem state)

| Metric                  | v1 (128 blk) | v3 (32 blk, unrolled) | v3 (no unroll) |
| ----------------------- | ------------ | --------------------- | -------------- |
| Duration                | 4.64 us      | 7.14 us               | 11.23 us       |
| Registers/thread        | 64           | 255 (maxed)           | 56             |
| Local mem spill          | 0            | 256 bytes             | 0              |
| Waves per SM            | 0.10         | 0.08                  | 0.08           |
| Theoretical occupancy   | 33.3%        | 10.42% (smem-limited) | 10.42%         |
| Achieved occupancy      | 3.24%        | 2.09%                 | 2.07%          |
| Memory throughput       | 120.5 GB/s   | 81.1 GB/s             | 51.4 GB/s      |
| Executed instructions   | 78,080       | 53,952                | 63,808         |
| IPC (active)            | 0.26         | 0.23                  | 0.15           |
| Elapsed cycles          | 6,398        | 9,909                 | 15,580         |
| Top stall               | scoreboard   | fixed-latency (31%)   | fixed-latency  |
| Smem load bank confl.   | N/A          | 1.4-way (14.7%)       | 1.4-way        |
| Smem store bank confl.  | N/A          | 1.7-way (31.7%)       | 1.7-way        |

v3 eliminates warp reductions (30% fewer instructions than v1) but has three problems:
1. **Unrolling tradeoff**: Unrolling the 128-iteration smem loops uses all 255 registers + 256B spill, but achieves 0.23 IPC. Without unroll, registers drop to 56 (no spill) but IPC drops to 0.15 and the kernel is 60% slower. The compiler needs unrolling to interleave smem accesses across loop iterations.
2. **Smem store bank conflicts (1.7-way)**: The collaborative state store pattern `state_smem[row * 129 + tid * 4 + i]` conflicts because multiple threads write to the same bank. The +1 stride padding only helps the per-thread compute pattern.
3. **Fixed-latency stalls (31%)**: Smem read-modify-write chains create data dependencies (read h, multiply, write back) that the scheduler cannot hide with only 1 active warp per SM.

### Conclusion

For this kernel at B=1, the optimal strategy is to maximize block count within the v1 architecture (1 warp, K-split, state in registers). The theoretical "better" designs (fewer redundant loads, no warp reductions) all reduce block count, which is the one thing that matters most when the GPU is 95%+ idle. BV=8 with 128 blocks remains the best configuration on RTX 3090.

## v1 BV Sweep + Micro-optimizations (fmaf, streaming stores)

Following the v1/v2/v3 architecture experiments, we tested whether pushing v1 further (smaller BV for more blocks) or applying micro-optimizations could improve performance.

### Changes applied to the v1 template

**1. `fmaf()` for fused multiply-add.** NCU reported 18,432 non-fused FP32 instructions with 26% FMA fusion opportunity. Replaced separate mul+add with `fmaf()` in the delta rule dot products and outer products:
```cpp
// Before:
partial += k_reg[i] * h[bv][i];
h[bv][i] += dv * k_reg[i];
// After:
partial = fmaf(k_reg[i], h[bv][i], partial);
h[bv][i] = fmaf(dv, k_reg[i], h[bv][i]);
```

**2. `__stcs()` streaming stores for state writeback.** State writeback (128 KB per head) pollutes the L2 cache with data not reused within the same kernel invocation. Used `__stcs()` (store cache streaming) to hint the hardware to bypass L2:
```cpp
// Before:
*reinterpret_cast<float4*>(ht + offset) = val;
// After:
__stcs(reinterpret_cast<float4*>(ht + offset), val);
```

**3. BV=4 (v1c, 256 blocks) and BV=2 (v1d, 512 blocks).** Tested whether more blocks could hide memory latency better, at the cost of increased k/q/gate load redundancy (each block reloads the same k, q, gate values for its head).

### NVBench results (RTX 3090, L2 flushed)

All variants include fmaf + streaming store changes.

| Variant  | BV | Blocks | GPU Time | BW       | vs v1 (pre-opt 5.61 us) |
| -------- | -- | ------ | -------- | -------- | ------------------------ |
| cuda-v1  | 8  | 128    | 5.65 us  | 187 GB/s | 0.99x                   |
| cuda-v1c | 4  | 256    | 5.56 us  | 190 GB/s | 1.01x                   |
| cuda-v1d | 2  | 512    | 5.54 us  | 190 GB/s | 1.01x                   |

### NVBench results (B200 via Modal, L2 flushed)

| Variant  | BV | Blocks | GPU Time | BW       | vs v1 (pre-opt 7.10 us) |
| -------- | -- | ------ | -------- | -------- | ------------------------ |
| cuda-v1  | 8  | 128    | 7.12 us  | 148 GB/s | 1.00x                   |
| cuda-v1c | 4  | 256    | 7.32 us  | 144 GB/s | 0.97x                   |
| cuda-v1d | 2  | 512    | 7.30 us  | 144 GB/s | 0.97x                   |
| cuda-v1b | 16 | 64     | 8.09 us  | 130 GB/s | 0.88x                   |

### Analysis

**Micro-optimizations (fmaf, __stcs) had no measurable effect.** v1 measured 5.65 us vs 5.61 us pre-optimization on RTX 3090, and 7.12 us vs 7.10 us on B200. Both are within noise. Likely explanations:
- nvcc may already emit FMA instructions for simple `a * b + c` patterns (fmaf is a hint, not a guarantee of different codegen)
- Streaming stores help when L2 is under pressure, but at <2% bandwidth utilization the L2 is nearly empty

**BV=4 and BV=2 are within noise on RTX 3090 but slightly slower on B200.** On RTX 3090, the extra blocks (256 or 512 vs 128) gave a marginal 1-2% improvement, but this is within the 8% noise margin. On B200 (148 SMs), they are 3% slower. The cost of redundant k/q/gate loads (each block loads the same 260 bytes per head) starts to matter when amortized over fewer V-rows.

**The "more blocks is better" rule has a sweet spot.** The monotonic relationship (more blocks = better) observed across v1/v1b/v2/v3 does not extend below BV=8. At BV=4, each block processes only 4 V-rows (16 float4 state loads + stores) while paying the same fixed overhead for k, q, gate computation, and warp reductions. At BV=2, this overhead is amortized over just 2 rows.

**BV=8 remains optimal across both GPUs.** The v1 architecture with BV=8 (128 blocks) hits the sweet spot between block count and per-block work efficiency. Further tuning of the v1 CUDA kernel is unlikely to yield significant gains at the instruction level; the kernel is fundamentally limited by the small problem size (B=1, 8 heads) leaving >95% of GPU SMs idle.

## V4 kernel: multi-warp latency hiding (the real fix)

### Why v1 was slower than Triton

Comparing NCU profiles of FLA (Triton) and CUDA v1 revealed the root cause: both launch 128 blocks on the same (16, 8) grid, but **Triton uses 8 warps (256 threads) per block while v1 uses 1 warp (32 threads)**. The 8x more warps give the GPU scheduler work to switch to while warps stall on memory:

| Metric | FLA (Triton) | CUDA v1 |
| ------ | ------------ | ------- |
| Block size | 256 (8 warps) | 32 (1 warp) |
| Regs/thread | 31 | 64 |
| IPC (active) | 0.86 | 0.26 |
| Active warps/scheduler | 3.00 | 1.02 |
| Eligible warps/scheduler | 0.36 | 0.17 |
| Achieved occupancy | 24.7% | 3.3% |
| L1 hit rate | 34.9% | 18.6% |
| NCU duration | 3.81 us | 4.70 us |

Triton executes 2.5x more instructions (195K vs 78K) but finishes 19% faster because it has 3x more active warps per scheduler for latency hiding. v1's single warp per block leaves the scheduler idle 83% of cycles.

The v1/v2/v3 architecture experiments missed this because v2 added warps while *also* reducing block count (4 warps, fewer blocks). The correct experiment is to add warps while keeping block count fixed.

### V4 design

1 V-row per warp, NUM_WARPS warps per block, no shared memory, no cross-warp communication. Grid: `(V / NUM_WARPS, B * HV)`. Each warp independently loads state, k, q, v, gates, computes the delta rule update, and stores results. k and q are redundantly loaded by all warps (256 bytes each, hits L1/L2).

With NUM_WARPS=8: grid = (16, 8) = 128 blocks, 256 threads/block = identical parallelism shape to FLA Triton. With NUM_WARPS=4: grid = (32, 8) = 256 blocks, 128 threads/block.

### NCU comparison: v4 vs FLA vs v1

| Metric | FLA (Triton) | CUDA v4 | CUDA v1 |
| ------ | ------------ | ------- | ------- |
| Duration | 3.81 us | 4.22 us | 4.70 us |
| Block size | 256 (8 warps) | 256 (8 warps) | 32 (1 warp) |
| Regs/thread | 31 | 31 | 64 |
| IPC (active) | 0.86 | **1.15** | 0.26 |
| Active warps/sched | 3.00 | **3.01** | 1.02 |
| Eligible warps/sched | 0.36 | **0.47** | 0.17 |
| Achieved occupancy | 24.7% | **25.6%** | 3.3% |
| L1 hit rate | 34.9% | **65.9%** | 18.6% |
| Elapsed cycles | 5,208 | 5,710 | 6,482 |
| Executed instructions | 195,584 | 294,912 | 78,080 |
| Fused FP32 insts | 22,528 | **45,056** | 16,384 |
| Non-fused FP32 insts | 45,056 | 32,768 | 18,432 |
| Uncoalesced excess sectors | 3.7% | **47%** | 14% |

**Note**: Table above shows pre-vectorization numbers. The 47% uncoalesced sectors and high L1 hit rate were fixed by vectorized bf16 loads (see "Vectorized bf16 loads" section below). After the fix: v4 regs dropped to 28, load efficiency matches FLA (28.1/32), duration improved to 4.10 us.

v4 matches or exceeds Triton on occupancy, IPC (1.15 vs 0.86), and eligible warps (0.47 vs 0.36). It uses fewer regs/thread than v1 (28 vs 64) because `h[KVEC]` is only 4 floats instead of `h[BV][KVEC]` = 32 floats.

v4 executes more total instructions (281K vs 195K for FLA, vs 75K for v1). The 8 warps each independently compute gates, load k/q, and do reductions, whereas Triton's compiler may share some of this work across warps internally.

### NVBench results (RTX 3090, L2 flushed)

| Variant | Warps | Blocks | GPU Time | vs FLA |
| ------- | ----- | ------ | -------- | ------ |
| FLA (Triton) | 8 | 128 | 5.25 us | 1.00x |
| cuda-v4b | 4 | 256 | 5.29 us | 0.99x |
| cuda-v4 | 8 | 128 | 5.33 us | 0.99x |
| cuda-v1 | 1 | 128 | 5.65 us | 0.93x |

### NVBench results (B200 via Modal, L2 flushed)

| Variant | Warps | Blocks | GPU Time | vs v1 |
| ------- | ----- | ------ | -------- | ----- |
| cuda-v4 | 8 | 128 | 7.07 us | 1.008x |
| cuda-v1 | 1 | 128 | 7.12 us | 1.00x |

### Key takeaways

1. **Warp count matters as much as block count for latency-bound kernels.** The v1/v2/v3 experiments showed block count dominates, but that was confounded by v2/v3 also changing other things (shared memory, cross-warp sync). v4 isolates the warp count variable: same grid, more warps, no shared memory.

2. **The sweet spot is v4b (4 warps, 256 blocks) on RTX 3090.** It combines the benefits of more warps for latency hiding with more blocks for SM coverage: 256 blocks / 82 SMs = 3.1 blocks/SM, with 4 warps each = 12.4 warps/SM. v4 (8 warps, 128 blocks) trades block count for warp count and lands at about the same performance.

3. **CUDA v4 closes the gap to within 1.5% of Triton on RTX 3090.** The remaining gap was from uncoalesced k/q loads (47% excess sectors). Fixed with vectorized bf16 loads (see "Vectorized bf16 loads" section below).

4. **On B200, v4 and v1 are essentially tied (~7.1 us).** B200 has 148 SMs, so 128 blocks gives <1 block/SM on average. The latency-hiding benefit of more warps is offset by even worse SM coverage. The kernel remains fundamentally grid-size-limited on B200.

## Vectorized bf16 loads (v1 + v4)

### Problem

NCU showed v4 had 47% excessive sectors from uncoalesced global loads (12.2 of 32 bytes utilized per sector). The root cause: individual `__nv_bfloat16` loads (2 bytes each). With KVEC=4 elements per thread at stride 8 bytes between lanes, each 2-byte scalar load wastes 75% of the 32-byte cache sector.

v1 had the same issue (22.7/32 bytes for loads, 14% excessive sectors), plus 8 individual bf16 scalar loads for the v vector (broadcast, so coalescing is fine but instruction count is high).

### Fix

Replaced scalar bf16 loads with vectorized wide loads:
- **q/k (both v1 and v4)**: 4 x 2-byte scalar `__nv_bfloat16` loads replaced with 1 x 8-byte `uint2` load, then reinterpret-cast to `__nv_bfloat162` pairs for unpacking.
- **v (v1 only)**: 8 x 2-byte scalar loads replaced with 1 x 16-byte `uint4` load.

Alignment is guaranteed: `k_base = lane * 4`, byte offset = `lane * 8`, always 8-byte aligned for `uint2`. For v1's v load, `i_v * BV * 2` is always 16-byte aligned (BV=8).

### NCU results (RTX 3090)

**v4 kernel:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Duration | 4.32 us | 4.10 us | -5.1% |
| Global load bytes/sector | 12.2 / 32 | 28.1 / 32 | +130% |
| Excessive sectors | 49,152 (47%) | 0 | Eliminated |
| Memory throughput | 126.70 GB/s | 146.81 GB/s | +15.9% |
| Executed instructions | 294,912 | 281,600 | -4.5% |
| Registers/thread | 31 | 28 | -3 |
| Eligible warps/scheduler | 0.44 | 0.49 | +11% |
| L1 hit rate | 65.9% | 35.6% | Expected drop (fewer redundant hits) |
| Global store bytes/sector | 30.2 / 32 | 10.7 / 32 | Regression (nvcc code-gen quirk) |

**v1 kernel:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Duration | 4.70 us | 4.67 us | -0.6% (noise) |
| Global load bytes/sector | 22.7 / 32 | 31.1 / 32 | +37% (near-perfect) |
| Excessive sectors | 6,144 (14%) | 0 | Eliminated |
| Memory throughput | 118.29 GB/s | 145.59 GB/s | +23.1% |
| Executed instructions | 78,080 | 75,648 | -3.1% |
| Registers/thread | 64 | 64 | Same |
| Global store bytes/sector | 30.2 / 32 | 30.2 / 32 | Same (no regression) |

### Key lessons

1. **Vectorized loads fix coalescing and give a small wall-clock win for v4.** v4 improved 5% in NCU duration (4.32 → 4.10 us); v1 was within noise (4.70 → 4.67 us). The load efficiency gains were dramatic (12.2 → 28.1 bytes/sector for v4, 22.7 → 31.1 for v1), but at <2% DRAM bandwidth utilization the wall-clock gains are modest. The kernel remains fundamentally latency-bound.

2. **L1 hit rate is misleading for scalar loads.** v4's 65.9% L1 hit rate before the fix looked healthy, but it was actually pathological: narrow 2-byte loads were repeatedly hitting the same 32-byte cache lines, inflating the hit rate while wasting bandwidth. After vectorization, the hit rate dropped to 35.6% (matching FLA) because each load now fetches all its data in one shot.

3. **nvcc may reorganize stores when register pressure changes.** v4's store efficiency regressed from 30.2/32 to 10.7/32 despite identical store code. The only change was the load path, which reduced registers from 31 to 28. The compiler likely broke the `float4` store into narrower stores under different register allocation. v1 (64 regs, unchanged) had no store regression. Verify with `cuobjdump --dump-sass` if needed.

4. **`uint2` reinterpret-cast is the right idiom for bf16x4 loads.** The pattern `uint2 raw = *(uint2*)ptr; __nv_bfloat162 pair = *(bfloat162*)&raw.x` guarantees a single 8-byte load instruction. Loading via `__nv_bfloat162*` directly would produce two 4-byte loads (not guaranteed to merge). For bf16x8 (v1's v load), use `uint4`.

## Competition pattern analysis (gpumode NVFP4 competition)

Analyzed 4 winning kernels from the gpumode NVFP4 competition targeting Blackwell B200. Files in `examples/gpumode-nvfp4-competition/`.

### Patterns identified in competition winners

1. **Warp specialization**: TMA warp, MMA warp, epilogue warps (nvfp4_gemm.py)
2. **TMEM + tcgen05 MMA**: Blackwell tensor core access (nvfp4_gemm.py, nvfp4_dual_gemm.py)
3. **TMA (Tensor Memory Accelerator)**: Hardware-driven async bulk copy with tensor map descriptors (nvfp4_gemm.py, nvfp4_group_gemm.py)
4. **L1/L2 cache eviction policies via inline PTX**: `ld.global.L1::no_allocate` for streaming data, `ld.global.L1::evict_last` for reused data (nvfp4_gemv.py)
5. **Multi-stage mbarrier pipeline**: Overlapping loads with compute (nvfp4_gemm.py)
6. **Per-problem-size config tuning**: Config maps keyed by (M, N, K) (nvfp4_dual_gemm.py)
7. **CTA clusters**: `__cluster_dims__` for shared L2 residency (nvfp4_group_gemm.py)

### Why these patterns don't transfer to GDN decode

All competition kernels target GEMM/GEMV: compute-bound or bandwidth-bound workloads processing large matrices. Our GDN decode kernel is latency-bound with tiny data (1 MB total, <2% bandwidth utilization). The patterns that help large workloads are irrelevant or counterproductive for our kernel:

- **Warp specialization / TMA / mbarrier**: Designed to overlap large data transfers with long-running compute. Our compute is ~70 cycles (gate math); nothing to overlap with.
- **TMEM / tensor cores**: Our matvecs are 128-element vectors, too small for tensor core tiles (min 16x16).
- **CTA clusters**: Useful for sharing data across blocks via L2. Our blocks are independent (different V-rows, different heads).
- **Per-problem-size tuning**: All 20 workloads have identical shapes (B=1, all dimensions fixed).

### Cache hint experiments (v5 kernel, removed)

Tested three approaches to cache control, all on the v4 architecture (8 warps, 128 blocks):

**v5 attempt 1: CUDA intrinsics (`__ldcs` / `__stcs`)**
- State loads: `__ldcs` (streaming, bypass L1 entirely)
- State stores: `__stcs` (streaming store, same as v4)
- q/k: default `.ca` (cache all levels)
- Result: no measurable difference vs v4 on RTX 3090 (NVBench: 5.273 vs 5.324 us, ~1% within 7% noise)

**v5 attempt 2: inline PTX with fine-grained eviction hints**
- State loads: `ld.global.L1::no_allocate.v4.f32` (don't allocate in L1 on miss, but hit if already there)
- q/k loads: `ld.global.L1::evict_last.v2.b32` (pin in L1, evict last)
- State stores: `st.global.cs.v4.f32` (streaming, bypass caches)
- Result on RTX 3090: 5.203 vs 5.331 us, ~2.4% faster but within noise
- **Result on B200: 7.302 vs 7.091 us, ~3% SLOWER.** Hardware's default cache management outperforms manual hints.

**v6 (removed): cp.async to SMEM**
- Used inline PTX `cp.async.cg.shared.global` to async-copy state into shared memory while computing gates
- Same mistake as v2/v3: the extra SMEM-to-register hop adds latency that outweighs any overlap benefit
- Result on RTX 3090: 5.479 vs 5.379 us, ~2% slower (NVBench)
- Removed from codebase without testing on B200

### Key lessons from competition pattern experiments

1. **Cache hints are counterproductive for tiny working sets.** With 4KB state per block vs 128KB L1 (RTX 3090) or 228KB (B200), there is no eviction pressure. The default cache policy is optimal.

2. **SMEM staging always loses for this kernel.** Tested three times (v2 shared-memory k/q, v3 shared-memory state, v6 cp.async to SMEM). The extra SMEM-to-register hop adds ~30 cycles per access that cannot be hidden.

3. **Competition patterns target the wrong bottleneck.** GEMM competition kernels are bandwidth-bound or compute-bound. Our kernel is latency-bound (40x gap between theoretical minimum and actual runtime). Techniques that optimize throughput don't help with latency.

4. **The compiler already handles cache well for simple access patterns.** nvcc and the hardware cache controller make good decisions for our straightforward coalesced float4 loads. Manual overrides via PTX or intrinsics add instruction overhead without improving data access timing.
