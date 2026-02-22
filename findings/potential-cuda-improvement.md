# Optimizing a Latency-Bound GPU Kernel: GDN Decode State Update

## Scope

This document covers how to minimize the wall-clock runtime of a **single** Gated Delta Network recurrent state update kernel on an NVIDIA B200 GPU, at the operating point `batch_size=1, time_step=1`.

We are not optimizing the full model forward pass. We are optimizing one kernel invocation that computes the GDN recurrence for one token.

---

## 1. What the Kernel Computes

The gated delta rule recurrence at decode time is:

```
S_t = alpha_t * S_{t-1} + beta_t * (v_t * k_t^T - alpha_t * (k_t^T @ S_{t-1}) * k_t^T)
o_t = q_t @ S_t
```

### Concrete dimensions

| Tensor      | Shape                      | Dtype | Size per head             | Total (8 heads) |
| ----------- | -------------------------- | ----- | ------------------------- | --------------- |
| State S     | `[B=1, H=8, V=128, K=128]` | fp32  | 128 * 128 * 4 = **64 KB** | **512 KB**      |
| q, k        | `[B=1, H=8, K=128]`        | bf16  | 256 B                     | 2 KB            |
| v           | `[B=1, H=8, V=128]`        | bf16  | 256 B                     | 2 KB            |
| alpha, beta | `[B=1, H=8]`               | fp32  | 4 B                       | 32 B            |
| output o    | `[B=1, H=8, V=128]`        | bf16  | 256 B                     | 2 KB            |

The state is kept in fp32 for numerical stability during accumulation. The inputs (q, k, v) and output are typically bf16.

### Per-head work breakdown

| Operation                 | FLOPs              | Description                                      |
| ------------------------- | ------------------ | ------------------------------------------------ |
| `k^T @ S`                 | 128 * 128 = 16,384 | Matrix-vector product, yields vector of size 128 |
| `alpha * (k^T @ S) * k^T` | ~384               | Scalar-vector ops                                |
| `v * k^T`                 | 128 * 128 = 16,384 | Outer product, yields 128x128 matrix             |
| `alpha * S`               | 128 * 128 = 16,384 | Scalar-matrix multiply                           |
| `S_t = ...`               | 128 * 128 = 16,384 | Matrix addition/subtraction                      |
| `q @ S_t`                 | 128 * 128 = 16,384 | Matrix-vector product for output                 |
| **Total per head**        | **~82K FLOPs**     |                                                  |
| **Total (8 heads)**       | **~656K FLOPs**    |                                                  |

For reference, B200 can do ~2.25 PFLOP/s in bf16 (less in fp32, but still enormous). A 656K FLOP kernel would take **0.0003 us** if compute-bound. The kernel will never be compute-bound.

### Per-head data movement

| Data                | Direction | Size per head |
| ------------------- | --------- | ------------- |
| State S             | read      | 64 KB         |
| State S             | write     | 64 KB         |
| q, k, v             | read      | ~768 B        |
| output o            | write     | ~256 B        |
| **Total per head**  |           | **~129 KB**   |
| **Total (8 heads)** |           | **~1 MB**     |

At B200's ~8 TB/s HBM bandwidth: 1 MB takes ~0.12 us. The kernel is not bandwidth-bound either.

**Conclusion**: The kernel's runtime is dominated by fixed-cost latency, not by the volume of compute or data movement.

---

## 2. Where Latency Comes From

### 2.1 Kernel launch overhead

From CPU `cudaLaunchKernel` call to the first instruction executing on the GPU: typically 3-10 us via the CUDA runtime API. This single cost may exceed the kernel's useful compute time.

Sources of launch latency:
- CPU-side driver validation and parameter marshaling
- Enqueueing work to the GPU's command processor
- GPU-side block scheduling and dispatch to SMs

### 2.2 Global memory read latency

Even if data is in L2 cache, the first load from any address has a latency of ~200 cycles (~100 ns at 2 GHz). From HBM: ~400+ cycles. For a kernel that does very little compute, these initial loads are a large fraction of total runtime.

The state S is the dominant data. At 64 KB per head in fp32, that's 512 cache lines (128 bytes each). Even with perfect coalescing (all threads in a warp load consecutive addresses, so one cache line serves the whole warp), the very first load to each cache line incurs the full latency. Subsequent loads to nearby cache lines can overlap thanks to the memory system's ability to have many outstanding requests, but there is still a minimum latency floor for the first load to return.

### 2.3 Instruction issue latency and ILP

**ILP (Instruction-Level Parallelism)** is the GPU's ability to execute multiple independent instructions from the same warp in overlapping fashion, without waiting for each one to finish before starting the next.

For example, if instruction B depends on the result of instruction A, the warp must stall until A completes. But if instructions A and B are independent (they read/write different registers), the scheduler can issue B immediately after A, and both execute concurrently in different pipeline stages.

Why this matters for our kernel: The GDN recurrence has data dependencies -- you can't compute `q @ S_t` until you've finished updating S. These serial dependencies create chains where the warp scheduler has nothing to issue while waiting. With very few warps active (low-occupancy, latency-bound kernel), there aren't other warps to switch to during these stalls.

Ways to increase ILP within the kernel:
- **Interleave independent work**: While one matvec result is being accumulated, start loading data for the next operation
- **Unroll loops**: The compiler can find more independent instructions in unrolled code
- **Process multiple rows of S simultaneously**: If thread 0 is working on S[0,:] and thread 1 on S[1,:], their instructions are independent

### 2.4 TLB misses

**TLB (Translation Lookaside Buffer)** is a small hardware cache that translates virtual memory addresses to physical memory addresses. Every memory access the GPU makes uses virtual addresses (pointers in your kernel). These must be translated to physical addresses (actual locations in HBM) before the memory system can fetch the data. The TLB caches recent translations so they're fast (~1 cycle). A TLB miss requires a multi-step page table walk, which costs ~20-50 extra cycles on top of the normal memory latency.

Why this matters: The GPU's TLB has limited entries. If the state S spans many memory pages, and those pages haven't been accessed recently, the first access to each page incurs a TLB miss penalty on top of the normal cache/memory latency.

For our kernel: 64 KB per head = 16 pages at 4 KB page size, or fits within 1 page at 2 MB page size. With 8 heads = 128 pages (4 KB) or 1 page (2 MB). Standard `cudaMalloc` on modern NVIDIA GPUs already uses 2 MB pages for allocations above a threshold, so TLB pressure is usually handled automatically. If profiling reveals TLB stalls, ensure the state allocation is large enough to trigger large-page allocation.

### 2.5 Synchronization overhead

If the kernel uses shared memory barriers (`__syncthreads`), each barrier adds ~20 cycles and forces all threads in the block to wait for the slowest thread. With multiple synchronization points (load state, compute, store state), these add up.

### 2.6 Kernel retirement

After the last instruction, the GPU must flush all pending writes to the memory system and retire the kernel. This adds tail latency before the next kernel (or CPU synchronization) can proceed.

---

## 3. Where Should the State Live?

This is the most important design decision. Let's think through it carefully.

### 3.1 The key distinction: within a kernel vs. between kernels

**Within a single kernel invocation**, the state absolutely should live in registers or shared memory. The entire computation (load S, update S, compute output, store S) happens inside one kernel call. While executing, the state sits in the fastest memory you can fit it in.

**Between kernel invocations** is the problem. When the kernel exits, registers and shared memory are freed. The next time the kernel runs (for the next token), S must be reloaded from wherever it was stored -- and that "wherever" is global memory (HBM), potentially cached in L2.

So L2 persistence is not an alternative to registers/shared memory -- it's complementary. The data flow is:

```
Kernel starts:
  HBM (or L2 if pinned) --> Registers / Shared Memory    [LOAD]
  
Kernel computes:
  Registers / Shared Memory                               [COMPUTE]

Kernel ends:
  Registers / Shared Memory --> HBM (hopefully stays in L2) [STORE]
  
...time passes, other kernels run...

Next invocation:
  L2 cache (if still resident) --> Registers / Shared Memory [LOAD, fast!]
  HBM (if evicted from L2) --> Registers / Shared Memory    [LOAD, slow!]
```

Pinning S in L2 ensures that the LOAD at the start of each invocation hits L2 (~200 cycles) rather than HBM (~400+ cycles). For 512 KB of state, that's the difference between ~25 ns and ~60 ns of load time.

### 3.2 Could we avoid the round-trip entirely?

Yes -- with a **persistent kernel** that never exits. The kernel stays resident on the GPU, looping and waiting for new work (the next token's q, k, v inputs). The state stays in registers or shared memory permanently. No load/store of S at all after the initial load.

This is what FlashRNN does for traditional RNNs. It eliminates the load/store overhead entirely but requires a fundamentally different execution model (the kernel must have a way to receive new inputs and signal completion without exiting).

### 3.3 Register vs. shared memory for S within the kernel

**State size per head**: 128 * 128 * 4 bytes (fp32) = **64 KB**

**Registers**:
- B200 has 256 KB of register file per SM, with a maximum of 255 32-bit registers per thread
- 64 KB of state / 32 threads per warp = 2 KB per thread = **512 fp32 registers per thread**
- This exceeds the 255-register hardware limit. One warp is not enough.
- With 2 warps (64 threads): 1 KB per thread = 256 registers. Still over the 255 limit by 1.
- With 4 warps (128 threads): 512 B per thread = **128 registers per thread**. This fits!
- With 4 warps, total register usage: 128 * 128 threads = 16,384 registers = 64 KB. That's 25% of the SM's 256 KB register file. Feasible.

So **yes, the state can live in registers** with 4 warps, each thread holding 128 fp32 values (a 128-element row of S). This is the fastest possible option. No shared memory loads needed for S access during compute.

**Trade-off**: With 128 registers per thread for S alone, the compiler needs additional registers for temporaries (loop counters, intermediate values, pointers, vector elements). If total register pressure exceeds ~160 regs/thread, the compiler will spill to local memory (which goes through L1/L2), destroying the benefit. Use `--maxrregcount=160` or `__launch_bounds__` to control this.

**Shared memory**:
- 64 KB per head fits easily in B200's 227 KB shared memory budget
- Access latency: ~30 cycles per access (vs. ~1 cycle for registers)
- Bank conflicts: shared memory has 32 banks. If multiple threads in a warp access the same bank, accesses serialize. The 128x128 fp32 matrix layout must be designed to avoid this (e.g., padding rows to avoid stride conflicts).

**Recommendation**: Try registers first (4 warps, 128 regs/thread for S). If register spilling becomes a problem, fall back to shared memory with careful bank-conflict-free layout.

---

## 4. Optimization Levers

### 4.1 Reduce launch overhead

**Within a single kernel launch**, you cannot eliminate launch overhead -- it's the cost of getting onto the GPU. But you can minimize it:

- **Use the CUDA driver API** (`cuLaunchKernel`) instead of the runtime API (`cudaLaunchKernel`). The driver API skips some runtime-layer overhead. Saves ~1-2 us.

- **Pre-warm the kernel**: The first launch of any kernel is slower (code must be loaded to instruction cache). Ensure a warmup call has already happened.

- **Minimize grid size**: Fewer blocks = less dispatch overhead. For 8 heads, launch exactly 8 blocks.

- **Avoid Python/Triton dispatch**: Triton adds Python-side overhead on top of CUDA launch. For a single latency-critical kernel, calling a pre-compiled CUDA kernel directly from C++ (or via `torch.ops` with minimal wrapper) saves microseconds.

### 4.2 Pin state S in L2 cache (for the between-invocations problem)

As explained in Section 3.1, this ensures fast loads when the kernel starts:

```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = state_ptr;
attr.accessPolicyWindow.num_bytes = 512 * 1024; // 512 KB for all 8 heads
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

B200 (GB200) has 126 MB of L2. Our 512 KB state is 0.4% of that. Trivially fits.

### 4.3 Load S once, compute everything in-register, store once

This is the core kernel structure. For one head, with 4 warps of 32 threads (128 threads total), each thread owns one row of S (128 fp32 values):

```cuda
// Each thread loads its row of S into registers
float s[128];  // 128 fp32 values = 512 bytes = 128 registers
for (int i = 0; i < 128; i++) {
    s[i] = S_global[head_id * 128 * 128 + threadIdx.x * 128 + i];
}

// --- All computation happens in registers from here ---

// Step 1: Compute k^T @ S (matvec)
// Each thread has row threadIdx.x of S, compute dot(k, S[threadIdx.x, :])
float ks = 0.0f;
for (int i = 0; i < 128; i++) {
    ks += k[i] * s[i];  // k broadcast via shared mem or __shfl
}
// ks is now (k^T @ S)[threadIdx.x]

// Step 2: Update S in-place in registers
// S_new = alpha * S + beta * (v * k^T - alpha * (k^T @ S) * k^T)
float correction = alpha * ks;  // scalar per thread
for (int i = 0; i < 128; i++) {
    float outer = v[threadIdx.x] * k[i];  // rank-1 update element
    s[i] = alpha * s[i] + beta * (outer - correction * k[i]);
}

// Step 3: Compute output o = q @ S_t
float out = 0.0f;
for (int i = 0; i < 128; i++) {
    out += q[i] * s[i];
}
// Reduce 'out' across threads to get final output vector

// --- Store updated S back ---
for (int i = 0; i < 128; i++) {
    S_global[head_id * 128 * 128 + threadIdx.x * 128 + i] = s[i];
}
```

The key insight: S never leaves registers during the entire computation. The only global memory traffic is the initial load and final store.

### 4.4 Use warp shuffles for vector broadcasts

The k and q vectors (128 elements each) need to be accessible by all 128 threads. Options:

**Option A: Shared memory broadcast**
```cuda
__shared__ float k_shared[128];
if (threadIdx.x < 128) k_shared[threadIdx.x] = k_global[head_id * 128 + threadIdx.x];
__syncthreads();
// All threads read k_shared[i] -- 30 cycle latency per access
```

**Option B: Warp shuffle broadcast (faster, but only within a warp)**
Within a warp (32 threads), use `__shfl_sync` to broadcast vector elements:
```cuda
// k is distributed: thread i in the warp holds k[i] for i < 32
// To get k[j], shuffle from lane j
float k_j = __shfl_sync(0xffffffff, my_k_element, j);
```
This costs ~1 cycle per shuffle vs. ~30 cycles for shared memory. But it only works within a single warp (32 threads). Since k has 128 elements and we have 4 warps, the hybrid approach is:
- Store k in shared memory once (one `__syncthreads`)
- Each warp loads its 32 needed elements from shared memory into registers
- Then use shuffles within the warp for the inner compute loop

This way the inner loop (128 iterations per dot product) only pays shared memory latency once per warp, then uses register-speed shuffles.

### 4.5 Minimize synchronization points

Every `__syncthreads()` costs ~20 cycles and serializes the block. With 4 warps per block, you need barriers only when warps share data:

Target: 2-3 total sync points in the entire kernel:
1. After loading k into shared memory
2. After loading q into shared memory (can combine with #1 if both loaded at once)
3. Before cross-warp reduction for the output vector

Avoid barriers within the main compute loops. Each warp works independently on its own rows of S.

### 4.6 Maximize ILP within the compute loop

The inner loops (dot products and updates) have natural ILP. Help the compiler find it:

**Loop unrolling**: Unroll by 4 so the compiler can interleave independent multiply-adds:
```cuda
// Without unrolling: serial dependency chain on acc
for (int i = 0; i < 128; i++)
    acc += k[i] * s[i];

// With unrolling: 4 independent accumulator chains
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
for (int i = 0; i < 128; i += 4) {
    acc0 += k[i+0] * s[i+0];
    acc1 += k[i+1] * s[i+1];
    acc2 += k[i+2] * s[i+2];
    acc3 += k[i+3] * s[i+3];
}
float acc = acc0 + acc1 + acc2 + acc3;
```

With 4 independent accumulators, the FMA (fused multiply-add) pipeline can have 4 instructions in flight simultaneously, instead of each waiting for the previous one's result. On B200, the FMA pipeline latency is ~4 cycles, so 4-way unrolling keeps the pipeline full.

**Interleave load and compute**: If loading k from shared memory, interleave loads with arithmetic so the 30-cycle shared memory latency is hidden by computation:
```cuda
float k0 = k_shared[i];     // load (30-cycle latency)
acc += k_prev * s_prev;     // compute (overlaps with load)
float k1 = k_shared[i+1];   // next load (overlaps with compute)
acc += k0 * s[i];           // compute (by now k0 has arrived)
```

### 4.7 Parallelization strategy

With 8 heads and the state fitting in registers with 4 warps:

**Recommended: 8 blocks, 128 threads each (4 warps per block)**
- Grid: `(8,)` -- one block per head
- Block: `(128,)` -- 4 warps, each thread holds one row of S (128 fp32 values)
- Total register usage: 8 blocks * 128 threads * ~160 regs = needs 8 SMs (out of 148)
- All heads run in parallel, completely independent

This launches only 8 blocks on a 148-SM GPU. 140 SMs are idle. That's fine -- we're optimizing latency, not utilization.

### 4.8 Reduce precision where safe

The state is fp32 for accumulation stability. But consider:

- **fp32 state with bf16 inputs**: This is the current setup. The fp32 accumulation in the state update is important because S accumulates many small updates over thousands of tokens.
- **Could S be bf16?** This would halve the state to 32 KB per head (64 regs/thread with 4 warps -- very comfortable). But the GDN recurrence involves subtracting `alpha * (k^T @ S) * k^T` from the rank-1 update, which can cause cancellation errors in low precision. The FLA issue tracker has reports of numerical issues even in bf16. Probably not safe to change.

### 4.9 Fuse surrounding ops

Every separate kernel launch costs 3-10 us. If the GDN kernel is preceded by the sigmoid/softplus that computes `alpha` from the raw projection, and followed by output normalization/gating, fusing these into the same kernel eliminates 2 launch boundaries = 6-20 us saved. This is "free" latency reduction.

### 4.10 Triton-specific considerations

If using Triton (rather than raw CUDA):

**Hard-code the config**: For bs=1, ts=1 with fixed H=8, V=K=128, skip autotuning:

```python
@triton.jit
def gdn_decode_kernel(
    S_ptr, q_ptr, k_ptr, v_ptr, o_ptr, alpha_ptr, beta_ptr,
    stride_sh, stride_sv, stride_sk,
    H: tl.constexpr,   # 8
    V: tl.constexpr,    # 128
    K: tl.constexpr,    # 128
):
    head_id = tl.program_id(0)
    # ...
```

**Use `tl.constexpr`** for H, V, K so Triton unrolls all loops at compile time.

**Eviction policies**:
```python
S_tile = tl.load(S_ptr + offsets, eviction_policy='evict_last')   # keep S in cache
k_vec = tl.load(k_ptr + offsets, eviction_policy='evict_first')   # k used once
```

**Limitation**: Triton manages registers implicitly. You cannot force S into registers -- Triton's compiler decides whether to use registers or shared memory. For maximum control over the register-vs-shared-memory decision, raw CUDA is the way to go.

---

## 5. B200 (Blackwell) Hardware Details

### 5.1 Memory hierarchy

| Level         | Size                      | Latency      | Relevance to our kernel                                        |
| ------------- | ------------------------- | ------------ | -------------------------------------------------------------- |
| Registers     | 256 KB/SM, max 255/thread | ~1 cycle     | **S lives here during compute** (128 regs/thread with 4 warps) |
| Shared Memory | up to 227 KB/block        | ~30 cycles   | Broadcast vectors (k, q, v); fallback if register spills       |
| L1 Cache      | Unified with shared mem   | ~30 cycles   | Auto-caches spills and small reads                             |
| L2 Cache      | 126 MB (GB200)            | ~200 cycles  | **S lives here between invocations**                           |
| HBM3e         | 192 GB, ~8 TB/s           | ~400+ cycles | Cold-start penalty if S is evicted from L2                     |
| TMEM          | 256 KB/SM dedicated       | Specialized  | Not useful -- our matmuls are too small for tensor cores       |

### 5.2 Register budget check

Our target: 128 fp32 registers per thread for S, plus ~30 for temporaries (accumulators, loop counters, pointers, vector elements).

Total: ~160 registers per thread. With 128 threads per block (4 warps), that's 20,480 registers per block. B200 has 65,536 registers per SM (256 KB / 4 bytes). Our block uses 31% of the SM's register file. Comfortable margin.

Use `__launch_bounds__(128, 1)` to tell the compiler: 128 threads per block, 1 block per SM maximum. This gives the compiler permission to use up to 255 registers per thread without worrying about multi-block occupancy.

### 5.3 SM count and dispatch

B200 has 148 SMs (74 per die, dual-die design). We launch 8 blocks, using 8 SMs. The other 140 SMs are idle.

The dual-die design means our 8 blocks could land on either die. If S is allocated on die 0's HBM but a block runs on die 1, there's a cross-die penalty via the NV-HBI link (~1.5 TB/s, extra ~100 ns). In practice, the CUDA scheduler tends to keep blocks near their data, and with L2 persistence the cross-die effect is reduced.

### 5.4 TMEM (Tensor Memory)

TMEM is B200's dedicated 256 KB/SM memory for tensor core intermediates, accessed via `tcgen05` PTX instructions.

**Not useful here.** Our matrix-vector products are GEMV operations (M=1, N=128, K=128). Tensor cores require tile sizes of at least 16x16. With M=1, you'd waste 15/16 of the tensor core's capacity. Scalar FMA instructions on CUDA cores are more efficient for this shape. Blackwell microbenchmarks confirm TMEM adds 3-5% overhead for small problems that fit in L1.

### 5.5 Clock frequencies

B200 boost clocks are typically ~2.1 GHz. Lock them for measurement:

```bash
nvidia-smi -lgc 2100   # lock to max boost
nvidia-smi -rgc         # reset after benchmarking
```

At 2.1 GHz: 1 cycle = ~0.48 ns. A 200-cycle L2 access = ~95 ns. A 30-cycle shared memory access = ~14 ns. A 1-cycle register access = ~0.48 ns.

---

## 6. Profiling the Kernel

### 6.1 What to measure

Standard GPU performance metrics (FLOP/s, bandwidth utilization) are meaningless for this kernel since you'll never approach hardware limits. Instead:

- **Kernel duration** (ns): The number you're optimizing. Measure via CUDA events or Nsight Systems.
- **Warp stall breakdown** (Nsight Compute): Which stalls dominate?
  - `stall_memory_dependency`: Warp waiting for a memory load to return. Indicates you should prefetch or restructure loads.
  - `stall_barrier`: Warp waiting at `__syncthreads`. Indicates too many sync points.
  - `stall_not_selected`: Warp is ready but scheduler picked another warp. Means low ILP -- the scheduler has nothing useful to issue for this warp.
  - `stall_math_pipe_throttle`: Rare for this kernel, would indicate compute saturation.
- **L2 hit rate**: Should be close to 100% if S is pinned. If not, you're paying HBM latency.
- **Register spills**: Check with `--ptxas-options=-v` during compilation. Any spills to local memory add L1/L2 latency on every access to the spilled value.
- **Shared memory bank conflicts**: If using shared memory for vector broadcasts, check `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` in Nsight Compute.

### 6.2 Clock-cycle profiling within the kernel

To find exactly where cycles are spent:

```cuda
__global__ void gdn_decode_instrumented(...) {
    long long t0 = clock64();
    
    // Phase 1: Load state S into registers
    float s[128];
    for (int i = 0; i < 128; i++)
        s[i] = S_global[...];
    long long t1 = clock64();
    
    // Phase 2: Compute k^T @ S
    float ks = 0.0f;
    // ...
    long long t2 = clock64();
    
    // Phase 3: Update S in registers
    // ...
    long long t3 = clock64();
    
    // Phase 4: Compute q @ S_t and reduce output
    // ...
    long long t4 = clock64();
    
    // Phase 5: Store S back to global memory
    for (int i = 0; i < 128; i++)
        S_global[...] = s[i];
    long long t5 = clock64();
    
    if (threadIdx.x == 0) {
        printf("Load: %lld, k@S: %lld, update: %lld, q@S: %lld, store: %lld cycles\n",
               t1-t0, t2-t1, t3-t2, t4-t3, t5-t4);
    }
}
```

This tells you which phase to attack. Typically for this kernel size, Load and Store will dominate.

---

## 7. Reference: Existing Implementations

The FLA library's `fused_recurrent_gated_delta_rule` kernel (Triton) is used for decode. It:
1. Parallelizes over `(batch, head)` in the grid
2. Loads the full state S for one head into SRAM
3. Loops over timesteps (just 1 in our case)
4. Computes the recurrence in-SRAM
5. Stores the updated S and output

The FLA code notes: "launching the triton kernel for just one token will actually be slower" -- launch overhead dominates for very short sequences.

The vLLM PR #30860 (`fused_sigmoid_gating_delta_rule_update`) fuses the sigmoid gating (producing `alpha` from the raw logit) directly into the recurrence kernel, eliminating one kernel boundary.

---

## 8. Priority-Ordered Optimization Checklist

### Must-do

1. **Hold S in registers (4 warps, 128 regs/thread for S)**. Load from global memory once at kernel start, compute everything in-register, store once at the end. Zero intermediate memory traffic for S.

2. **Pin S in L2** (`cudaAccessPolicyWindow`). Ensures the load at kernel start hits L2 (~200 cycles) not HBM (~400+ cycles). Only 512 KB, trivially fits in B200's 126 MB L2.

3. **Load S once, compute everything, store once**. Structure the kernel so the load-compute-store is a single pass with no redundant S accesses.

4. **Use `__launch_bounds__(128, 1)`**. Tells the compiler to optimize for 128 threads and 1 block per SM maximum, allowing maximum register usage per thread.

### Should-do

5. **Warp shuffles for k/q broadcast within each warp**. Avoids shared memory latency in the inner compute loops. Use shared memory only for the initial inter-warp broadcast of the 128-element vectors.

6. **Unroll inner loops by 4**. Gives the compiler 4 independent FMA chains to interleave, keeping the ~4-cycle FMA pipeline full (maximizes ILP).

7. **Minimize `__syncthreads`** to 2-3 total: one after loading k into shared memory, one after loading q, and one before the cross-warp output reduction.

8. **Fuse surrounding ops into the kernel**. The sigmoid/softplus producing `alpha`, and the output normalization/gating. Each eliminated kernel boundary saves 3-10 us.

### Worth trying

9. **Persistent kernel**. Never exit; loop and wait for new inputs. Eliminates launch overhead and S load/store entirely after the first invocation. Highest engineering effort.

10. **Experiment with 2 warps vs 4 warps**. 2 warps = 256 regs/thread, which exceeds the 255 limit by 1 (would need to spill 1 register or restructure slightly). 4 warps = 128 regs/thread, comfortable but less ILP per thread. Profile both.

11. **CUDA driver API launch**. If calling from Python, bypass PyTorch's dispatcher with a thin C++ extension calling `cuLaunchKernel` directly.

12. **Memory layout of S**. Store S contiguously per head: `S[head][row][col]` so each thread's 128-element row is a contiguous 512-byte region. This maximizes cache line utilization during the initial load (4 cache lines per thread, all sequential).
