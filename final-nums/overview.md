# GDN Decode Kernel: Overview and Learnings

## What this project is

My solution for the FlashInfer AI Kernel Generation Contest, GDN
decode track.
GDN is a linear-time recurrent model replacing softmax attention in LLMs like Qwen3-Next-80B. The decode kernel updates a fixed-size state matrix per head and produces attention outputs.
The baselines are:

* PyTorch eager
* PyTorch compiled
* A FlashInfer baseline

I built several kernel variants (2 Triton, multiple CUDA), with the best achieving up to ~760x speedup over PyTorch eager and outperforming the FlashInfer baseline.

## What I optimized

The biggest wins came from eliminating unnecessary work around the kernel:
removing redundant state copies between layout formats, fusing small scalar ops
into the main kernel to avoid separate kernel launches, and removing Python-side
dispatch overhead. On the CUDA side, increasing warp count per block to hide
memory latency matched Triton performance.

## What I learned

The kernel is latency-bound, not bandwidth-bound. At batch size 1, the GPU is
95%+ idle with under 2% bandwidth utilization. The bottleneck is DRAM round-trip
latency with too little concurrent work to hide it. This means most standard GPU
optimization techniques (shared memory, cache hints, TMA, warp specialization)
either don't help or actively hurt. I tested each of these multiple times and
every attempt made things worse. The only levers that mattered were maximizing
block count across SMs and maximizing warps per block for latency hiding.

I also found a timing bug in the competition benchmark harness that inflated
microsecond-scale kernels by ~13x. On the unfixed harness, CUDA kernels appear
faster than Triton due to lower Python dispatch overhead, even though the raw
kernel time is slightly worse.

---

## Final Benchmark Numbers (2026-02-24)

GPU: B200 (Modal). All algos pass correctness on all 20 workloads.
RTX 3090 results: [rtx3090.md](rtx3090.md).

## FI bench_gpu_time() (B200)

Uses FlashInfer's `bench_gpu_time(enable_cupti=True)` for pure GPU kernel timing via CUPTI hardware callbacks. This is the same methodology FlashInfer uses in their own benchmarks (PR #2370, #2498). CUPTI strips away all CPU launch overhead, measuring only kernel execution on the GPU.

| Algo          | Median (us) | Speedup | vs FI  |
| ------------- | ----------- | ------- | ------ |
| fla-recurrent | 2.56        | ~760x   | 1.31x  |
| cuda-v4       | 2.62        | ~743x   | 1.28x  |
| cuda-v1       | 3.04        | ~640x   | 1.11x  |
| fi-baseline   | 3.36        | ~579x   | N/A    |
| fla-tma       | 12.70       | ~153x   | 0.26x  |
| pt-compiled   | 835.15      | ~2.3x   | 0.004x |
| pt-reference  | 1946.32     | ~1.0x   | 0.002x |

fla-recurrent edges out cuda-v4 by ~2%. All fast kernels under 3.4 us.

## NVBench (B200)

| Algo          | B200     | Speedup | vs FI |
| ------------- | -------- | ------- | ----- |
| fla-recurrent | 6.60 us  | ~75x    | 1.25x |
| cuda-v4       | 6.62 us  | ~75x    | 1.25x |
| cuda-v1       | 7.82 us  | ~64x    | 1.06x |
| fi-baseline   | 8.27 us  | ~60x    | N/A   |
| fla-tma       | 12.58 us | ~39x    | 0.66x |
| pt-compiled   | 255.4 us | ~1.9x   | 0.03x |
| pt-reference  | 496.6 us | ~1.0x   | 0.02x |

Rankings match CUPTI. Absolute times are ~2x higher because NVBench uses CUDA events, which include ~4 us of CPU launch overhead per iteration (event recording latency, GPU scheduling delays).

## FI Bench (Bug) (B200)

| Algo          | Latency | Speedup |
| ------------- | ------- | ------- |
| cuda-v4       | 11.6 us | ~103x   |
| cuda-v1       | 12.0 us | ~98x    |
| fi-baseline   | 27.3 us | ~44x    |
| fla-recurrent | 38.2 us | ~31x    |
| fla-tma       | 37.9 us | ~31x    |
| pt-compiled   | 712 us  | ~1.6x   |
| pt-reference  | 1381 us | ~0.88x  |

**Rankings differ from FI bench_gpu_time() and NVBench.** Without the timing fix, `torch.cuda.synchronize()` in the bench loop inflates latencies. CUDA kernels appear fastest because they have less Python dispatch overhead. Triton kernels (fla-recurrent, fla-tma) are penalized most. This ranking is **not representative** of true kernel performance.

## Timing tiers

Three measurement methods give three different absolute times, but the same ranking:

| Tier         | Method                               | What's measured                 | B200 fla-recurrent |
| ------------ | ------------------------------------ | ------------------------------- | ------------------ |
| CUPTI        | `bench_gpu_time(enable_cupti=True)`  | Pure GPU kernel time            | 2.56 us            |
| CUDA Events  | NVBench / `bench_gpu_time()` default | Kernel + event record overhead  | 6.60 us            |
| Sync-in-loop | fi-bench (buggy `do_bench`)          | Kernel + full CPU sync per iter | 38.2 us            |

The ~4 us gap between CUPTI and CUDA events is CPU launch overhead. The ~30 us gap between CUDA events and fi-bench is the `torch.cuda.synchronize()` bug in the benchmark harness.

## Key Takeaway

The competition uses fi-bench (without our timing fix). On that harness, **cuda-v4 wins at ~103x** despite fla-recurrent being the faster kernel. The sync overhead in fi-bench's `do_bench()` dominates, and CUDA's lower dispatch latency gives it the edge.
