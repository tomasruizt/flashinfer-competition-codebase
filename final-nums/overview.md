# GDN Decode Kernel: Overview and Learnings

## What this project is

My entry (team lmu-css) for the FlashInfer AI Kernel Generation Contest, GDN
track. GDN is a linear-time recurrent model replacing softmax attention in LLMs
like Qwen3-Next-80B. The decode kernel updates a fixed-size state matrix per head
and produces one output vector per token. I built 7 kernel variants (Triton,
CUDA, FlashInfer baseline, PyTorch eager/compiled), achieving up to ~265x speedup.

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

GPU: RTX 3090 (local), B200 (Modal). All algos pass correctness on all 20 workloads.

## NVBench (trustworthy kernel times)

| Algo          | RTX 3090 | B200     | Beats FI |
|---------------|----------|----------|----------|
| fla-recurrent | 5.28 us  | 7.08 us  | ✅ |
| cuda-v4       | 5.32 us  | 7.34 us  | ✅ |
| cuda-v1       | 5.83 us  | 7.37 us  | ✅ |
| fi-baseline   | 5.54 us  | 8.09 us  | -- |
| fla-tma       | 9.04 us  | 14.70 us | ❌ |
| pt-compiled   | 138.9 us | 283.3 us | ❌ |
| pt-reference  | 276.1 us | 546.8 us | ❌ |

Sorted by B200 time. RTX 3090 ranking is the same except fi-baseline and cuda-v1 swap.

## FI-Bench Local (RTX 3090, with timing fix)

| Algo          | Latency  | Speedup |
|---------------|----------|---------|
| fla-recurrent | 4.37 us  | ~265x   |
| cuda-v4       | 4.47 us  | ~265x   |
| fi-baseline   | 4.72 us  | ~245x   |
| cuda-v1       | 4.82 us  | ~242x   |
| fla-tma       | 7.55 us  | ~154x   |
| pt-compiled   | 680 us   | ~1.75x  |
| pt-reference  | 1370 us  | ~0.89x  |

Agrees with NVBench ranking. Slightly lower absolute times (timing fix removes sync overhead).

## FI-Bench Modal (B200, NO timing fix)

| Algo          | Latency  | Speedup |
|---------------|----------|---------|
| cuda-v4       | 11.6 us  | ~103x   |
| cuda-v1       | 12.0 us  | ~98x    |
| fi-baseline   | 27.3 us  | ~44x    |
| fla-recurrent | 38.2 us  | ~31x    |
| fla-tma       | 37.9 us  | ~31x    |
| pt-compiled   | 712 us   | ~1.6x   |
| pt-reference  | 1381 us  | ~0.88x  |

**Rankings differ from NVBench.** Without the timing fix, `torch.cuda.synchronize()` in the bench loop inflates latencies. CUDA kernels appear fastest because they have less Python dispatch overhead. Triton kernels (fla-recurrent, fla-tma) are penalized most. This ranking is **not representative** of true kernel performance.

## Key Takeaway

The competition uses fi-bench (without our timing fix). On that harness, **cuda-v4 wins at ~103x** despite fla-recurrent being the faster kernel. The sync overhead in fi-bench's `do_bench()` dominates, and CUDA's lower dispatch latency gives it the edge.
