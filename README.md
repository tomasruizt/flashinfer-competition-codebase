# GDN Decode Kernel: Overview and Learnings

## The Project

This is my solution for the FlashInfer AI Kernel Generation Contest, GDN decode track.
GDN (Gated Delta Net) is a linear-time recurrent model that replaces softmax attention in LLMs like Qwen3-Next-80B.
The decode kernel updates a fixed-size state matrix per head and produces attention outputs.
I compared my solution against three baselines:

* PyTorch eager
* PyTorch compiled
* The FlashInfer CuTe-DSL kernel

I built several kernel variants (2 Triton, multiple CUDA). The best achieves ~760x speedup over PyTorch eager and is ~1.3x faster than the FlashInfer baseline (2.56 µs vs 3.36 µs on B200, measured via CUPTI).

## Optimizations that Work

The biggest wins came from eliminating unnecessary work around the kernel:

1. **Contiguous Memory**: Operating directly on the contiguous memory layout, and avoiding the transpose + contiguous call.
2. **Fusion**: Computing gates inside the kernel instead of launching separate CUDA kernels.

The kernel is latency-bound, rather than compute- or bandwidth-bound (under 2% bandwidth utilization at batch size 1), so the only on-GPU lever that mattered was hiding latency:
I.e. by increasing block count across SMs and warps per block to hide GMEM load latency.

A **side quest** was finding a timing bug in the competition benchmark harness ([flashinfer-bench#195](https://github.com/flashinfer-ai/flashinfer-bench/issues/195)) that inflated microsecond-scale kernels by ~13x.
I submitted the issue to the competition organizers.
They acknowledged the bug and suggested using CUPTI timings instead.

## Optimizations that Don't Work

Because the kernel is latency-bound with ~25% occupancy, most standard GPU optimization techniques either don't help or actively hurt.
I tested shared memory staging (3 kernel variants), L1/L2 cache hints, TMA async copies, warp specialization, and cp.async pipelines.
Every attempt made things worse.
These techniques target bandwidth- or compute-bound kernels; with so little data in flight, the extra instructions and synchronization barriers just add latency.
This is likely why the FlashInfer CuTe-DSL kernel underperforms: It stages data through shared memory, which adds overhead and barriers that hurt the latency.

## Results

GPU: B200 via [Modal](https://modal.com) (serverless cloud GPU platform). All algorithms pass correctness on all 20 workloads.

## CUPTI Benchmarks

This comparison uses FlashInfer's `bench_gpu_time(enable_cupti=True)` for pure GPU kernel timing via CUPTI (CUDA Profiling Tools Interface) hardware callbacks, the same methodology FlashInfer uses in their own benchmarks ([PR #2370](https://github.com/flashinfer-ai/flashinfer/pull/2370), [PR #2498](https://github.com/flashinfer-ai/flashinfer/pull/2498)).
CUPTI strips away all CPU launch overhead, measuring only kernel execution on the GPU.
Speedup is relative to the PyTorch eager baseline; the last column is relative to the FlashInfer CuTe-DSL kernel.

| Algorithm            | Median (µs) | Speedup | vs FlashInfer |
| -------------------- | ----------- | ------- | ------------- |
| fla-recurrent (mine) | 2.56        | ~760x   | 1.31x         |
| cuda-v4 (mine)       | 2.62        | ~743x   | 1.28x         |
| cuda-v1 (mine)       | 3.04        | ~640x   | 1.11x         |
| fi-baseline          | 3.36        | ~579x   | 1.00x         |
| fla-tma (mine)       | 12.70       | ~153x   | 0.26x         |
| pt-compiled          | 835.15      | ~2.3x   | 0.004x        |
| pt-reference         | 1946.32     | ~1.0x   | 0.002x        |

Raw log: [fi-timing-modal/all.txt](final-nums/fi-timing-modal/all.txt)

## NVBench

[NVBench](https://github.com/NVIDIA/nvbench) is NVIDIA's official C++ kernel benchmarking framework.
It uses CUDA events for timing, with L2 cache flushing, GPU thermal throttling detection, and statistical convergence to reduce runtime variability.

| Algorithm            | Median (µs) | Speedup | vs FlashInfer |
| -------------------- | ----------- | ------- | ------------- |
| fla-recurrent (mine) | 6.60        | ~75x    | 1.25x         |
| cuda-v4 (mine)       | 6.62        | ~75x    | 1.25x         |
| cuda-v1 (mine)       | 7.82        | ~64x    | 1.06x         |
| fi-baseline          | 8.27        | ~60x    | 1.00x         |
| fla-tma (mine)       | 12.58       | ~39x    | 0.66x         |
| pt-compiled          | 255.4       | ~1.9x   | 0.03x         |
| pt-reference         | 496.6       | ~1.0x   | 0.02x         |

Raw log: [nvbench-modal/all.txt](final-nums/nvbench-modal/all.txt)

While absolute times differ, rankings match CUPTI.

## Conclusion

For latency-bound kernels, less is more: remove unnecessary work, maximize parallelism, and don't add abstractions just to be cute.
The best Triton kernel (fla-recurrent) and the best CUDA kernel (cuda-v4) are within 2% of each other, both beating the FlashInfer CuTe-DSL kernel by ~1.3x.
The gap probably comes from avoiding shared memory staging, not from clever compute tricks.
