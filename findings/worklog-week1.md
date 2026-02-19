# Worklog 2026-02-19

Proton intra-kernel profiling and kernel cleanup.

- Set up Proton profiling for the FLA Triton kernel (`scripts/profile_proton.py`).
  - `make proton-fla` generates both timeline trace and op-measurement hatchet file.
  - Key gotcha: `torch.compile` serializes Triton kernel source into a new scope,
    losing any `pl` (profiler language) imports. Fix: disable dynamo for profiling,
    use `PROTON_PROFILE` env var + `tl.constexpr` guard so scopes are no-ops during benchmarks.
- Added Proton scope annotations to the kernel: `load_initial_state`, `load_qkv`,
  `state_update`, `store_output`, `store_final_state` — gated behind `if PROFILE:`.
- Stripped dead code from the inlined FLA kernel for our decode-only use case.
- Warp-level timeline trace (Perfetto) shows the per-CTA breakdown clearly:

  ![GDN decode warp-level timeline](../images/gdn-profiler-warp-level-runtimes.png)

  - `load_initial_state` dominates (~40% of kernel time) — loading the 128x128 f32 state tile from GMEM.
  - `load_qkv` is the second largest — loading q, k, v vectors.
  - `state_update` (delta rule matvecs + outer product) and `store_final_state` are roughly equal.
  - `store_output` is negligible (just a [BV=8] vector).
  - Confirms the kernel is **memory-bound**: loads/stores >> compute.

# Worklog 2026-02-18

Project setup and first baselines for the GDN decode kernel.

- Set up project structure, documented GDN math and kernel interface in `CLAUDE.md`.
- **torch.compile'd PyTorch reference:** ~0.73 ms, ~1.8x speedup (RTX 3090).
- **FLA fused recurrent Triton kernel:** ~0.14 ms, ~9.5x speedup (RTX 3090).
  Called the Triton kernel directly (not the FLA wrapper) so torch.compile can
  fuse gate computation and state transposes.
- Added multi-algo dispatch (`--algo` flag / `ALGO` env var) across all scripts.
- Modal pipeline: hit `COMPILE_ERROR` on B200 because `flash-linear-attention`
  wasn't in the Modal image (not a dep of `flashinfer-bench`). Fixed.
- FLA fused recurrent achieves between 11.37 and 13.22x speedups on modal.
- I wanted compare the performance on the leaderboard, but the submission infrastructure is not yet operational.
- worked time: 4h13
