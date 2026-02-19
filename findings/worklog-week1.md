# Worklog 2026-02-19

Proton intra-kernel profiling and kernel cleanup.

- Set up Proton profiling for the FLA Triton kernel (`scripts/profile_proton.py`).
- Stripped dead code from the inlined FLA kernel for our decode-only use case.
- Warp-level timeline trace (Perfetto) shows the per-CTA breakdown:
  ![GDN decode warp-level timeline](../images/gdn-profiler-warp-level-runtimes.png)
- Refactoring the FLA kernel yielded massive speedups. In particular moving the decay computation inside the kernel, and remove the state transposition + contiguous() calls.
- worked time: 5h25

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
