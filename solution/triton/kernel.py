import math
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.profiler.language as pl


@torch.no_grad()
@torch.compile(fullgraph=True)
def kernel_pt_reference(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    Gated Delta Net decode reference implementation (k-last layout).
    Writes directly into DPS buffers (output, new_state).

    Single-token recurrent step: reads from a fixed-size state matrix using q,
    then updates the state by erasing old information at key k and writing new
    value v. GVA config: q/k heads are repeat-interleaved to match v heads.

    Args:
        q:         [B, 1, num_q_heads=4, K=128]     bf16 — query, used to read from state.
        k:         [B, 1, num_k_heads=4, K=128]     bf16 — key, the "address" to erase/write in state.
        v:         [B, 1, num_v_heads=8, V=128]     bf16 — value, new information to write into state.
        state:     [B, num_v_heads=8, V=128, K=128]  f32 — recurrent state (k-last layout [H,V,K]).
                   Optional; zeros if None.
        A_log:     [num_v_heads=8]                    f32 — learnable log-space base decay rate per head.
                   Controls how fast each head forgets: g = exp(-exp(A_log) * softplus(a + dt_bias)).
        a:         [B, 1, num_v_heads=8]             bf16 — input-dependent decay (per token, per head).
        dt_bias:   [num_v_heads=8]                    f32 — learnable decay bias, added to a before softplus.
        b:         [B, 1, num_v_heads=8]             bf16 — update gate input. beta = sigmoid(b) controls
                   how much of the new value to write vs retaining old value at key k.
        scale:     scalar f32 — output scale factor, default 1/sqrt(K) = 1/sqrt(128).
        output:    [B, 1, num_v_heads=8, V=128]     bf16 — DPS output buffer.
        new_state: [B, num_v_heads=8, V=128, K=128]  f32 — DPS state output buffer.

    Recurrence (per head, working in [K,V] layout):
        g     = exp(-exp(A_log) * softplus(a + dt_bias))   # global decay ∈ (0,1)
        beta  = sigmoid(b)                                  # update gate ∈ (0,1)
        S_old = g * S                                       # decay entire state
        S_new = S_old - k^T @ (k @ S_old) + k^T @ (beta * v + (1-beta) * (k @ S_old))
        out   = scale * q @ S_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    # fmt: off
    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]       # [K=128]        — query vector
            k_h = k_exp[b_idx, h_idx]       # [K=128]        — key vector (L2-normalized)
            v_h = v_f32[b_idx, h_idx]       # [V=128]        — value vector
            h_state = (
                state_f32[b_idx, h_idx].clone().transpose(-1, -2)
            )                                # [K=128, V=128] — state transposed from [V,K] storage
            g_val = g_f32[b_idx, h_idx]     # scalar          — global decay gate ∈ (0,1)
            beta_val = beta_f32[b_idx, h_idx]  # scalar       — update gate ∈ (0,1)

            old_state = g_val * h_state     # [K, V]          — decayed state
            old_v = k_h @ old_state         # [K] @ [K, V] -> [V] — matvec: value currently stored at key k
            new_v = beta_val * v_h + (1 - beta_val) * old_v  # [V] — blended new/old value
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)  # [K,1] @ [1,V] -> [K, V] — outer product: erase old
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)  # [K,1] @ [1,V] -> [K, V] — outer product: write new
            h_state = old_state - state_remove + state_update     # [K, V] — updated state

            output[b_idx, 0, h_idx] = (scale * (q_h @ h_state)).to(torch.bfloat16)  # [V] into [B,1,H,V]
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K] back to storage layout
    # fmt: on


@torch.no_grad()
@torch.compile(fullgraph=True)
def kernel_fla_recurrent(
    q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state
):
    """
    Wrapper adapting our competition interface to the fused Triton kernel.
    Gate computation (decay, sigmoid) is fused into the kernel.
    State transpose k-last→k-first still done here (kernel uses k-first layout).
    """
    K = q.shape[-1]
    B, T, H, _ = k.shape
    assert T == 1
    V = v.shape[-1]
    HV = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Transpose state from k-last [B, H, V, K] to k-first [B, H, K, V]
    initial_state = state.float().transpose(-1, -2).contiguous()  # [B, 8, K, V]

    N = B
    BK = 128
    BV = 32  # autotune winner on RTX 3090 (was 8 in original FLA)
    NV = triton.cdiv(V, BV)

    # Kernel writes ht in k-first [B, HV, K, V] layout, but DPS new_state is
    # k-last [B, HV, V, K] — need a temp buffer for the transpose.
    ht_kfirst = q.new_empty(N, HV, K, V, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a_gate=a,
        dt_bias=dt_bias,
        b_gate=b,
        o=output,
        h0=initial_state,
        USE_INITIAL_STATE=True,
        ht=ht_kfirst,
        scale=scale,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        PROFILE=bool(os.environ.get("PROTON_PROFILE")),
        num_warps=2,
        num_stages=3,
    )

    # ht_kfirst: [B, 8, K, V] → transpose to k-last [B, 8, V, K] into DPS buffer
    new_state.copy_(ht_kfirst.transpose(-1, -2))


# ---------------------------------------------------------------------------
# Triton kernel: fused recurrent GDN forward
# Inlined from fla.ops.gated_delta_rule.fused_recurrent (flash-linear-attention)
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ---------------------------------------------------------------------------


@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    A_log,
    a_gate,
    dt_bias,
    b_gate,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    PROFILE: tl.constexpr = False,
):
    # Grid: (NV=V/BV, B*HV) — each program handles a [BK, BV] tile of one head's state
    # For decode: BK=128 (full K), BV=8, so NV=16 tiles cover the 128-wide V dimension
    if PROFILE:
        pl.enter_scope("gdn_recurrent")
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV  # batch index, v-head index
    i_h = i_hv // (HV // H)  # q/k-head index (GVA: 2 v-heads per qk-head)

    o_k = tl.arange(0, BK)  # [BK] — offsets into K dimension
    o_v = i_v * BV + tl.arange(0, BV)  # [BV] — offsets into V dimension (tile)

    # Pointers into this (batch=i_n, head) for the single token (T=1)
    p_q = q + (i_n * H + i_h) * K + o_k  # q: [B, H, K] flattened
    p_k = k + (i_n * H + i_h) * K + o_k  # k: [B, H, K] flattened
    p_v = v + (i_n * HV + i_hv) * V + o_v  # v: [B, HV, V] flattened
    p_o = o + (i_n * HV + i_hv) * V + o_v  # o: [B, HV, V] flattened

    mask_k = o_k < K  # [BK]
    mask_v = o_v < V  # [BV]
    mask_h = mask_k[:, None] & mask_v[None, :]  # [BK, BV]

    # Load state tile
    if PROFILE:
        pl.enter_scope("load_initial_state")
    b_h = tl.zeros([BK, BV], dtype=tl.float32)  # [BK=128, BV=8] slice of [K, V]
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # [BK, BV] from GMEM
    if PROFILE:
        pl.exit_scope("load_initial_state")

    # Load inputs (single token)
    if PROFILE:
        pl.enter_scope("load_qkv")
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # [BK] — query
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # [BK] — key
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # [BV] — value
    if PROFILE:
        pl.exit_scope("load_qkv")

    b_q = b_q * scale  # [BK] — scaled query

    # Compute gates from raw inputs (fused — avoids separate PyTorch kernels)
    b_A = tl.load(A_log + i_hv).to(tl.float32)        # scalar f32 — log base decay
    b_a = tl.load(a_gate + i_n * HV + i_hv).to(tl.float32)  # scalar bf16→f32
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)     # scalar f32 — decay bias
    b_b = tl.load(b_gate + i_n * HV + i_hv).to(tl.float32)  # scalar bf16→f32

    x = b_a + b_dt
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # softplus
    b_h *= tl.exp(-tl.exp(b_A) * sp)  # decay state: g = exp(-exp(A_log) * softplus(a + dt_bias))
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # sigmoid(b) — update gate

    if PROFILE:
        pl.enter_scope("state_update")
    # Delta rule: retrieve old value, blend with new, update state
    # k@S:   sum([BK, BV] * [BK, 1], dim=0) -> [BV]  (matvec: value stored at key k)
    # blend: scalar * ([BV] - [BV]) -> [BV]           (beta-weighted delta)
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
    # outer product: [BK, 1] * [BV] -> [BK, BV]      (state update)
    b_h += b_k[:, None] * b_v
    if PROFILE:
        pl.exit_scope("state_update")

    # q@S: sum([BK, BV] * [BK, 1], dim=0) -> [BV]    (matvec: read output from state)
    b_o = tl.sum(b_h * b_q[:, None], 0)
    if PROFILE:
        pl.enter_scope("store_output")
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)  # [BV] -> GMEM
    if PROFILE:
        pl.exit_scope("store_output")

    # Store final state
    p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    if PROFILE:
        pl.enter_scope("store_final_state")
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)  # [BK, BV] -> GMEM
    if PROFILE:
        pl.exit_scope("store_final_state")
        pl.exit_scope("gdn_recurrent")
