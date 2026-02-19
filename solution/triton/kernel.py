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
    Wrapper adapting our competition interface to FLA's fused_recurrent_gated_delta_rule_fwd_kernel.
    Writes directly into DPS buffers (output, new_state).

    Interface differences:
        Ours                              FLA kernel
        ----                              ----------
        A_log, a, dt_bias → g (decay)     g: [B, T, HV] log-space decay (pre-computed)
        b → beta (update gate)            beta: [B, T, HV] already sigmoid'd
        state: [B, H, V, K] (k-last)     h0: [B, H, K, V] (k-first)
    """
    K = q.shape[-1]
    B, T, H, _ = k.shape
    V = v.shape[-1]
    HV = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute log-space decay: g = log(decay) = -exp(A_log) * softplus(a + dt_bias)
    # FLA kernel does: h *= exp(g), so g must be log of the decay factor
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())  # [B, 1, 8]

    # Compute beta = sigmoid(b)
    beta = torch.sigmoid(b.float())  # [B, 1, 8]

    # Transpose state from k-last [B, H, V, K] to k-first [B, H, K, V] for FLA
    if state is not None:
        initial_state = state.float().transpose(-1, -2).contiguous()  # [B, 8, K, V]
    else:
        initial_state = None

    N = B
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))  # gv is None
    NV = triton.cdiv(V, BV)

    # Kernel writes ht in k-first [B, HV, K, V] layout, but DPS new_state is
    # k-last [B, HV, V, K] — need a temp buffer for the transpose.
    ht_kfirst = q.new_empty(N, HV, K, V, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        beta=beta,
        o=output,
        h0=initial_state,
        ht=ht_kfirst,
        cu_seqlens=None,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=True,
        USE_QK_L2NORM_IN_KERNEL=False,
        PROFILE=bool(os.environ.get("PROTON_PROFILE")),
        num_warps=1,
        num_stages=3,
    )

    # ht_kfirst: [B, 8, K, V] → transpose to k-last [B, 8, V, K] into DPS buffer
    new_state.copy_(ht_kfirst.transpose(-1, -2))


# ---------------------------------------------------------------------------
# Triton kernel: fused recurrent GDN forward
# Inlined from fla.ops.gated_delta_rule.fused_recurrent (flash-linear-attention)
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_GV": lambda args: args["gv"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    PROFILE: tl.constexpr = False,
):
    # Grid: (NV=V/BV, B*HV) — each program handles a [BK, BV] tile of one head's state
    # For decode: BK=128 (full K), BV=8, so NV=16 tiles cover the 128-wide V dimension
    if PROFILE:
        pl.enter_scope("gdn_recurrent")
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV  # batch index, v-head index
    i_h = i_hv // (HV // H)  # q/k-head index (GVA: 2 v-heads per qk-head)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)  # [BK] — offsets into K dimension
    o_v = i_v * BV + tl.arange(0, BV)  # [BV] — offsets into V dimension (tile)

    # Pointers to the start of this (batch, head, token=bos) for each input
    p_q = q + (bos * H + i_h) * K + o_k  # q: [B*T, H, K] flattened
    p_k = k + (bos * H + i_h) * K + o_k  # k: [B*T, H, K] flattened
    p_v = v + (bos * HV + i_hv) * V + o_v  # v: [B*T, HV, V] flattened
    if USE_G:
        p_g = g + bos * HV + i_hv  # g: [B*T, HV] — scalar per head per token
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv  # beta: [B*T, HV] — scalar per head per token
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v  # o: [B*T, HV, V] flattened

    mask_k = o_k < K  # [BK] — True where BK <= K
    mask_v = o_v < V  # [BV] — True where tile is in bounds
    mask_h = mask_k[:, None] & mask_v[None, :]  # [BK, BV]

    b_h = tl.zeros(
        [BK, BV], dtype=tl.float32
    )  # state tile: [BK=128, BV=8] slice of [K, V]
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)  # [BK, BV] from GMEM

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # [BK] — query
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # [BK] — key
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # [BV] — value (tile)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)  # [BK] — L2 normalize
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)  # [BK] — L2 normalize
        b_q = b_q * scale  # [BK] — scaled query
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)  # scalar — update gate
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)  # [BV]

        # Decay state
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)  # scalar — log decay
            b_h *= tl.exp(b_g)  # [BK, BV] *= scalar

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)  # [BK]
            b_h *= tl.exp(b_gk[:, None])  # [BK, BV] *= [BK, 1]

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)  # [BV]
            b_h *= tl.exp(b_gv[None, :])  # [BK, BV] *= [1, BV]

        # Delta rule: retrieve old value, blend with new, update state
        # k@S:   sum([BK, BV] * [BK, 1], dim=0) -> [BV]  (matvec: value stored at key k)
        # blend: scalar * ([BV] - [BV]) -> [BV]           (beta-weighted delta)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        # outer product: [BK, 1] * [BV] -> [BK, BV]      (state update)
        b_h += b_k[:, None] * b_v

        # q@S: sum([BK, BV] * [BK, 1], dim=0) -> [BV]    (matvec: read output from state)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)  # [BV] -> GMEM

        # Advance pointers to next token
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)  # [BK, BV] -> GMEM
    if PROFILE:
        pl.exit_scope("gdn_recurrent")
