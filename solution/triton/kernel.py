"""
Gated Delta Net decode kernel – PyTorch reference baseline.

Entry point: kernel()  (destination-passing style)
Definition:  gdn_decode_qk4_v8_d128_k_last
"""

import math
import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _reference_decode(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).

    Single-token recurrent step: reads from a fixed-size state matrix using q,
    then updates the state by erasing old information at key k and writing new
    value v. GVA config: q/k heads are repeat-interleaved to match v heads.

    Args:
        q:       [B, 1, num_q_heads=4, K=128]  bf16 — query, used to read from state.
        k:       [B, 1, num_k_heads=4, K=128]  bf16 — key, the "address" to erase/write in state.
        v:       [B, 1, num_v_heads=8, V=128]  bf16 — value, new information to write into state.
        state:   [B, num_v_heads=8, V=128, K=128] f32 — recurrent state (k-last layout [H,V,K]).
                 Optional; zeros if None.
        A_log:   [num_v_heads=8]                f32 — learnable log-space base decay rate per head.
                 Controls how fast each head forgets: g = exp(-exp(A_log) * softplus(a + dt_bias)).
        a:       [B, 1, num_v_heads=8]          bf16 — input-dependent decay (per token, per head).
        dt_bias: [num_v_heads=8]                f32 — learnable decay bias, added to a before softplus.
        b:       [B, 1, num_v_heads=8]          bf16 — update gate input. beta = sigmoid(b) controls
                 how much of the new value to write vs retaining old value at key k.
        scale:   scalar f32 — output scale factor, default 1/sqrt(K) = 1/sqrt(128).

    Returns:
        output:    [B, 1, num_v_heads=8, V=128] bf16 — attention output (scale * q @ state_new).
        new_state: [B, num_v_heads=8, V=128, K=128] f32 — updated recurrent state (k-last layout).

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

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

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

            output[b_idx, h_idx] = scale * (q_h @ h_state)  # [K] @ [K, V] -> [V] — read from state
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K] back to storage layout

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state


@torch.no_grad()
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """DPS entry point: writes results into pre-allocated output and new_state.

    Args:
        q..scale: see _reference_decode.
        output:    [B, 1, num_v_heads=8, V=128] bf16 — pre-allocated output tensor.
        new_state: [B, num_v_heads=8, V=128, K=128] f32 — pre-allocated state tensor.
    """
    out, ns = _reference_decode(q, k, v, state, A_log, a, dt_bias, b, scale)
    output.copy_(out)
    new_state.copy_(ns)
