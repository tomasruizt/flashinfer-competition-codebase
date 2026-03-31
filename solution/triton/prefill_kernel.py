import math

import torch
import torch.nn.functional as F

from .fla_prefill_code import chunk_gated_delta_rule, fused_gate_cumsum


@torch.cuda.nvtx.range("kernel_prefill_fla_chunk")
@torch.no_grad()
def kernel_prefill_fla_chunk(
    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state
):
    """
    FLA chunkwise prefill wrapper for the competition interface (DPS).

    Adapts between competition tensors and FLA's chunk_gated_delta_rule:
    - Fused gate computation: g (log-space decay + cumsum) and beta (sigmoid) in one Triton kernel
    - Reshapes: [total_seq_len, H, D] -> [1, total_seq_len, H, D] (FLA expects B=1 for varlen)
    - Transposes state: [N, H, V, K] (k-last) <-> [N, H, K, V] (FLA)
    """
    # Fused gate computation + cumsum in a single Triton kernel
    g, beta = fused_gate_cumsum(a, b, dt_bias, A_log, cu_seqlens=cu_seqlens)

    # Add batch dim: [total_seq_len, ...] -> [1, total_seq_len, ...]
    # GVA: repeat-interleave q/k from 4 heads to 8 so FLA produces 8-head state
    num_v_heads = v.shape[1]
    num_q_heads = q.shape[1]
    q_4d = q.repeat_interleave(num_v_heads // num_q_heads, dim=1).unsqueeze(0)
    k_4d = k.repeat_interleave(num_v_heads // num_q_heads, dim=1).unsqueeze(0)
    v_4d = v.unsqueeze(0)
    # g and beta already [1, T, H] from fused_gate_cumsum

    # Transpose state: [N, H, V, K] -> [N, H, K, V]
    initial_state = state.transpose(-1, -2) if state is not None else None

    o, ht = chunk_gated_delta_rule(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        skip_cumsum=True,
    )

    # Remove batch dim and write to DPS outputs
    output.copy_(o.squeeze(0))
    # Transpose state back: [N, H, K, V] -> [N, H, V, K]
    new_state.copy_(ht.transpose(-1, -2))


@torch.no_grad()
def kernel_prefill_reference(
    q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, **_kwargs
):
    """
    Gated Delta Net prefill reference implementation (k-last layout).

    Processes variable-length batched sequences using cu_seqlens.
    Per-token recurrent update identical to decode, but loops over all tokens
    in each sequence.

    Args:
        q:          [total_seq_len, num_q_heads=4, K=128]     bf16
        k:          [total_seq_len, num_k_heads=4, K=128]     bf16
        v:          [total_seq_len, num_v_heads=8, V=128]     bf16
        state:      [num_seqs, num_v_heads=8, V=128, K=128]   f32 (k-last)
        A_log:      [num_v_heads=8]                            f32
        a:          [total_seq_len, num_v_heads=8]             bf16
        dt_bias:    [num_v_heads=8]                            f32
        b:          [total_seq_len, num_v_heads=8]             bf16
        cu_seqlens: [num_seqs+1]                               int32
        scale:      scalar f32

    Returns:
        output:     [total_seq_len, num_v_heads=8, V=128]     bf16
        new_state:  [num_seqs, num_v_heads=8, V=128, K=128]   f32
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [total_seq_len, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [total_seq_len, HV]
    beta = torch.sigmoid(b.float())  # [total_seq_len, HV]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = (
                state[seq_idx].clone().float().transpose(-1, -2)
            )  # [H,V,K] -> [H,K,V]
        else:
            state_HKV = torch.zeros(
                (num_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = k_H1K.float() @ old_state_HKV.float()
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum(
                "hkl,hlv->hkv", k_H1K.transpose(-1, -2), old_v_H1V
            )
            state_update = torch.einsum(
                "hkl,hlv->hkv", k_H1K.transpose(-1, -2), new_v_H1V
            )
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * (q_H1K.float() @ state_HKV.float())
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state
