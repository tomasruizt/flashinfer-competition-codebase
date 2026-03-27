import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def kernel_prefill_reference(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
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
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)  # [H,V,K] -> [H,K,V]
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
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * (q_H1K.float() @ state_HKV.float())
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)  # [H,K,V] -> [H,V,K]

    return output, new_state
