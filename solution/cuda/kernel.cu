/*
 * CUDA kernel for GDN (Gated Delta Net) decode.
 *
 * Port of the FLA Triton kernel (fused_recurrent_gated_delta_rule_fwd_kernel).
 * Grid: (V/BV, B*HV) = (16, 8) with 1 warp (32 threads) per block.
 * Each block handles a [BV=8, BK=128] tile of one head's state matrix.
 * Each thread owns KVEC=4 consecutive K-elements.
 *
 * Uses TVM FFI for framework integration (compiled by TVMFFIBuilder).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

static constexpr int BV = 8;
static constexpr int BK = 128;
static constexpr int WARP_SIZE = 32;
static constexpr int KVEC = BK / WARP_SIZE;  // 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q,        // [B, 1, H_qk, K]
    const __nv_bfloat16* __restrict__ k,        // [B, 1, H_qk, K]
    const __nv_bfloat16* __restrict__ v,        // [B, 1, HV, V]
    const float* __restrict__ h0,               // [B, HV, V, K] state (k-last)
    const float* __restrict__ A_log,            // [HV]
    const __nv_bfloat16* __restrict__ a_gate,   // [B, 1, HV]
    const float* __restrict__ dt_bias,          // [HV]
    const __nv_bfloat16* __restrict__ b_gate,   // [B, 1, HV]
    float scale,
    __nv_bfloat16* __restrict__ output,         // [B, 1, HV, V]
    float* __restrict__ ht,                     // [B, HV, V, K] new state (k-last)
    int H_qk,
    int HV,
    int K_dim,
    int V_dim
) {
    // Grid mapping (same as Triton kernel)
    const int i_v = blockIdx.x;               // V-tile index [0..V/BV-1]
    const int i_nh = blockIdx.y;              // batch*HV flattened
    const int i_hv = i_nh % HV;              // v-head index
    const int i_h = i_hv / (HV / H_qk);     // q/k-head index (GVA: 2 v-heads per qk-head)
    const int i_n = i_nh / HV;               // batch index

    const int tid = threadIdx.x;              // [0..31]
    const int k_base = tid * KVEC;            // K-offset for this thread

    // --- Load state tile [BV, BK] f32 via float4 (coalesced 128-bit) ---
    const int state_head_offset = i_nh * V_dim * K_dim;
    float h[BV][KVEC];
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = i_v * BV + bv;
        float4 val = *reinterpret_cast<const float4*>(
            h0 + state_head_offset + v_idx * K_dim + k_base);
        h[bv][0] = val.x;
        h[bv][1] = val.y;
        h[bv][2] = val.z;
        h[bv][3] = val.w;
    }

    // --- Load q, k [KVEC per thread] bf16 -> f32, q pre-scaled ---
    float q_reg[KVEC], k_reg[KVEC];
    {
        const __nv_bfloat16* q_ptr = q + (i_n * H_qk + i_h) * K_dim + k_base;
        const __nv_bfloat16* k_ptr = k + (i_n * H_qk + i_h) * K_dim + k_base;
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            q_reg[i] = __bfloat162float(q_ptr[i]) * scale;
            k_reg[i] = __bfloat162float(k_ptr[i]);
        }
    }

    // --- Load v [BV] bf16 -> f32 (broadcast: all threads load same 8 values) ---
    float v_reg[BV];
    {
        const __nv_bfloat16* v_ptr = v + (i_n * HV + i_hv) * V_dim + i_v * BV;
        #pragma unroll
        for (int bv = 0; bv < BV; bv++) {
            v_reg[bv] = __bfloat162float(v_ptr[bv]);
        }
    }

    // --- Load scalars and compute gates ---
    const float b_A  = A_log[i_hv];
    const float b_a  = __bfloat162float(a_gate[i_n * HV + i_hv]);
    const float b_dt = dt_bias[i_hv];
    const float b_b  = __bfloat162float(b_gate[i_n * HV + i_hv]);

    const float x  = b_a + b_dt;
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));  // softplus
    const float g  = expf(-expf(b_A) * sp);                     // decay gate
    const float beta = 1.0f / (1.0f + expf(-b_b));              // sigmoid

    // --- Decay state: h *= g ---
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            h[bv][i] *= g;
        }
    }

    // --- Delta rule: matvec k@S, blend, outer product (fused per v-row) ---
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        // Partial dot product k . h[bv] (4 elements per thread)
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            partial += k_reg[i] * h[bv][i];
        }
        // Warp all-reduce: every thread gets old_v[bv]
        float old_v_bv = warp_reduce_sum(partial);

        // Blend: delta_v = beta * (v - old_v)
        float dv = beta * (v_reg[bv] - old_v_bv);

        // Outer product: h[bv][k] += dv * k[k]
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            h[bv][i] += dv * k_reg[i];
        }
    }

    // --- Output matvec q@S -> out[BV] (warp reduce, thread 0 stores) ---
    {
        __nv_bfloat16* o_ptr = output + (i_n * HV + i_hv) * V_dim + i_v * BV;
        #pragma unroll
        for (int bv = 0; bv < BV; bv++) {
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < KVEC; i++) {
                partial += q_reg[i] * h[bv][i];
            }
            float out_bv = warp_reduce_sum(partial);
            if (tid == 0) {
                o_ptr[bv] = __float2bfloat16(out_bv);
            }
        }
    }

    // --- Store state tile [BV, BK] f32 via float4 (coalesced 128-bit) ---
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = i_v * BV + bv;
        float4 val;
        val.x = h[bv][0];
        val.y = h[bv][1];
        val.z = h[bv][2];
        val.w = h[bv][3];
        *reinterpret_cast<float4*>(
            ht + state_head_offset + v_idx * K_dim + k_base) = val;
    }
}

// ---------------------------------------------------------------------------
// TVM FFI host wrapper
// ---------------------------------------------------------------------------

void KernelCuda(
    tvm::ffi::TensorView q,         // [B, 1, H_qk, K] bf16
    tvm::ffi::TensorView k,         // [B, 1, H_qk, K] bf16
    tvm::ffi::TensorView v,         // [B, 1, HV, V] bf16
    tvm::ffi::TensorView state,     // [B, HV, V, K] f32
    tvm::ffi::TensorView A_log,     // [HV] f32
    tvm::ffi::TensorView a,         // [B, 1, HV] bf16
    tvm::ffi::TensorView dt_bias,   // [HV] f32
    tvm::ffi::TensorView b_gate,    // [B, 1, HV] bf16
    double scale,
    tvm::ffi::TensorView output,    // [B, 1, HV, V] bf16
    tvm::ffi::TensorView new_state  // [B, HV, V, K] f32
) {
    const int B = q.size(0);
    const int H_qk = q.size(2);
    const int HV = v.size(2);
    const int V_dim = v.size(3);
    const int K_dim = q.size(3);

    dim3 grid(V_dim / BV, B * HV);
    dim3 block(WARP_SIZE);

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    gdn_decode_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(state.data_ptr()),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),
        static_cast<float>(scale),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float*>(new_state.data_ptr()),
        H_qk, HV, K_dim, V_dim
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda, KernelCuda);
