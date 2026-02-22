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

template <int BV>
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
            partial = fmaf(k_reg[i], h[bv][i], partial);
        }
        // Warp all-reduce: every thread gets old_v[bv]
        float old_v_bv = warp_reduce_sum(partial);

        // Blend: delta_v = beta * (v - old_v)
        float dv = beta * (v_reg[bv] - old_v_bv);

        // Outer product: h[bv][k] += dv * k[k]
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            h[bv][i] = fmaf(dv, k_reg[i], h[bv][i]);
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
                partial = fmaf(q_reg[i], h[bv][i], partial);
            }
            float out_bv = warp_reduce_sum(partial);
            if (tid == 0) {
                o_ptr[bv] = __float2bfloat16(out_bv);
            }
        }
    }

    // --- Store state tile [BV, BK] f32 via float4, streaming (bypass L2) ---
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = i_v * BV + bv;
        float4 val;
        val.x = h[bv][0];
        val.y = h[bv][1];
        val.z = h[bv][2];
        val.w = h[bv][3];
        __stcs(reinterpret_cast<float4*>(
            ht + state_head_offset + v_idx * K_dim + k_base), val);
    }
}

// ---------------------------------------------------------------------------
// TVM FFI host wrappers: v1 (BV=8), v1b (BV=16), v1c (BV=4), v1d (BV=2)
// ---------------------------------------------------------------------------

#define DEFINE_V1_HOST(FuncName, Bv)                                           \
void FuncName(                                                                 \
    tvm::ffi::TensorView q,                                                    \
    tvm::ffi::TensorView k,                                                    \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView state,                                                \
    tvm::ffi::TensorView A_log,                                                \
    tvm::ffi::TensorView a,                                                    \
    tvm::ffi::TensorView dt_bias,                                              \
    tvm::ffi::TensorView b_gate,                                               \
    double scale,                                                              \
    tvm::ffi::TensorView output,                                               \
    tvm::ffi::TensorView new_state                                             \
) {                                                                            \
    const int B = q.size(0);                                                   \
    const int H_qk = q.size(2);                                               \
    const int HV = v.size(2);                                                  \
    const int V_dim = v.size(3);                                               \
    const int K_dim = q.size(3);                                               \
    dim3 grid(V_dim / (Bv), B * HV);                                          \
    dim3 block(WARP_SIZE);                                                     \
    DLDevice dev = q.device();                                                 \
    cudaStream_t stream = static_cast<cudaStream_t>(                           \
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));                    \
    gdn_decode_kernel<Bv><<<grid, block, 0, stream>>>(                         \
        static_cast<const __nv_bfloat16*>(q.data_ptr()),                       \
        static_cast<const __nv_bfloat16*>(k.data_ptr()),                       \
        static_cast<const __nv_bfloat16*>(v.data_ptr()),                       \
        static_cast<const float*>(state.data_ptr()),                           \
        static_cast<const float*>(A_log.data_ptr()),                           \
        static_cast<const __nv_bfloat16*>(a.data_ptr()),                       \
        static_cast<const float*>(dt_bias.data_ptr()),                         \
        static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),                  \
        static_cast<float>(scale),                                             \
        static_cast<__nv_bfloat16*>(output.data_ptr()),                        \
        static_cast<float*>(new_state.data_ptr()),                             \
        H_qk, HV, K_dim, V_dim                                                \
    );                                                                         \
}

DEFINE_V1_HOST(KernelCuda,    8)   // 128 blocks (v1)
DEFINE_V1_HOST(KernelCudaV1b, 16)  //  64 blocks (v1b)
DEFINE_V1_HOST(KernelCudaV1c,  4)  // 256 blocks (v1c)
DEFINE_V1_HOST(KernelCudaV1d,  2)  // 512 blocks (v1d)

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda, KernelCuda);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v1b, KernelCudaV1b);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v1c, KernelCudaV1c);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v1d, KernelCudaV1d);

// ===========================================================================
// V2 kernel: 4 warps per block, shared memory k/q, templated on BV_PER_WARP
// ===========================================================================

static constexpr int V2_NUM_WARPS = 4;
static constexpr int V2_BLOCK_SIZE = V2_NUM_WARPS * WARP_SIZE;  // 128

template <int BV_PER_WARP>
__global__ __launch_bounds__(128)
void gdn_decode_v2_kernel(
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
    // Grid: (V_dim / (BV_PER_WARP * NUM_WARPS), B * HV)
    const int i_v = blockIdx.x;                // V-tile index
    const int i_nh = blockIdx.y;               // batch*HV flattened
    const int i_hv = i_nh % HV;               // v-head index
    const int i_h = i_hv / (HV / H_qk);      // q/k-head index (GVA)
    const int i_n = i_nh / HV;                // batch index

    const int tid = threadIdx.x;               // [0..127]
    const int warp_id = tid / WARP_SIZE;       // [0..3]
    const int lane = tid % WARP_SIZE;          // [0..31]
    const int k_base = lane * KVEC;            // K-offset for this thread

    // V-row range for this warp within this block's tile
    constexpr int BV_PER_BLOCK = BV_PER_WARP * V2_NUM_WARPS;
    const int block_v_base = i_v * BV_PER_BLOCK;
    const int warp_v_base = block_v_base + warp_id * BV_PER_WARP;

    // Shared memory for k/q vectors (reused)
    __shared__ float kq_shared[128];

    // --- Load state tile [BV_PER_WARP, 128] f32 via float4 (coalesced) per warp ---
    const int state_head_offset = i_nh * V_dim * K_dim;
    float h[BV_PER_WARP][KVEC];
    #pragma unroll
    for (int bv = 0; bv < BV_PER_WARP; bv++) {
        const int v_idx = warp_v_base + bv;
        float4 val = *reinterpret_cast<const float4*>(
            h0 + state_head_offset + v_idx * K_dim + k_base);
        h[bv][0] = val.x;
        h[bv][1] = val.y;
        h[bv][2] = val.z;
        h[bv][3] = val.w;
    }

    // --- Load k into shared memory (128 threads load 1 element each) ---
    {
        const __nv_bfloat16* k_ptr = k + (i_n * H_qk + i_h) * K_dim;
        if (tid < K_dim) {
            kq_shared[tid] = __bfloat162float(k_ptr[tid]);
        }
    }
    __syncthreads();

    // Each thread reads its 4 k-elements from shared memory
    float k_reg[KVEC];
    #pragma unroll
    for (int i = 0; i < KVEC; i++) {
        k_reg[i] = kq_shared[k_base + i];
    }

    // --- Load v for this warp's V-rows ---
    float v_reg[BV_PER_WARP];
    {
        const __nv_bfloat16* v_ptr = v + (i_n * HV + i_hv) * V_dim;
        #pragma unroll
        for (int bv = 0; bv < BV_PER_WARP; bv++) {
            v_reg[bv] = __bfloat162float(v_ptr[warp_v_base + bv]);
        }
    }

    // --- Compute gates (all threads compute same values) ---
    const float b_A  = A_log[i_hv];
    const float b_a  = __bfloat162float(a_gate[i_n * HV + i_hv]);
    const float b_dt = dt_bias[i_hv];
    const float b_b  = __bfloat162float(b_gate[i_n * HV + i_hv]);

    const float x  = b_a + b_dt;
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    const float g  = expf(-expf(b_A) * sp);
    const float beta = 1.0f / (1.0f + expf(-b_b));

    // --- Decay state: h *= g ---
    #pragma unroll
    for (int bv = 0; bv < BV_PER_WARP; bv++) {
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            h[bv][i] *= g;
        }
    }

    // --- Delta rule: matvec k@S, blend, outer product (fused per v-row) ---
    #pragma unroll
    for (int bv = 0; bv < BV_PER_WARP; bv++) {
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            partial += k_reg[i] * h[bv][i];
        }
        float old_v_bv = warp_reduce_sum(partial);

        float dv = beta * (v_reg[bv] - old_v_bv);

        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            h[bv][i] += dv * k_reg[i];
        }
    }

    // --- Load q into shared memory (reuse kq_shared buffer) ---
    __syncthreads();
    {
        const __nv_bfloat16* q_ptr = q + (i_n * H_qk + i_h) * K_dim;
        if (tid < K_dim) {
            kq_shared[tid] = __bfloat162float(q_ptr[tid]) * scale;
        }
    }
    __syncthreads();

    // Each thread reads its 4 q-elements from shared memory
    float q_reg[KVEC];
    #pragma unroll
    for (int i = 0; i < KVEC; i++) {
        q_reg[i] = kq_shared[k_base + i];
    }

    // --- Output matvec q@S -> out[BV_PER_WARP] (warp reduce, lane 0 stores) ---
    {
        __nv_bfloat16* o_ptr = output + (i_n * HV + i_hv) * V_dim + warp_v_base;
        #pragma unroll
        for (int bv = 0; bv < BV_PER_WARP; bv++) {
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < KVEC; i++) {
                partial += q_reg[i] * h[bv][i];
            }
            float out_bv = warp_reduce_sum(partial);
            if (lane == 0) {
                o_ptr[bv] = __float2bfloat16(out_bv);
            }
        }
    }

    // --- Store state tile [BV_PER_WARP, 128] f32 via float4 (coalesced) ---
    #pragma unroll
    for (int bv = 0; bv < BV_PER_WARP; bv++) {
        const int v_idx = warp_v_base + bv;
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
// TVM FFI host wrappers: v2 (BV=32), v2b (BV=16), v2c (BV=8)
// ---------------------------------------------------------------------------

// Helper macro to avoid repeating the host wrapper boilerplate
#define DEFINE_V2_HOST(FuncName, BvPerWarp)                                    \
void FuncName(                                                                 \
    tvm::ffi::TensorView q,                                                    \
    tvm::ffi::TensorView k,                                                    \
    tvm::ffi::TensorView v,                                                    \
    tvm::ffi::TensorView state,                                                \
    tvm::ffi::TensorView A_log,                                                \
    tvm::ffi::TensorView a,                                                    \
    tvm::ffi::TensorView dt_bias,                                              \
    tvm::ffi::TensorView b_gate,                                               \
    double scale,                                                              \
    tvm::ffi::TensorView output,                                               \
    tvm::ffi::TensorView new_state                                             \
) {                                                                            \
    const int B = q.size(0);                                                   \
    const int H_qk = q.size(2);                                               \
    const int HV = v.size(2);                                                  \
    const int V_dim = v.size(3);                                               \
    const int K_dim = q.size(3);                                               \
    constexpr int bv_per_block = (BvPerWarp) * V2_NUM_WARPS;                   \
    dim3 grid(V_dim / bv_per_block, B * HV);                                   \
    dim3 block(V2_BLOCK_SIZE);                                                 \
    DLDevice dev = q.device();                                                 \
    cudaStream_t stream = static_cast<cudaStream_t>(                           \
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));                    \
    gdn_decode_v2_kernel<BvPerWarp><<<grid, block, 0, stream>>>(               \
        static_cast<const __nv_bfloat16*>(q.data_ptr()),                       \
        static_cast<const __nv_bfloat16*>(k.data_ptr()),                       \
        static_cast<const __nv_bfloat16*>(v.data_ptr()),                       \
        static_cast<const float*>(state.data_ptr()),                           \
        static_cast<const float*>(A_log.data_ptr()),                           \
        static_cast<const __nv_bfloat16*>(a.data_ptr()),                       \
        static_cast<const float*>(dt_bias.data_ptr()),                         \
        static_cast<const __nv_bfloat16*>(b_gate.data_ptr()),                  \
        static_cast<float>(scale),                                             \
        static_cast<__nv_bfloat16*>(output.data_ptr()),                        \
        static_cast<float*>(new_state.data_ptr()),                             \
        H_qk, HV, K_dim, V_dim                                                \
    );                                                                         \
}

DEFINE_V2_HOST(KernelCudaV2,  32)  // 1 block/head, 8 blocks total
DEFINE_V2_HOST(KernelCudaV2b, 16)  // 2 blocks/head, 16 blocks total
DEFINE_V2_HOST(KernelCudaV2c,  8)  // 4 blocks/head, 32 blocks total

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v2,  KernelCudaV2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v2b, KernelCudaV2b);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v2c, KernelCudaV2c);

// ===========================================================================
// V3 kernel: 1 thread per V-row, state in shared memory, no warp reductions
// ===========================================================================
//
// Grid: (V_dim / 32, B * HV) = (4, 8) = 32 blocks
// Block: 32 threads (1 warp), thread t owns V-row (block_v_base + t)
// Shared memory: state[32][129] padded + kq[128] = ~17 KB
//
// Bank conflict analysis (129-stride padding):
//   Thread t accessing state_smem[t * 129 + j]: bank = (t + j) % 32
//   For fixed j across 32 threads: 32 distinct banks. Zero conflicts.

static constexpr int V3_BLOCK_SIZE = 32;  // 1 warp
static constexpr int V3_STATE_STRIDE = 129;  // 128 + 1 padding
// Layout: state[32][129] + k[128] + q[128]
static constexpr int V3_SMEM_BYTES =
    (V3_BLOCK_SIZE * V3_STATE_STRIDE + BK + BK) * sizeof(float);  // ~17.5 KB


__global__ void gdn_decode_v3_kernel(
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
    const int i_v = blockIdx.x;               // V-tile index [0..3]
    const int i_nh = blockIdx.y;              // batch*HV flattened
    const int i_hv = i_nh % HV;
    const int i_h = i_hv / (HV / H_qk);
    const int i_n = i_nh / HV;
    const int tid = threadIdx.x;              // [0..31]

    const int block_v_base = i_v * V3_BLOCK_SIZE;

    // Dynamic shared memory layout: state[32][129], k[128], q[128]
    extern __shared__ float smem[];
    float* state_smem = smem;                                     // [32 * 129]
    float* k_smem = smem + V3_BLOCK_SIZE * V3_STATE_STRIDE;      // [128]
    float* q_smem = k_smem + BK;                                  // [128]

    const int state_head_offset = i_nh * V_dim * K_dim;

    // --- Collaborative state load: 32 threads x float4 = 128 floats per row ---
    #pragma unroll
    for (int row = 0; row < V3_BLOCK_SIZE; row++) {
        const float4 val = *reinterpret_cast<const float4*>(
            h0 + state_head_offset + (block_v_base + row) * K_dim + tid * KVEC);
        const int base = row * V3_STATE_STRIDE + tid * KVEC;
        state_smem[base + 0] = val.x;
        state_smem[base + 1] = val.y;
        state_smem[base + 2] = val.z;
        state_smem[base + 3] = val.w;
    }

    // --- Load k and q into shared memory (32 threads x 4 = 128 elements each) ---
    {
        const __nv_bfloat16* k_ptr = k + (i_n * H_qk + i_h) * K_dim + tid * KVEC;
        const __nv_bfloat16* q_ptr = q + (i_n * H_qk + i_h) * K_dim + tid * KVEC;
        #pragma unroll
        for (int i = 0; i < KVEC; i++) {
            k_smem[tid * KVEC + i] = __bfloat162float(k_ptr[i]);
            q_smem[tid * KVEC + i] = __bfloat162float(q_ptr[i]) * scale;
        }
    }
    // No explicit sync: single warp executes in lockstep (SIMT).
    // If correctness breaks on a future nvcc, add __syncwarp() here.

    // --- Load v (1 value per thread) ---
    const float v_val = __bfloat162float(
        v[(i_n * HV + i_hv) * V_dim + block_v_base + tid]);

    // --- Gates ---
    const float b_A  = A_log[i_hv];
    const float b_a  = __bfloat162float(a_gate[i_n * HV + i_hv]);
    const float b_dt = dt_bias[i_hv];
    const float b_b  = __bfloat162float(b_gate[i_n * HV + i_hv]);

    const float x  = b_a + b_dt;
    const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
    const float g  = expf(-expf(b_A) * sp);
    const float beta = 1.0f / (1.0f + expf(-b_b));

    // --- Pass 1: fused decay + k@S dot product ---
    // UNROLL_FACTOR controls the register/IPC tradeoff:
    //   128 (full): 255 regs + 256B spill, IPC 0.23, 7.14 us
    //   1 (none):   56 regs, IPC 0.15, 11.23 us
    //   Partial (8/16/32): should find middle ground
    const int my_row = tid * V3_STATE_STRIDE;
    float old_v = 0.0f;
    #pragma unroll
    for (int j = 0; j < BK; j++) {
        float h_val = g * state_smem[my_row + j];
        state_smem[my_row + j] = h_val;
        old_v += k_smem[j] * h_val;
    }

    // --- Pass 2: fused outer product update (k) + q@S output (q) ---
    const float dv = beta * (v_val - old_v);
    float out_val = 0.0f;
    #pragma unroll
    for (int j = 0; j < BK; j++) {
        float h_val = state_smem[my_row + j] + dv * k_smem[j];
        state_smem[my_row + j] = h_val;
        out_val += q_smem[j] * h_val;
    }

    // --- Store output (1 value per thread) ---
    output[(i_n * HV + i_hv) * V_dim + block_v_base + tid] =
        __float2bfloat16(out_val);

    // --- Collaborative state store ---
    __syncthreads();  // ensure all state_smem writes from compute are visible
    #pragma unroll
    for (int row = 0; row < V3_BLOCK_SIZE; row++) {
        const int base = row * V3_STATE_STRIDE + tid * KVEC;
        float4 val;
        val.x = state_smem[base + 0];
        val.y = state_smem[base + 1];
        val.z = state_smem[base + 2];
        val.w = state_smem[base + 3];
        *reinterpret_cast<float4*>(
            ht + state_head_offset + (block_v_base + row) * K_dim + tid * KVEC) = val;
    }
}

// ---------------------------------------------------------------------------
// TVM FFI host wrapper for V3
// ---------------------------------------------------------------------------

void KernelCudaV3(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView A_log,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView dt_bias,
    tvm::ffi::TensorView b_gate,
    double scale,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView new_state
) {
    const int B = q.size(0);
    const int H_qk = q.size(2);
    const int HV = v.size(2);
    const int V_dim = v.size(3);
    const int K_dim = q.size(3);

    dim3 grid(V_dim / V3_BLOCK_SIZE, B * HV);  // (4, 8)
    dim3 block(V3_BLOCK_SIZE);                  // 32

    DLDevice dev = q.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    gdn_decode_v3_kernel<<<grid, block, V3_SMEM_BYTES, stream>>>(
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_cuda_v3, KernelCudaV3);
