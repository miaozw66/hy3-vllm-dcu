// routed_w13.hip — batch routed-MoE w13 GEMM, M1 per (token, expert) pair.
//
// For each pair p with token t = tok_ids[p] and expert e = exp_ids[p],
// computes the int8 GEMM out[p, n] = sum_k qA[t, k] * w1[e, n, k], then
// dequantizes by per-token a_scale[t] and per-channel w1_scale[e, n] to bf16.
//
// Weight layout: raw w1 is [E, N, K] int8; launch_pack_routed_w1 packs each
// expert into the mmac-operand layout [n16, kt, lane, 8] so a wavefront's
// 8-byte loads are contiguous (512 B per (n16, kt) tile), byte-identical to
// the validated csrc M=1 kernel's pack.  The routed kernel therefore mirrors
// that kernel: block (n16, p) streams the full K slice of expert e across kt,
// writing only the M=1 row.  K%32==0, N%16==0.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace {

constexpr int kWaveSize = 64;
constexpr int kLaneBytes = 8;

using rw_int2_t = int __attribute__((ext_vector_type(2)));
using rw_int4_t = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ rw_int2_t rw_load_int8x8(const int8_t* p) {
  const auto* w = reinterpret_cast<const int32_t*>(p);
  rw_int2_t r;
  r.x = w[0];
  r.y = w[1];
  return r;
}

__device__ __forceinline__ rw_int4_t rw_mmac_i32_16x16x32_i8(
    rw_int2_t a, rw_int2_t b, rw_int4_t acc) {
#if defined(__gfx928__)
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %0"
               : "+v"(acc)
               : "v"(a), "v"(b));
  __builtin_amdgcn_sched_barrier(0);
  return acc;
#else
  return acc;
#endif
}

// raw w1 [E, N, K] -> packed [E, n16tiles, kt32, 64, 8]
__global__ void pack_routed_w1_kernel(
    const int8_t* __restrict__ raw, int8_t* __restrict__ packed,
    const int64_t total, const int E, const int N, const int K,
    const int kt32, const int64_t per_expert) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int e = static_cast<int>(idx / per_expert);
  const int64_t rem = idx % per_expert;
  const int byte = static_cast<int>(rem & 7);
  const int64_t t = rem >> 3;
  const int lane = static_cast<int>(t & 63);
  const int64_t tile = t >> 6;
  const int kt = static_cast<int>(tile % kt32);
  const int n16 = static_cast<int>(tile / kt32);
  const int k = kt * 32 + (lane >> 4) * 8 + byte;
  const int n = n16 * 16 + (lane & 15);
  packed[idx] = raw[(static_cast<int64_t>(e) * N + n) * K + k];
}

// Inverse of pack_routed_w1_kernel. One temporary output tensor is required;
// the Python integration copies it back into the original Parameter storage.
__global__ void unpack_routed_w1_kernel(
    const int8_t* __restrict__ packed, int8_t* __restrict__ raw,
    const int64_t total, const int E, const int N, const int K,
    const int kt32, const int64_t per_expert) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int e = static_cast<int>(idx / per_expert);
  const int64_t rem = idx % per_expert;
  const int byte = static_cast<int>(rem & 7);
  const int64_t t = rem >> 3;
  const int lane = static_cast<int>(t & 63);
  const int64_t tile = t >> 6;
  const int kt = static_cast<int>(tile % kt32);
  const int n16 = static_cast<int>(tile / kt32);
  const int k = kt * 32 + (lane >> 4) * 8 + byte;
  const int n = n16 * 16 + (lane & 15);
  raw[(static_cast<int64_t>(e) * N + n) * K + k] = packed[idx];
}

__global__ __launch_bounds__(kWaveSize, 4)
void routed_w13_kernel(
    const int8_t* __restrict__ qA,       // [M, K] int8 per-token-quantized
    const int8_t* __restrict__ w1p,      // packed [E, n16tiles, kt32, 64, 8]
    const float* __restrict__ a_scale,   // [M] per-token scale
    const float* __restrict__ w1_scale,  // [E, N] per-channel scale
    const int32_t* __restrict__ tok_ids, // [P]
    const int32_t* __restrict__ exp_ids, // [P]
    hip_bfloat16* __restrict__ out,      // [P, N] bf16
    const int N, const int n16tiles, const int kt32,
    const int64_t n16_stride) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n16 = static_cast<int>(blockIdx.x);
  const int p = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int t = tok_ids[p];
  const int e = exp_ids[p];

  const int64_t expert_base = static_cast<int64_t>(e) * n16tiles * n16_stride;
  const int8_t* const b_base =
      w1p + expert_base + static_cast<int64_t>(n16) * n16_stride;
  const int8_t* const a_base = qA + static_cast<int64_t>(t) * (kt32 * 32);

  rw_int4_t acc{0, 0, 0, 0};
#pragma unroll 4
  for (int kt = 0; kt < kt32; ++kt) {
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
    const rw_int2_t b =
        rw_load_int8x8(b_base + static_cast<int64_t>(kt) * 512 + lane * 8);
    acc = rw_mmac_i32_16x16x32_i8(a, b, acc);
  }

  if (row16 != 0) {
    return;
  }
  const float as = a_scale[t];
  const float* const ws = w1_scale + static_cast<int64_t>(e) * N + n16 * 16;
  hip_bfloat16* const ob = out + static_cast<int64_t>(p) * N + n16 * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = kg + i * 4;
    ob[column] = hip_bfloat16(static_cast<float>(acc[i]) * as * ws[column]);
  }
}

template <int kSplitK>
__global__ __launch_bounds__(kWaveSize, 4)
void routed_w13_splitk_partial_kernel(
    const int8_t* __restrict__ qA, const int8_t* __restrict__ w1p,
    const int32_t* __restrict__ tok_ids, const int32_t* __restrict__ exp_ids,
    int32_t* __restrict__ partial, const int P, const int N,
    const int n16tiles, const int kt32, const int64_t n16_stride) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n16 = static_cast<int>(blockIdx.x);
  const int p = static_cast<int>(blockIdx.y);
  const int split = static_cast<int>(blockIdx.z);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int t = tok_ids[p];
  const int e = exp_ids[p];
  const int tiles_per_split = kt32 / kSplitK;
  const int first_kt = split * tiles_per_split;
  const int8_t* const a_base =
      qA + static_cast<int64_t>(t) * (kt32 * 32);
  const int8_t* const b_base =
      w1p + static_cast<int64_t>(e) * n16tiles * n16_stride +
      static_cast<int64_t>(n16) * n16_stride;

  rw_int4_t acc{0, 0, 0, 0};
#pragma unroll 4
  for (int li = 0; li < tiles_per_split; ++li) {
    const int kt = first_kt + li;
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
    const rw_int2_t b =
        rw_load_int8x8(b_base + static_cast<int64_t>(kt) * 512 + lane * 8);
    acc = rw_mmac_i32_16x16x32_i8(a, b, acc);
  }
  if (row16 != 0) {
    return;
  }
  int32_t* const dst =
      partial + (static_cast<int64_t>(split) * P + p) * N + n16 * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[kg + i * 4] = acc[i];
  }
}

template <int kSplitK>
__global__ __launch_bounds__(64)
void routed_w13_splitk_combine_kernel(
    const int32_t* __restrict__ partial,
    const float* __restrict__ a_scale,
    const float* __restrict__ w1_scale,
    const int32_t* __restrict__ tok_ids,
    const int32_t* __restrict__ exp_ids,
    hip_bfloat16* __restrict__ out, const int P, const int N) {
  const int n16 = static_cast<int>(blockIdx.x);
  const int p = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  if (lane >= 16) {
    return;
  }
  const int col = n16 * 16 + lane;
  int32_t sum = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    sum += partial[(static_cast<int64_t>(s) * P + p) * N + col];
  }
  const int t = tok_ids[p];
  const int e = exp_ids[p];
  out[static_cast<int64_t>(p) * N + col] = hip_bfloat16(
      static_cast<float>(sum) * a_scale[t] *
      w1_scale[static_cast<int64_t>(e) * N + col]);
}

template <int kWaves>
__global__ __launch_bounds__(kWaveSize * kWaves, 2)
void routed_w13_multiwave_kernel(
    const int8_t* __restrict__ qA, const int8_t* __restrict__ w1p,
    const float* __restrict__ a_scale, const float* __restrict__ w1_scale,
    const int32_t* __restrict__ tok_ids, const int32_t* __restrict__ exp_ids,
    hip_bfloat16* __restrict__ out, const int N, const int n16tiles,
    const int kt32, const int64_t n16_stride) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int n16 = static_cast<int>(blockIdx.x) * kWaves + wave;
  const int p = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int t = tok_ids[p];
  const int e = exp_ids[p];
  const int8_t* const a_base = qA + static_cast<int64_t>(t) * (kt32 * 32);
  const int8_t* const b_base =
      w1p + static_cast<int64_t>(e) * n16tiles * n16_stride +
      static_cast<int64_t>(n16) * n16_stride;
  rw_int4_t acc{0, 0, 0, 0};
#pragma unroll 4
  for (int kt = 0; kt < kt32; ++kt) {
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
    const rw_int2_t b =
        rw_load_int8x8(b_base + static_cast<int64_t>(kt) * 512 + lane * 8);
    acc = rw_mmac_i32_16x16x32_i8(a, b, acc);
  }
  if (row16 != 0) {
    return;
  }
  const float as = a_scale[t];
  const float* const ws = w1_scale + static_cast<int64_t>(e) * N + n16 * 16;
  hip_bfloat16* const ob = out + static_cast<int64_t>(p) * N + n16 * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = kg + i * 4;
    ob[column] = hip_bfloat16(static_cast<float>(acc[i]) * as * ws[column]);
  }
}

template <int kTiles>
__global__ __launch_bounds__(kWaveSize, 4)
void routed_w13_multitile_kernel(
    const int8_t* __restrict__ qA, const int8_t* __restrict__ w1p,
    const float* __restrict__ a_scale, const float* __restrict__ w1_scale,
    const int32_t* __restrict__ tok_ids, const int32_t* __restrict__ exp_ids,
    hip_bfloat16* __restrict__ out, const int N, const int n16tiles,
    const int kt32, const int64_t n16_stride) {
  const int lane = static_cast<int>(threadIdx.x);
  const int group = static_cast<int>(blockIdx.x);
  const int p = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int t = tok_ids[p];
  const int e = exp_ids[p];
  const int8_t* const a_base = qA + static_cast<int64_t>(t) * (kt32 * 32);
  const int64_t expert_base = static_cast<int64_t>(e) * n16tiles * n16_stride;

  rw_int4_t acc[kTiles];
#pragma unroll
  for (int u = 0; u < kTiles; ++u) {
    acc[u] = rw_int4_t{0, 0, 0, 0};
  }
#pragma unroll 4
  for (int kt = 0; kt < kt32; ++kt) {
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
#pragma unroll
    for (int u = 0; u < kTiles; ++u) {
      const int n16 = group * kTiles + u;
      const int8_t* const b =
          w1p + expert_base + static_cast<int64_t>(n16) * n16_stride +
          static_cast<int64_t>(kt) * 512 + lane * 8;
      acc[u] = rw_mmac_i32_16x16x32_i8(a, rw_load_int8x8(b), acc[u]);
    }
  }
  if (row16 != 0) {
    return;
  }
  const float as = a_scale[t];
#pragma unroll
  for (int u = 0; u < kTiles; ++u) {
    const int n16 = group * kTiles + u;
    const float* const ws = w1_scale + static_cast<int64_t>(e) * N + n16 * 16;
    hip_bfloat16* const ob = out + static_cast<int64_t>(p) * N + n16 * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int column = kg + i * 4;
      ob[column] = hip_bfloat16(static_cast<float>(acc[u][i]) * as * ws[column]);
    }
  }
}

__global__ __launch_bounds__(256)
void silu_requant_kernel(const hip_bfloat16* __restrict__ w13,
                         int8_t* __restrict__ qact,
                         float* __restrict__ act_scale, const int P) {
  constexpr int kInter = 192;
  __shared__ float vals[256];
  __shared__ float maxbuf[256];
  const int p = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(threadIdx.x);
  float value = 0.0f;
  if (col < kInter) {
    const int64_t base = static_cast<int64_t>(p) * (2 * kInter);
    const float gate = static_cast<float>(w13[base + col]);
    const float up = static_cast<float>(w13[base + kInter + col]);
    // Real path materializes activation into bf16 before requantization.
    value = static_cast<float>(
        hip_bfloat16((gate / (1.0f + expf(-gate))) * up));
  }
  vals[col] = value;
  maxbuf[col] = fabsf(value);
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (col < stride) {
      maxbuf[col] = fmaxf(maxbuf[col], maxbuf[col + stride]);
    }
    __syncthreads();
  }
  const float scale = fmaxf(maxbuf[0], 1.0e-10f) / 127.0f;
  if (col == 0) {
    act_scale[p] = scale;
  }
  if (col < kInter) {
    float q = roundf(vals[col] / scale);
    q = fminf(127.0f, fmaxf(-128.0f, q));
    qact[static_cast<int64_t>(p) * kInter + col] = static_cast<int8_t>(q);
  }
}

template <int kWaves>
__global__ __launch_bounds__(kWaveSize * kWaves, 2)
void routed_w2_multiwave_kernel(
    const int8_t* __restrict__ qact, const int8_t* __restrict__ w2p,
    const float* __restrict__ act_scale, const float* __restrict__ w2_scale,
    const int32_t* __restrict__ exp_ids,
    const float* __restrict__ topk_weights,
    hip_bfloat16* __restrict__ pair_out, const int N, const int n16tiles,
    const int kt32, const int64_t n16_stride) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int n16 = static_cast<int>(blockIdx.x) * kWaves + wave;
  const int p = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int e = exp_ids[p];
  const int8_t* const a_base = qact + static_cast<int64_t>(p) * (kt32 * 32);
  const int8_t* const b_base =
      w2p + static_cast<int64_t>(e) * n16tiles * n16_stride +
      static_cast<int64_t>(n16) * n16_stride;
  rw_int4_t acc{0, 0, 0, 0};
#pragma unroll
  for (int kt = 0; kt < kt32; ++kt) {
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
    const rw_int2_t b =
        rw_load_int8x8(b_base + static_cast<int64_t>(kt) * 512 + lane * 8);
    acc = rw_mmac_i32_16x16x32_i8(a, b, acc);
  }
  if (row16 != 0) {
    return;
  }
  const float output_scale = act_scale[p] * topk_weights[p];
  const float* const ws = w2_scale + static_cast<int64_t>(e) * N + n16 * 16;
  hip_bfloat16* const ob = pair_out + static_cast<int64_t>(p) * N + n16 * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = kg + i * 4;
    ob[column] = hip_bfloat16(
        static_cast<float>(acc[i]) * output_scale * ws[column]);
  }
}

// w2 fixed shape K=192,N=4096. One wave computes kTiles adjacent N16 tiles;
// all MMACs share each activation load. Only row16==0 is a real M=1 row.
template <int kTiles>
__global__ __launch_bounds__(kWaveSize, 4)
void routed_w2_kernel(
    const int8_t* __restrict__ qact,
    const int8_t* __restrict__ w2p,
    const float* __restrict__ act_scale,
    const float* __restrict__ w2_scale,
    const int32_t* __restrict__ exp_ids,
    const float* __restrict__ topk_weights,
    hip_bfloat16* __restrict__ pair_out,
    const int N, const int n16tiles, const int kt32,
    const int64_t n16_stride) {
  const int lane = static_cast<int>(threadIdx.x);
  const int group = static_cast<int>(blockIdx.x);
  const int p = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int e = exp_ids[p];
  const int8_t* const a_base = qact + static_cast<int64_t>(p) * (kt32 * 32);
  const int64_t expert_base = static_cast<int64_t>(e) * n16tiles * n16_stride;

  rw_int4_t acc[kTiles];
#pragma unroll
  for (int u = 0; u < kTiles; ++u) {
    acc[u] = rw_int4_t{0, 0, 0, 0};
  }
#pragma unroll
  for (int kt = 0; kt < kt32; ++kt) {
    const rw_int2_t a = rw_load_int8x8(a_base + kt * 32 + kg * 8);
#pragma unroll
    for (int u = 0; u < kTiles; ++u) {
      const int n16 = group * kTiles + u;
      const int8_t* const b =
          w2p + expert_base + static_cast<int64_t>(n16) * n16_stride +
          static_cast<int64_t>(kt) * 512 + lane * 8;
      acc[u] = rw_mmac_i32_16x16x32_i8(a, rw_load_int8x8(b), acc[u]);
    }
  }

  if (row16 != 0) {
    return;
  }
  const float output_scale = act_scale[p] * topk_weights[p];
#pragma unroll
  for (int u = 0; u < kTiles; ++u) {
    const int n16 = group * kTiles + u;
    const float* const ws = w2_scale + static_cast<int64_t>(e) * N + n16 * 16;
    hip_bfloat16* const ob = pair_out + static_cast<int64_t>(p) * N + n16 * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int column = kg + i * 4;
      ob[column] = hip_bfloat16(
          static_cast<float>(acc[u][i]) * output_scale * ws[column]);
    }
  }
}

__global__ __launch_bounds__(256)
void routed_sum_kernel(const hip_bfloat16* __restrict__ pair_out,
                       hip_bfloat16* __restrict__ out,
                       const int M, const int topk, const int N) {
  const int m = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= N) {
    return;
  }
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (k < topk) {
      const int p = m * topk + k;
      sum += static_cast<float>(
          pair_out[static_cast<int64_t>(p) * N + col]);
    }
  }
  out[static_cast<int64_t>(m) * N + col] = hip_bfloat16(sum);
}

}  // namespace

// Host: pack one expert-plane stack [E, N, K] into mmac-operand layout.
int64_t launch_pack_routed_w1(const int8_t* raw, int8_t* packed, int E, int N,
                              int K, hipStream_t stream) {
  const int n16tiles = N / 16;
  const int kt32 = K / 32;
  const int64_t per_expert =
      static_cast<int64_t>(n16tiles) * kt32 * kWaveSize * kLaneBytes;
  const int64_t total = static_cast<int64_t>(E) * per_expert;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(pack_routed_w1_kernel, dim3(blocks), dim3(threads), 0,
                     stream, raw, packed, total, E, N, K, kt32, per_expert);
  return total;
}

int64_t launch_unpack_routed_w1(const int8_t* packed, int8_t* raw, int E,
                                int N, int K, hipStream_t stream) {
  const int n16tiles = N / 16;
  const int kt32 = K / 32;
  const int64_t per_expert =
      static_cast<int64_t>(n16tiles) * kt32 * kWaveSize * kLaneBytes;
  const int64_t total = static_cast<int64_t>(E) * per_expert;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  hipLaunchKernelGGL(unpack_routed_w1_kernel, dim3(blocks), dim3(threads), 0,
                     stream, packed, raw, total, E, N, K, kt32, per_expert);
  return total;
}

int64_t launch_routed_w13(
    const int8_t* qA, const int8_t* w1p, const float* a_scale,
    const float* w1_scale, const int32_t* tok_ids, const int32_t* exp_ids,
    void* out, int P, int K, int N, hipStream_t stream) {
  const int n16tiles = N / 16;
  const int kt32 = K / 32;
  const int64_t n16_stride =
      static_cast<int64_t>(kt32) * kWaveSize * kLaneBytes;
  constexpr int kWaves = 4;
  if (n16tiles % kWaves == 0) {
    hipLaunchKernelGGL((routed_w13_multiwave_kernel<kWaves>),
                       dim3(n16tiles / kWaves, P), dim3(kWaveSize * kWaves), 0,
                       stream, qA, w1p, a_scale, w1_scale, tok_ids, exp_ids,
                       reinterpret_cast<hip_bfloat16*>(out), N, n16tiles, kt32,
                       n16_stride);
  } else {
    // The public w13 op permits any N multiple of 16; preserve all tiles.
    hipLaunchKernelGGL(routed_w13_kernel, dim3(n16tiles, P), dim3(kWaveSize),
                       0, stream, qA, w1p, a_scale, w1_scale, tok_ids, exp_ids,
                       reinterpret_cast<hip_bfloat16*>(out), N, n16tiles, kt32,
                       n16_stride);
  }
  return static_cast<int64_t>(P) * N * K;
}

void launch_routed_w13_splitk2(
    const int8_t* qA, const int8_t* w1p, const float* a_scale,
    const float* w1_scale, const int32_t* tok_ids, const int32_t* exp_ids,
    int32_t* partial, void* out, int P, int K, int N, hipStream_t stream) {
  constexpr int kSplitK = 2;
  const int n16tiles = N / 16;
  const int kt32 = K / 32;
  const int64_t n16_stride =
      static_cast<int64_t>(kt32) * kWaveSize * kLaneBytes;
  hipLaunchKernelGGL((routed_w13_splitk_partial_kernel<kSplitK>),
                     dim3(n16tiles, P, kSplitK), dim3(kWaveSize), 0, stream,
                     qA, w1p, tok_ids, exp_ids, partial, P, N, n16tiles,
                     kt32, n16_stride);
  hipLaunchKernelGGL((routed_w13_splitk_combine_kernel<kSplitK>),
                     dim3(n16tiles, P), dim3(64), 0, stream, partial,
                     a_scale, w1_scale, tok_ids, exp_ids,
                     reinterpret_cast<hip_bfloat16*>(out), P, N);
}

void launch_silu_requant(const void* w13_out, int8_t* qact,
                         float* a2_scale, int P, hipStream_t stream) {
  hipLaunchKernelGGL(silu_requant_kernel, dim3(P), dim3(256), 0, stream,
                     reinterpret_cast<const hip_bfloat16*>(w13_out), qact,
                     a2_scale, P);
}

void launch_routed_w2(const int8_t* qact, const int8_t* w2p,
                      const float* a2_scale, const float* w2_scale,
                      const int32_t* exp_ids, const float* topk_weights,
                      void* pair_out, int P, int K2, int N2,
                      hipStream_t stream) {
  const int n16tiles = N2 / 16;
  const int kt32 = K2 / 32;
  const int64_t n16_stride =
      static_cast<int64_t>(kt32) * kWaveSize * kLaneBytes;
  constexpr int kTiles = 1;
  hipLaunchKernelGGL((routed_w2_kernel<kTiles>),
                     dim3(n16tiles / kTiles, P), dim3(kWaveSize), 0, stream,
                     qact, w2p, a2_scale, w2_scale, exp_ids, topk_weights,
                     reinterpret_cast<hip_bfloat16*>(pair_out), N2,
                     n16tiles, kt32, n16_stride);
}

void launch_routed_sum(const void* pair_out, void* out, int M, int topk,
                       int N2, hipStream_t stream) {
  hipLaunchKernelGGL(routed_sum_kernel, dim3((N2 + 255) / 256, M),
                     dim3(256), 0, stream,
                     reinterpret_cast<const hip_bfloat16*>(pair_out),
                     reinterpret_cast<hip_bfloat16*>(out), M, topk, N2);
}

void launch_routed_full(
    const int8_t* qA, const int8_t* w1p, const int8_t* w2p,
    const float* a1_scale, const float* w1_scale, const float* w2_scale,
    const int32_t* tok_ids, const int32_t* exp_ids,
    const float* topk_weights, void* w13_out, int8_t* qact,
    float* a2_scale, void* pair_out, void* out,
    int M, int P, int K1, int N1, int K2, int N2, int topk,
    hipStream_t stream) {
  launch_routed_w13(qA, w1p, a1_scale, w1_scale, tok_ids, exp_ids,
                    w13_out, P, K1, N1, stream);
  launch_silu_requant(w13_out, qact, a2_scale, P, stream);
  launch_routed_w2(qact, w2p, a2_scale, w2_scale, exp_ids, topk_weights,
                   pair_out, P, K2, N2, stream);
  launch_routed_sum(pair_out, out, M, topk, N2, stream);
}
