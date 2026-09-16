// w8a8_m32_smallm.hpp — m <= 32 single-wavefront split-k DUMMA kernels.
//
// Consumed by the four M4096 deliverable .hip files via their launch_w8a8_gemm
// small-M arm. Design (verified 2026-08-31, see plan
// sleepy-dreaming-chipmunk.md):
//   * 64-thread single wavefront per 16x16 output tile; no LDS staging, no
//     s_barrier, no combine barrier — data is read straight from global into
//     the v_mmac_i32_16x16x32_i8 operands (the csrc m=1 kernel pattern).
//   * split-k two-stage: partial kernel writes int32 partial sums into the
//     caller workspace, combine kernel reduces kSplitK slabs, applies the
//     per-row activation scale and per-column weight scale, masks rows >= m,
//     and stores bf16.
//   * b is the packed n-major [N, K] weight already produced by each .hip's
//     launch_pack_w8a8_weight for the exact assigned (k, n) pair:
//     packed[col*K + kk] == raw[kk*N + col]. No new pack is needed.
//   * accumulator ownership (gfx928): lane&15 = row, lane>>4 = col_mod4,
//     acc[i] -> column col_mod4 + 4*i.
//   * A rows with base_row+row16 >= m are clamped to row 0 on load; their
//     garbage acc is discarded by the combine store mask.
//
// partial layout (int32): partial[(split*bands + band)*n16tiles*256
//     + n16*256 + row16*16 + (col_mod4 + 4*i)].

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

namespace {

using zth_int2_t = int __attribute__((ext_vector_type(2)));
using zth_int4_t = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ zth_int2_t zth_load_int8x8(const int8_t* p) {
  const auto* w = reinterpret_cast<const int32_t*>(p);
  zth_int2_t r;
  r.x = w[0];
  r.y = w[1];
  return r;
}

__device__ __forceinline__ zth_int4_t zth_mmac_i32_16x16x32_i8(
    zth_int2_t a, zth_int2_t b, zth_int4_t acc) {
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

template <int kSplitK>
__global__ __launch_bounds__(64, 4)
void w8a8_dumma_m32_splitk_partial_kernel(
    const int8_t* __restrict__ a,     // [m, k] row-major int8 activation
    const int8_t* __restrict__ b,     // n-major [n, k] int8 weight (packed)
    int32_t* __restrict__ partial,    // workspace, split-k partials
    int m, int n, int k, int bands, int n16tiles) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n16 = static_cast<int>(blockIdx.x);
  const int band = static_cast<int>(blockIdx.y);
  const int split = static_cast<int>(blockIdx.z);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int base_row = band * 16;
  const int kt32 = k >> 5;                 // k / 32
  const int kt_per_split = kt32 / kSplitK; // dispatch guards exact division
  const int first_kt = split * kt_per_split;

  const int arow = (base_row + row16 < m) ? (base_row + row16) : 0;
  const int64_t a_base =
      static_cast<int64_t>(arow) * k + static_cast<int64_t>(first_kt) * 32 + kg * 8;
  const int64_t b_base =
      static_cast<int64_t>(n16 * 16 + row16) * k + static_cast<int64_t>(first_kt) * 32 + kg * 8;

  zth_int4_t acc{0, 0, 0, 0};
#pragma unroll 4
  for (int li = 0; li < kt_per_split; ++li) {
    const int64_t ko = static_cast<int64_t>(li) * 32;
    const zth_int2_t av = zth_load_int8x8(a + a_base + ko);
    const zth_int2_t bv = zth_load_int8x8(b + b_base + ko);
    acc = zth_mmac_i32_16x16x32_i8(av, bv, acc);
  }

  if (base_row + row16 < m) {
    const int64_t tile =
        (static_cast<int64_t>(split) * bands + band) * n16tiles + n16;
    int32_t* dst = partial + tile * 256 + row16 * 16 + kg;
    dst[0] = acc[0];
    dst[4] = acc[1];
    dst[8] = acc[2];
    dst[12] = acc[3];
  }
}

// Two-tile variant: one wavefront computes two adjacent 16-col tiles
// (16x32 columns) with 8 accumulators, sharing the A-row loads. grid.x must be
// n16tiles/2. Halves the partial-block count, which is the dominant cost for
// large-N shapes (o_proj/down n_tile=256). Combine is unchanged (each 16x16
// tile keeps its own partial slab).
template <int kSplitK>
__global__ __launch_bounds__(64, 4)
void w8a8_dumma_m32_splitk_partial2_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ partial,
    int m, int n, int k, int bands, int n16tiles) {
  const int lane = static_cast<int>(threadIdx.x);
  const int tp = static_cast<int>(blockIdx.x);
  const int band = static_cast<int>(blockIdx.y);
  const int split = static_cast<int>(blockIdx.z);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int base_row = band * 16;
  const int kt32 = k >> 5;
  const int kt_per_split = kt32 / kSplitK;
  const int first_kt = split * kt_per_split;

  const int arow = (base_row + row16 < m) ? (base_row + row16) : 0;
  const int64_t a_base =
      static_cast<int64_t>(arow) * k + static_cast<int64_t>(first_kt) * 32 + kg * 8;
  const int nA = tp * 2;
  const int nB = nA + 1;
  const int64_t b_baseA =
      static_cast<int64_t>(nA * 16 + row16) * k + static_cast<int64_t>(first_kt) * 32 + kg * 8;
  const int64_t b_baseB =
      static_cast<int64_t>(nB * 16 + row16) * k + static_cast<int64_t>(first_kt) * 32 + kg * 8;

  zth_int4_t accA{0, 0, 0, 0};
  zth_int4_t accB{0, 0, 0, 0};
#pragma unroll 4
  for (int li = 0; li < kt_per_split; ++li) {
    const int64_t ko = static_cast<int64_t>(li) * 32;
    const zth_int2_t av = zth_load_int8x8(a + a_base + ko);
    const zth_int2_t bvA = zth_load_int8x8(b + b_baseA + ko);
    const zth_int2_t bvB = zth_load_int8x8(b + b_baseB + ko);
    accA = zth_mmac_i32_16x16x32_i8(av, bvA, accA);
    accB = zth_mmac_i32_16x16x32_i8(av, bvB, accB);
  }

  if (base_row + row16 < m) {
    const int64_t slab =
        (static_cast<int64_t>(split) * bands + band) * n16tiles * 256;
    int32_t* dstA = partial + slab + nA * 256 + row16 * 16 + kg;
    dstA[0] = accA[0];
    dstA[4] = accA[1];
    dstA[8] = accA[2];
    dstA[12] = accA[3];
    int32_t* dstB = partial + slab + nB * 256 + row16 * 16 + kg;
    dstB[0] = accB[0];
    dstB[4] = accB[1];
    dstB[8] = accB[2];
    dstB[12] = accB[3];
  }
}

// Fused single-kernel variant for large-N, small-K shapes (o_proj k=1024 /
// down k=192): each wavefront computes kTiles adjacent 16-col tiles over the
// full K in one kernel, then applies the per-row x_scale and per-col
// weight_scale, masks rows >= m, and stores bf16 directly. No workspace, no
// second (combine) launch, no partial round-trip -- for these shapes the
// two-stage split-k overhead dominates (L=2 partial alone profiled ~41us on
// o_proj m=32). grid.x = n16tiles/kTiles, grid.y = bands.
template <int kTiles>
__global__ __launch_bounds__(64, 4)
void w8a8_dumma_m32_fused_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,
    int m, int n, int k, int bands) {
  const int lane = static_cast<int>(threadIdx.x);
  const int tp = static_cast<int>(blockIdx.x);
  const int band = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int row = band * 16 + row16;
  const int kt32 = k >> 5;
  const int arow = (row < m) ? row : 0;
  const int64_t a_base = static_cast<int64_t>(arow) * k + kg * 8;
  const float xs = (row < m) ? x_scale[row] : 0.0f;
  const int64_t out_base = static_cast<int64_t>(row) * n;

  zth_int4_t acc[kTiles];
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    acc[t] = zth_int4_t{0, 0, 0, 0};
  }
  const int col_ofs = tp * kTiles * 16;
#pragma unroll 4
  for (int li = 0; li < kt32; ++li) {
    const int64_t ko = static_cast<int64_t>(li) * 32;
    const zth_int2_t av = zth_load_int8x8(a + a_base + ko);
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
      const int64_t b_base =
          static_cast<int64_t>(col_ofs + t * 16 + row16) * k + ko + kg * 8;
      const zth_int2_t bv = zth_load_int8x8(b + b_base);
      acc[t] = zth_mmac_i32_16x16x32_i8(av, bv, acc[t]);
    }
  }

  if (row >= m) {
    return;
  }
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    const int col0 = col_ofs + t * 16;
    const int64_t ob = out_base + col0 + kg;
    out[ob] =
        __float2bfloat16(static_cast<float>(acc[t][0]) * xs * weight_scale[col0 + kg]);
    out[ob + 4] =
        __float2bfloat16(static_cast<float>(acc[t][1]) * xs * weight_scale[col0 + kg + 4]);
    out[ob + 8] =
        __float2bfloat16(static_cast<float>(acc[t][2]) * xs * weight_scale[col0 + kg + 8]);
    out[ob + 12] =
        __float2bfloat16(static_cast<float>(acc[t][3]) * xs * weight_scale[col0 + kg + 12]);
  }
}

template <int kSplitK>
__global__ __launch_bounds__(64, 4)
void w8a8_dumma_m32_combine_kernel(
    const int32_t* __restrict__ partial,
    const float* __restrict__ x_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ out,   // [m, n] bf16
    int m, int n, int k, int bands, int n16tiles) {
  const int lane = static_cast<int>(threadIdx.x);
  const int n16 = static_cast<int>(blockIdx.x);
  const int band = static_cast<int>(blockIdx.y);
  const int row16 = lane & 15;
  const int kg = lane >> 4;
  const int row = band * 16 + row16;
  if (row >= m) {
    return;
  }
  const int64_t tile_off =
      (static_cast<int64_t>(band) * n16tiles + n16) * 256 + row16 * 16 + kg;
  const int64_t slab = static_cast<int64_t>(bands) * n16tiles * 256;
  int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    const int32_t* p = partial + s * slab + tile_off;
    s0 += p[0];
    s1 += p[4];
    s2 += p[8];
    s3 += p[12];
  }
  const int col0 = n16 * 16;
  const float xs = x_scale[row];
  const int64_t out_base = static_cast<int64_t>(row) * n + col0;
  out[out_base + kg] =
      __float2bfloat16(static_cast<float>(s0) * xs * weight_scale[col0 + kg]);
  out[out_base + kg + 4] =
      __float2bfloat16(static_cast<float>(s1) * xs * weight_scale[col0 + kg + 4]);
  out[out_base + kg + 8] =
      __float2bfloat16(static_cast<float>(s2) * xs * weight_scale[col0 + kg + 8]);
  out[out_base + kg + 12] =
      __float2bfloat16(static_cast<float>(s3) * xs * weight_scale[col0 + kg + 12]);
}

}  // namespace
