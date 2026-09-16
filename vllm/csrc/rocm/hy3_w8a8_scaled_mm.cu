#include "rocm/ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr int kHy3M = 1;
constexpr int kHy3K = 4096;
constexpr int kHy3N = 384;
constexpr int kN16Tiles = kHy3N / 16;
constexpr int kK32Tiles = kHy3K / 32;
constexpr int kWaveSize = 64;
constexpr int kPackedBytesPerLane = 8;

using int2_t = int __attribute__((ext_vector_type(2)));
using int4_t = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int2_t load_int8x8(const int8_t* ptr) {
  const auto* words = reinterpret_cast<const int32_t*>(ptr);
  int2_t result;
  result.x = words[0];
  result.y = words[1];
  return result;
}

// packed B fragment 与 MMAC operand 使用相同的 lane 布局。
__device__ __forceinline__ int4_t mmac_i32_16x16x32_i8(int2_t a, int2_t b,
                                                        int4_t acc) {
#if defined(__gfx928__)
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %0"
               : "+v"(acc)
               : "v"(a), "v"(b));
  __builtin_amdgcn_sched_barrier(0);
  return acc;
#else
  // 该 custom op 仅在 gfx928 分派，其他目标不汇编不支持的 MMAC 指令。
  return acc;
#endif
}

__global__ void hy3_w8a8_pack_kernel(const int8_t* __restrict__ weight_kn,
                                      int8_t* __restrict__ packed_weight) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int64_t kPackedElements =
      static_cast<int64_t>(kN16Tiles) * kK32Tiles * kWaveSize * kPackedBytesPerLane;
  if (index >= kPackedElements) {
    return;
  }

  const int byte = index % kPackedBytesPerLane;
  const int64_t lane_index = index / kPackedBytesPerLane;
  const int lane = lane_index % kWaveSize;
  const int64_t tile_index = lane_index / kWaveSize;
  const int kt = tile_index % kK32Tiles;
  const int n16 = tile_index / kK32Tiles;
  const int k_group = lane >> 4;
  const int col16 = lane & 15;
  const int k = kt * 32 + k_group * 8 + byte;
  const int n = n16 * 16 + col16;

  packed_weight[index] = weight_kn[static_cast<int64_t>(k) * kHy3N + n];
}

__device__ __forceinline__ int32_t dot4_i32_i8(int32_t a, int32_t b, int32_t acc) {
#if defined(__gfx928__)
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %0"
               : "+v"(acc)
               : "v"(a), "v"(b));
#endif
  return acc;
}

// M=1 专用 B layout: [N64][K4][lane64][4B]。
__global__ void hy3_w8a8_dot_pack_kernel(const int8_t* __restrict__ weight_kn,
                                          int8_t* __restrict__ packed_weight) {
  constexpr int kN64Tiles = kHy3N / kWaveSize;
  constexpr int kK4Tiles = kHy3K / 4;
  constexpr int64_t kElements =
      static_cast<int64_t>(kN64Tiles) * kK4Tiles * kWaveSize * 4;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kElements) {
    return;
  }
  const int byte = index & 3;
  const int64_t lane_index = index >> 2;
  const int lane = lane_index & 63;
  const int64_t tile_index = lane_index >> 6;
  const int k4 = tile_index & (kK4Tiles - 1);
  const int n64 = tile_index >> 10;
  packed_weight[index] = weight_kn[static_cast<int64_t>(k4 * 4 + byte) * kHy3N +
                                    n64 * kWaveSize + lane];
}

// 一个 wave 负责 N=16；A 复制到逻辑 M16，仅输出 row-zero accumulator。
__global__ __launch_bounds__(kWaveSize, 4)
void hy3_w8a8_m1k4096n384_kernel(const int8_t* __restrict__ activation,
                                  const int8_t* __restrict__ packed_weight,
                                  const float* __restrict__ activation_scale,
                                  const float* __restrict__ weight_scale,
                                  hip_bfloat16* __restrict__ output) {
  const int lane = threadIdx.x;
  const int n16 = blockIdx.x;
  const int row16 = lane & 15;
  const int k_group = lane >> 4;

  int4_t accumulator{};
  const int8_t* const packed_base = packed_weight +
      static_cast<int64_t>(n16) * kK32Tiles * kWaveSize * kPackedBytesPerLane;

#pragma unroll
  for (int kt = 0; kt < kK32Tiles; ++kt) {
    const int2_t a = load_int8x8(activation + kt * 32 + k_group * 8);
    const int2_t b = load_int8x8(
        packed_base + static_cast<int64_t>(kt) * kWaveSize * kPackedBytesPerLane +
        lane * kPackedBytesPerLane);
    accumulator = mmac_i32_16x16x32_i8(a, b, accumulator);
  }

  if (row16 != 0) {
    return;
  }

  const float a_scale = activation_scale[0];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = n16 * 16 + k_group + i * 4;
    output[column] = hip_bfloat16(static_cast<float>(accumulator[i]) * a_scale *
                                  weight_scale[column]);
  }
}

template <int SplitK>
__global__ __launch_bounds__(kWaveSize, 4)
void hy3_w8a8_m1k4096n384_splitk_kernel(
    const int8_t* __restrict__ activation,
    const int8_t* __restrict__ packed_weight,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    int32_t* __restrict__ partial,
    uint32_t* __restrict__ tickets,
    hip_bfloat16* __restrict__ output) {
  constexpr int kTilesPerSplit = kK32Tiles / SplitK;
  const int lane = threadIdx.x;
  const int n16 = blockIdx.x;
  const int split = blockIdx.y;
  const int row16 = lane & 15;
  const int k_group = lane >> 4;
  const int first_kt = split * kTilesPerSplit;

  int4_t accumulator{};
  const int8_t* const packed_base = packed_weight +
      static_cast<int64_t>(n16) * kK32Tiles * kWaveSize * kPackedBytesPerLane;
#pragma unroll
  for (int local_kt = 0; local_kt < kTilesPerSplit; ++local_kt) {
    const int kt = first_kt + local_kt;
    const int2_t a = load_int8x8(activation + kt * 32 + k_group * 8);
    const int2_t b = load_int8x8(
        packed_base + static_cast<int64_t>(kt) * kWaveSize * kPackedBytesPerLane +
        lane * kPackedBytesPerLane);
    accumulator = mmac_i32_16x16x32_i8(a, b, accumulator);
  }

  if (row16 == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int column = n16 * 16 + k_group + i * 4;
      partial[static_cast<int64_t>(split) * kHy3N + column] = accumulator[i];
    }
  }

  uint32_t ticket = 0;
  if (lane == 0) {
    ticket = __hip_atomic_fetch_add(tickets + n16, 1u, __ATOMIC_ACQ_REL,
                                    __HIP_MEMORY_SCOPE_AGENT);
  }
  ticket = __builtin_amdgcn_readfirstlane(ticket);
  if (ticket != static_cast<uint32_t>(SplitK - 1) || row16 != 0) {
    return;
  }

  const float a_scale = activation_scale[0];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = n16 * 16 + k_group + i * 4;
    int32_t sum = 0;
#pragma unroll
    for (int s = 0; s < SplitK; ++s) {
      sum += partial[static_cast<int64_t>(s) * kHy3N + column];
    }
    output[column] = hip_bfloat16(static_cast<float>(sum) * a_scale *
                                  weight_scale[column]);
  }

  if (lane == 0) {
    tickets[n16] = 0;
  }
}

// M=1 向量 dot 路径：每个 lane 只计算一个真实输出列，避免 M16 MMAC 浪费。
__global__ __launch_bounds__(kWaveSize, 4)
void hy3_w8a8_m1k4096n384_dot_splitk32_kernel(
    const int8_t* __restrict__ activation, const int8_t* __restrict__ packed_weight,
    const float* __restrict__ activation_scale, const float* __restrict__ weight_scale,
    int32_t* __restrict__ partial, uint32_t* __restrict__ tickets,
    hip_bfloat16* __restrict__ output) {
  constexpr int kSplitK = 32;
  constexpr int kK4Tiles = kHy3K / 4;
  constexpr int kTilesPerSplit = kK4Tiles / kSplitK;
  constexpr int kN64Tiles = kHy3N / kWaveSize;
  const int lane = threadIdx.x;
  const int n64 = blockIdx.x;
  const int split = blockIdx.y;
  const int column = n64 * kWaveSize + lane;
  const int first_k4 = split * kTilesPerSplit;
  __shared__ __align__(16) int32_t a_lds[kTilesPerSplit];
  if (lane < kTilesPerSplit) {
    a_lds[lane] = reinterpret_cast<const int32_t*>(activation + first_k4 * 4)[lane];
  }
  __syncthreads();

  int32_t acc = 0;
  const int8_t* const packed_base = packed_weight +
      static_cast<int64_t>(n64) * kK4Tiles * kWaveSize * 4;
#pragma unroll
  for (int local_k4 = 0; local_k4 < kTilesPerSplit; ++local_k4) {
    const int32_t b = reinterpret_cast<const int32_t*>(
        packed_base + static_cast<int64_t>(first_k4 + local_k4) * kWaveSize * 4)[lane];
    acc = dot4_i32_i8(a_lds[local_k4], b, acc);
  }
  partial[static_cast<int64_t>(split) * kHy3N + column] = acc;

  uint32_t ticket = 0;
  if (lane == 0) {
    ticket = __hip_atomic_fetch_add(tickets + n64, 1u, __ATOMIC_ACQ_REL,
                                    __HIP_MEMORY_SCOPE_AGENT);
  }
  ticket = __builtin_amdgcn_readfirstlane(ticket);
  if (ticket != kSplitK - 1) {
    return;
  }

  int32_t sum = 0;
#pragma unroll
  for (int s = 0; s < kSplitK; ++s) {
    sum += partial[static_cast<int64_t>(s) * kHy3N + column];
  }
  output[column] = hip_bfloat16(static_cast<float>(sum) * activation_scale[0] *
                                weight_scale[column]);
  if (lane == 0) {
    tickets[n64] = 0;
  }
}

// Split-K=16 专用变体：每个 CTA 只从 global 读取一次 256 B A slice，
// 再将八个 MMAC A fragment 常驻 VGPR，主循环只搬运 packed B。
__global__ __launch_bounds__(kWaveSize, 4)
void hy3_w8a8_m1k4096n384_splitk16_aresident_kernel(
    const int8_t* __restrict__ activation,
    const int8_t* __restrict__ packed_weight,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    int32_t* __restrict__ partial,
    uint32_t* __restrict__ tickets,
    hip_bfloat16* __restrict__ output) {
  constexpr int kSplitK = 16;
  constexpr int kTilesPerSplit = kK32Tiles / kSplitK;
  const int lane = threadIdx.x;
  const int n16 = blockIdx.x;
  const int split = blockIdx.y;
  const int row16 = lane & 15;
  const int k_group = lane >> 4;
  const int first_kt = split * kTilesPerSplit;
  __shared__ __align__(16) int8_t a_lds[kTilesPerSplit * 32];

  reinterpret_cast<int32_t*>(a_lds)[lane] =
      reinterpret_cast<const int32_t*>(activation + first_kt * 32)[lane];
  __syncthreads();

  int2_t a_fragments[kTilesPerSplit];
#pragma unroll
  for (int local_kt = 0; local_kt < kTilesPerSplit; ++local_kt) {
    a_fragments[local_kt] = load_int8x8(a_lds + local_kt * 32 + k_group * 8);
  }

  int4_t accumulator{};
  const int8_t* const packed_base = packed_weight +
      static_cast<int64_t>(n16) * kK32Tiles * kWaveSize * kPackedBytesPerLane;
#pragma unroll
  for (int local_kt = 0; local_kt < kTilesPerSplit; ++local_kt) {
    const int kt = first_kt + local_kt;
    const int2_t b = load_int8x8(
        packed_base + static_cast<int64_t>(kt) * kWaveSize * kPackedBytesPerLane +
        lane * kPackedBytesPerLane);
    accumulator = mmac_i32_16x16x32_i8(a_fragments[local_kt], b, accumulator);
  }

  if (row16 == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int column = n16 * 16 + k_group + i * 4;
      partial[static_cast<int64_t>(split) * kHy3N + column] = accumulator[i];
    }
  }

  uint32_t ticket = 0;
  if (lane == 0) {
    ticket = __hip_atomic_fetch_add(tickets + n16, 1u, __ATOMIC_ACQ_REL,
                                    __HIP_MEMORY_SCOPE_AGENT);
  }
  ticket = __builtin_amdgcn_readfirstlane(ticket);
  if (ticket != kSplitK - 1 || row16 != 0) {
    return;
  }

  const float a_scale = activation_scale[0];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = n16 * 16 + k_group + i * 4;
    int32_t sum = 0;
#pragma unroll
    for (int s = 0; s < kSplitK; ++s) {
      sum += partial[static_cast<int64_t>(s) * kHy3N + column];
    }
    output[column] = hip_bfloat16(static_cast<float>(sum) * a_scale *
                                  weight_scale[column]);
  }
  if (lane == 0) {
    tickets[n16] = 0;
  }
}

void check_hy3_w8a8_inputs(const at::Tensor& activation_q,
                           const at::Tensor& packed_weight,
                           const at::Tensor& activation_scale,
                           const at::Tensor& weight_scale) {
  TORCH_CHECK(activation_q.is_cuda() && packed_weight.is_cuda() &&
                  activation_scale.is_cuda() && weight_scale.is_cuda(),
              "HY3 W8A8 asm kernel requires HIP tensors");
  TORCH_CHECK(activation_q.scalar_type() == at::kChar,
              "HY3 W8A8 asm kernel requires int8 activations");
  TORCH_CHECK(packed_weight.scalar_type() == at::kChar,
              "HY3 W8A8 asm kernel requires int8 packed weights");
  TORCH_CHECK(activation_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "HY3 W8A8 asm kernel requires float32 scales");
  TORCH_CHECK(activation_q.is_contiguous() && packed_weight.is_contiguous() &&
                  activation_scale.is_contiguous() && weight_scale.is_contiguous(),
              "HY3 W8A8 asm kernel requires contiguous tensors");
  TORCH_CHECK(activation_q.sizes() == at::IntArrayRef({kHy3M, kHy3K}),
              "HY3 W8A8 asm kernel requires activation [1, 4096]");
  TORCH_CHECK(packed_weight.sizes() ==
                  at::IntArrayRef({kN16Tiles, kK32Tiles, kWaveSize,
                                    kPackedBytesPerLane}),
              "HY3 W8A8 asm kernel requires packed weight [24, 128, 64, 8]");
  TORCH_CHECK(activation_scale.numel() == 1,
              "HY3 W8A8 asm kernel requires one activation scale");
  TORCH_CHECK(weight_scale.numel() == kHy3N,
              "HY3 W8A8 asm kernel requires 384 channel scales");
}

template <int SplitK>
torch::Tensor run_hy3_w8a8_scaled_mm_splitk(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  check_hy3_w8a8_inputs(activation_q, packed_weight, activation_scale, weight_scale);
  at::cuda::OptionalCUDAGuard device_guard(device_of(activation_q));
  auto output = at::empty({kHy3M, kHy3N}, activation_q.options().dtype(at::kBFloat16));
  auto partial = at::empty({SplitK, kHy3N}, activation_q.options().dtype(at::kInt));
  auto tickets = at::zeros({kN16Tiles}, activation_q.options().dtype(at::kInt));
  hipLaunchKernelGGL((hy3_w8a8_m1k4096n384_splitk_kernel<SplitK>),
                     dim3(kN16Tiles, SplitK), dim3(kWaveSize), 0,
                     at::cuda::getCurrentCUDAStream(),
                     activation_q.data_ptr<int8_t>(), packed_weight.data_ptr<int8_t>(),
                     activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
                     partial.data_ptr<int32_t>(),
                     reinterpret_cast<uint32_t*>(tickets.data_ptr<int32_t>()),
                     reinterpret_cast<hip_bfloat16*>(output.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor hy3_w8a8_pack(const at::Tensor& weight_kn) {
  TORCH_CHECK(weight_kn.is_cuda() && weight_kn.scalar_type() == at::kChar &&
                  weight_kn.is_contiguous(),
              "HY3 W8A8 pack requires contiguous CUDA/HIP int8 weight");
  TORCH_CHECK(weight_kn.sizes() == at::IntArrayRef({kHy3K, kHy3N}),
              "HY3 W8A8 pack requires weight [4096, 384]");

  at::cuda::OptionalCUDAGuard device_guard(device_of(weight_kn));
  auto packed = at::empty({kN16Tiles, kK32Tiles, kWaveSize, kPackedBytesPerLane},
                          weight_kn.options());
  constexpr int threads = 256;
  constexpr int64_t elements =
      static_cast<int64_t>(kN16Tiles) * kK32Tiles * kWaveSize * kPackedBytesPerLane;
  const dim3 blocks((elements + threads - 1) / threads);
  hipLaunchKernelGGL(hy3_w8a8_pack_kernel, blocks, dim3(threads), 0,
                     at::cuda::getCurrentCUDAStream(), weight_kn.data_ptr<int8_t>(),
                     packed.data_ptr<int8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

torch::Tensor hy3_w8a8_dot_pack(const at::Tensor& weight_kn) {
  TORCH_CHECK(weight_kn.is_cuda() && weight_kn.scalar_type() == at::kChar &&
                  weight_kn.is_contiguous() &&
                  weight_kn.sizes() == at::IntArrayRef({kHy3K, kHy3N}),
              "HY3 W8A8 dot pack requires contiguous int8 weight [4096, 384]");
  at::cuda::OptionalCUDAGuard device_guard(device_of(weight_kn));
  constexpr int kN64Tiles = kHy3N / kWaveSize;
  constexpr int kK4Tiles = kHy3K / 4;
  auto packed = at::empty({kN64Tiles, kK4Tiles, kWaveSize, 4}, weight_kn.options());
  constexpr int threads = 256;
  constexpr int64_t elements =
      static_cast<int64_t>(kN64Tiles) * kK4Tiles * kWaveSize * 4;
  hipLaunchKernelGGL(hy3_w8a8_dot_pack_kernel, dim3((elements + threads - 1) / threads),
                     dim3(threads), 0, at::cuda::getCurrentCUDAStream(),
                     weight_kn.data_ptr<int8_t>(), packed.data_ptr<int8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

torch::Tensor hy3_w8a8_scaled_mm(const at::Tensor& activation_q,
                                  const at::Tensor& packed_weight,
                                  const at::Tensor& activation_scale,
                                  const at::Tensor& weight_scale) {
  check_hy3_w8a8_inputs(activation_q, packed_weight, activation_scale, weight_scale);
  at::cuda::OptionalCUDAGuard device_guard(device_of(activation_q));
  auto output = at::empty({kHy3M, kHy3N}, activation_q.options().dtype(at::kBFloat16));
  hipLaunchKernelGGL(hy3_w8a8_m1k4096n384_kernel, dim3(kN16Tiles), dim3(kWaveSize),
                     0, at::cuda::getCurrentCUDAStream(),
                     activation_q.data_ptr<int8_t>(), packed_weight.data_ptr<int8_t>(),
                     activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
                     reinterpret_cast<hip_bfloat16*>(output.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor hy3_w8a8_dot_scaled_mm_splitk32(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  constexpr int kSplitK = 32;
  constexpr int kN64Tiles = kHy3N / kWaveSize;
  constexpr int kK4Tiles = kHy3K / 4;
  TORCH_CHECK(activation_q.is_cuda() && packed_weight.is_cuda() &&
                  activation_scale.is_cuda() && weight_scale.is_cuda() &&
                  activation_q.scalar_type() == at::kChar &&
                  packed_weight.scalar_type() == at::kChar &&
                  activation_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "HY3 W8A8 dot kernel requires CUDA/HIP int8 inputs and float32 scales");
  TORCH_CHECK(activation_q.is_contiguous() && packed_weight.is_contiguous() &&
                  activation_scale.is_contiguous() && weight_scale.is_contiguous() &&
                  activation_q.sizes() == at::IntArrayRef({kHy3M, kHy3K}) &&
                  packed_weight.sizes() == at::IntArrayRef({kN64Tiles, kK4Tiles,
                                                            kWaveSize, 4}) &&
                  activation_scale.numel() == 1 && weight_scale.numel() == kHy3N,
              "HY3 W8A8 dot kernel received an invalid fixed-shape ABI");
  at::cuda::OptionalCUDAGuard device_guard(device_of(activation_q));
  auto output = at::empty({kHy3M, kHy3N}, activation_q.options().dtype(at::kBFloat16));
  auto partial = at::empty({kSplitK, kHy3N}, activation_q.options().dtype(at::kInt));
  auto tickets = at::zeros({kN64Tiles}, activation_q.options().dtype(at::kInt));
  hipLaunchKernelGGL(hy3_w8a8_m1k4096n384_dot_splitk32_kernel,
                     dim3(kN64Tiles, kSplitK), dim3(kWaveSize), 0,
                     at::cuda::getCurrentCUDAStream(),
                     activation_q.data_ptr<int8_t>(), packed_weight.data_ptr<int8_t>(),
                     activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
                     partial.data_ptr<int32_t>(),
                     reinterpret_cast<uint32_t*>(tickets.data_ptr<int32_t>()),
                     reinterpret_cast<hip_bfloat16*>(output.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor hy3_w8a8_scaled_mm_splitk4(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  return run_hy3_w8a8_scaled_mm_splitk<4>(activation_q, packed_weight,
                                            activation_scale, weight_scale);
}

torch::Tensor hy3_w8a8_scaled_mm_splitk8(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  return run_hy3_w8a8_scaled_mm_splitk<8>(activation_q, packed_weight,
                                            activation_scale, weight_scale);
}

torch::Tensor hy3_w8a8_scaled_mm_splitk16(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  return run_hy3_w8a8_scaled_mm_splitk<16>(activation_q, packed_weight,
                                             activation_scale, weight_scale);
}

void hy3_w8a8_scaled_mm_splitk16_aresident_out(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale,
    at::Tensor& partial, at::Tensor& tickets, at::Tensor& output) {
  constexpr int kSplitK = 16;
  check_hy3_w8a8_inputs(activation_q, packed_weight, activation_scale, weight_scale);
  TORCH_CHECK(partial.is_cuda() && tickets.is_cuda() && output.is_cuda(),
              "HY3 W8A8 asm workspace requires HIP tensors");
  TORCH_CHECK(partial.scalar_type() == at::kInt && tickets.scalar_type() == at::kInt &&
                  output.scalar_type() == at::kBFloat16,
              "HY3 W8A8 asm workspace dtypes must be int32, int32, bf16");
  TORCH_CHECK(partial.is_contiguous() && tickets.is_contiguous() && output.is_contiguous(),
              "HY3 W8A8 asm workspace requires contiguous tensors");
  TORCH_CHECK(partial.sizes() == at::IntArrayRef({kSplitK, kHy3N}) &&
                  tickets.sizes() == at::IntArrayRef({kN16Tiles}) &&
                  output.sizes() == at::IntArrayRef({kHy3M, kHy3N}),
              "HY3 W8A8 asm workspace has an invalid shape");
  at::cuda::OptionalCUDAGuard device_guard(device_of(activation_q));
  hipLaunchKernelGGL(hy3_w8a8_m1k4096n384_splitk16_aresident_kernel,
                     dim3(kN16Tiles, kSplitK), dim3(kWaveSize), 0,
                     at::cuda::getCurrentCUDAStream(),
                     activation_q.data_ptr<int8_t>(), packed_weight.data_ptr<int8_t>(),
                     activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
                     partial.data_ptr<int32_t>(),
                     reinterpret_cast<uint32_t*>(tickets.data_ptr<int32_t>()),
                     reinterpret_cast<hip_bfloat16*>(output.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hy3_w8a8_scaled_mm_splitk16_aresident(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale) {
  constexpr int kSplitK = 16;
  check_hy3_w8a8_inputs(activation_q, packed_weight, activation_scale, weight_scale);
  auto output = at::empty({kHy3M, kHy3N}, activation_q.options().dtype(at::kBFloat16));
  auto partial = at::empty({kSplitK, kHy3N}, activation_q.options().dtype(at::kInt));
  auto tickets = at::zeros({kN16Tiles}, activation_q.options().dtype(at::kInt));
  hy3_w8a8_scaled_mm_splitk16_aresident_out(
      activation_q, packed_weight, activation_scale, weight_scale, partial, tickets, output);
  return output;
}
