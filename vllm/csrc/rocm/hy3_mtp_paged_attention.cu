#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include <torch/all.h>

#include <limits>

namespace {

constexpr int kQueryHeads = 8;
constexpr int kHeadSize = 128;
constexpr int kQueryLength = 4;
constexpr int kBlockSize = 16;
constexpr int kThreads = 128;

template <typename scalar_t>
__device__ float scalar_to_float(scalar_t value);

template <>
__device__ float scalar_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float scalar_to_float<__hip_bfloat16>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ scalar_t float_to_scalar(float value);

template <>
__device__ __half float_to_scalar<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __hip_bfloat16 float_to_scalar<__hip_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename scalar_t>
__global__ void hy3_mtp_paged_attention_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_tables,
    int total_query_tokens,
    int num_cache_blocks,
    int max_blocks_per_seq,
    float scale) {
  const int dim = threadIdx.x;
  const int query_head = blockIdx.z;
  const int query_pos = blockIdx.y;
  const int request_idx = blockIdx.x;
  const int query_start = query_start_loc[request_idx];
  const int query_end = query_start_loc[request_idx + 1];
  const int query_token = query_start + query_pos;
  const int sequence_length = seq_lens[request_idx];
  const int query_abs_pos = sequence_length - kQueryLength + query_pos;

  // All metadata values are device-owned. Reject malformed rows before using
  // them as tensor offsets so an invalid optional fast path cannot dereference
  // outside Q or paged KV storage.
  if (query_end - query_start != kQueryLength || query_token < 0
      || query_token >= total_query_tokens || sequence_length < kQueryLength
      || query_abs_pos >= max_blocks_per_seq * kBlockSize) {
    return;
  }

  __shared__ float dot_values[kThreads];
  __shared__ float block_max;
  __shared__ float block_sum;
  __shared__ float block_weight;

  const scalar_t* query_ptr = q + (query_token * kQueryHeads + query_head) * kHeadSize;
  const float query_value = scalar_to_float(query_ptr[dim]);

  float running_max = -std::numeric_limits<float>::infinity();
  float running_sum = 0.0f;
  float running_value = 0.0f;

  for (int kv_pos = 0; kv_pos <= query_abs_pos; ++kv_pos) {
    const int logical_block = kv_pos / kBlockSize;
    const int block_offset = kv_pos % kBlockSize;
    const int physical_block =
        block_tables[request_idx * max_blocks_per_seq + logical_block];
    if (physical_block < 0 || physical_block >= num_cache_blocks) {
      return;
    }
    const int cache_offset =
        ((physical_block * kBlockSize + block_offset) * kHeadSize) + dim;

    dot_values[dim] = query_value * scalar_to_float(key_cache[cache_offset]);
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
      if (dim < offset) {
        dot_values[dim] += dot_values[dim + offset];
      }
      __syncthreads();
    }

    if (dim == 0) {
      const float score = dot_values[0] * scale;
      const float next_max = fmaxf(running_max, score);
      const float prior_scale = expf(running_max - next_max);
      const float current_weight = expf(score - next_max);
      block_max = next_max;
      block_sum = running_sum * prior_scale + current_weight;
      block_weight = current_weight;
    }
    __syncthreads();

    const float prior_scale = expf(running_max - block_max);
    running_value = running_value * prior_scale
                    + block_weight * scalar_to_float(value_cache[cache_offset]);
    running_max = block_max;
    running_sum = block_sum;
    __syncthreads();
  }

  out[(query_token * kQueryHeads + query_head) * kHeadSize + dim] =
      float_to_scalar<scalar_t>(running_value / running_sum);
}

template <typename scalar_t>
void launch_hy3_mtp_paged_attention(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    double softmax_scale) {
  const int batch_size = seq_lens.size(0);
  const dim3 grid(batch_size, kQueryLength, kQueryHeads);
  const dim3 block(kThreads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  hipLaunchKernelGGL(
      hy3_mtp_paged_attention_kernel<scalar_t>, grid, block, 0, stream,
      static_cast<scalar_t*>(out.data_ptr()),
      static_cast<const scalar_t*>(q.data_ptr()),
      static_cast<const scalar_t*>(key_cache.data_ptr()),
      static_cast<const scalar_t*>(value_cache.data_ptr()),
      query_start_loc.data_ptr<int>(), seq_lens.data_ptr<int>(),
      block_tables.data_ptr<int>(), q.size(0), key_cache.size(0),
      block_tables.size(1), static_cast<float>(softmax_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void hy3_mtp_paged_attention(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda() && out.is_cuda() && key_cache.is_cuda()
                  && value_cache.is_cuda() && query_start_loc.is_cuda()
                  && seq_lens.is_cuda() && block_tables.is_cuda(),
              "HY3 MTP HIP attention requires ROCm tensors");
  TORCH_CHECK(q.device() == out.device() && q.device() == key_cache.device()
                  && q.device() == value_cache.device()
                  && q.device() == query_start_loc.device()
                  && q.device() == seq_lens.device()
                  && q.device() == block_tables.device(),
              "HY3 MTP HIP attention tensors must share a device");
  TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && key_cache.is_contiguous()
                  && value_cache.is_contiguous() && query_start_loc.is_contiguous()
                  && seq_lens.is_contiguous() && block_tables.is_contiguous(),
              "HY3 MTP HIP attention requires contiguous tensors");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kQueryHeads && q.size(2) == kHeadSize,
              "HY3 MTP HIP attention requires Q shape [tokens, 8, 128]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes()
                  && key_cache.size(1) == kBlockSize && key_cache.size(2) == 1
                  && key_cache.size(3) == kHeadSize,
              "HY3 MTP HIP attention requires NHD KV cache [blocks, 16, 1, 128]");
  TORCH_CHECK(query_start_loc.dim() == 1 && seq_lens.dim() == 1
                  && block_tables.dim() == 2,
              "HY3 MTP HIP attention metadata ranks must be [B+1], [B], [B, blocks]");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt
                  && block_tables.scalar_type() == at::kInt,
              "HY3 MTP HIP attention metadata must be int32");
  TORCH_CHECK(query_start_loc.numel() == seq_lens.numel() + 1
                  && block_tables.size(0) == seq_lens.numel(),
              "HY3 MTP HIP attention metadata must have B+1 prefixes and B rows");
  TORCH_CHECK(q.size(0) == seq_lens.numel() * kQueryLength
                  && out.sizes() == q.sizes(),
              "HY3 MTP HIP attention requires four query tokens per request");
  TORCH_CHECK(q.scalar_type() == key_cache.scalar_type()
                  && q.scalar_type() == value_cache.scalar_type()
                  && q.scalar_type() == out.scalar_type(),
              "HY3 MTP HIP attention requires matching Q, KV, and output dtypes");

  if (q.scalar_type() == at::kBFloat16) {
    launch_hy3_mtp_paged_attention<__hip_bfloat16>(
        out, q, key_cache, value_cache, query_start_loc, seq_lens, block_tables,
        softmax_scale);
  } else if (q.scalar_type() == at::kHalf) {
    launch_hy3_mtp_paged_attention<__half>(
        out, q, key_cache, value_cache, query_start_loc, seq_lens, block_tables,
        softmax_scale);
  } else {
    TORCH_CHECK(false, "HY3 MTP HIP attention supports only BF16 and FP16");
  }
}
