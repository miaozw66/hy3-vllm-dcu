#include <torch/extension.h>

void hy3_mtp_paged_attention(
    torch::Tensor& out, const torch::Tensor& q,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables, double softmax_scale);

// A private namespace prevents this JIT-only validation module from colliding
// with vLLM's production _rocm_C registration.
TORCH_LIBRARY_FRAGMENT(hy3_mtp_jit_validation, m) {
  m.def("paged_attention(Tensor! out, Tensor q, Tensor key_cache, "
        "Tensor value_cache, Tensor query_start_loc, Tensor seq_lens, "
        "Tensor block_tables, float softmax_scale) -> ()");
}

TORCH_LIBRARY_IMPL(hy3_mtp_jit_validation, CUDA, m) {
  m.impl("paged_attention", &hy3_mtp_paged_attention);
}
