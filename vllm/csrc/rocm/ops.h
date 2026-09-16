#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor hy3_w8a8_pack(const at::Tensor& weight_kn);
torch::Tensor hy3_w8a8_dot_pack(const at::Tensor& weight_kn);

torch::Tensor hy3_w8a8_scaled_mm(const at::Tensor& activation_q,
                                  const at::Tensor& packed_weight,
                                  const at::Tensor& activation_scale,
                                  const at::Tensor& weight_scale);
torch::Tensor hy3_w8a8_dot_scaled_mm_splitk32(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale);

torch::Tensor hy3_w8a8_scaled_mm_splitk4(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale);

torch::Tensor hy3_w8a8_scaled_mm_splitk8(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale);

torch::Tensor hy3_w8a8_scaled_mm_splitk16(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale);

torch::Tensor hy3_w8a8_scaled_mm_splitk16_aresident(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale);

void hy3_w8a8_scaled_mm_splitk16_aresident_out(
    const at::Tensor& activation_q, const at::Tensor& packed_weight,
    const at::Tensor& activation_scale, const at::Tensor& weight_scale,
    at::Tensor& partial, at::Tensor& tickets, at::Tensor& output);

void hy3_mtp_paged_attention(
    torch::Tensor& out, const torch::Tensor& q,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables, double softmax_scale);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
