#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);

  rocm_ops.def("hy3_w8a8_pack(Tensor weight_kn) -> Tensor");
  rocm_ops.impl("hy3_w8a8_pack", torch::kCUDA, &hy3_w8a8_pack);
  rocm_ops.def("hy3_w8a8_dot_pack(Tensor weight_kn) -> Tensor");
  rocm_ops.impl("hy3_w8a8_dot_pack", torch::kCUDA, &hy3_w8a8_dot_pack);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm(Tensor activation_q, Tensor packed_weight, "
      "Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_scaled_mm", torch::kCUDA, &hy3_w8a8_scaled_mm);

  rocm_ops.def(
      "hy3_w8a8_dot_scaled_mm_splitk32(Tensor activation_q, Tensor packed_weight, "
      "Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_dot_scaled_mm_splitk32", torch::kCUDA,
                &hy3_w8a8_dot_scaled_mm_splitk32);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm_splitk4(Tensor activation_q, Tensor packed_weight, "
      "Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_scaled_mm_splitk4", torch::kCUDA,
                &hy3_w8a8_scaled_mm_splitk4);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm_splitk8(Tensor activation_q, Tensor packed_weight, "
      "Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_scaled_mm_splitk8", torch::kCUDA,
                &hy3_w8a8_scaled_mm_splitk8);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm_splitk16(Tensor activation_q, Tensor packed_weight, "
      "Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_scaled_mm_splitk16", torch::kCUDA,
                &hy3_w8a8_scaled_mm_splitk16);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm_splitk16_aresident(Tensor activation_q, "
      "Tensor packed_weight, Tensor activation_scale, Tensor weight_scale) -> Tensor");
  rocm_ops.impl("hy3_w8a8_scaled_mm_splitk16_aresident", torch::kCUDA,
                &hy3_w8a8_scaled_mm_splitk16_aresident);

  rocm_ops.def(
      "hy3_w8a8_scaled_mm_splitk16_aresident_out(Tensor activation_q, "
      "Tensor packed_weight, Tensor activation_scale, Tensor weight_scale, "
      "Tensor! partial, Tensor! tickets, Tensor! output) -> ()");
  rocm_ops.impl("hy3_w8a8_scaled_mm_splitk16_aresident_out", torch::kCUDA,
                &hy3_w8a8_scaled_mm_splitk16_aresident_out);

  rocm_ops.def(
      "hy3_mtp_paged_attention(Tensor! out, Tensor q, Tensor key_cache, "
      "Tensor value_cache, Tensor query_start_loc, Tensor seq_lens, "
      "Tensor block_tables, float softmax_scale) -> ()");
  rocm_ops.impl("hy3_mtp_paged_attention", torch::kCUDA,
                &hy3_mtp_paged_attention);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
