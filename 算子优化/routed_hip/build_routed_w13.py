#!/usr/bin/env python3
"""Build routed_w13.hip into a torch extension exposing
torch.ops.routed_w13.w1_pack(raw[E,N,K]) -> packed
torch.ops.routed_w13.out(qA, w1p, a_scale, w1_scale, tok_ids, exp_ids, out)

Usage: python3 build_routed_w13.py
"""
import os
import torch
import torch.utils.cpp_extension as cpp

HIPFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "routed_w13.hip")

try:
    import hashlib
    import shutil
    try:
        from torch.utils.cpp_extension import get_build_directory
        _bdir = get_build_directory("routed_w13", verbose=False)
    except ImportError:
        from torch.utils.cpp_extension import _get_build_directory
        _bdir = _get_build_directory("routed_w13", verbose=False)
    _md5f = os.path.join(_bdir, ".routed_w13_src_md5")
    _hasher = hashlib.md5()
    _hasher.update(open(HIPFILE, "rb").read())
    _cur = _hasher.hexdigest()
    if os.path.exists(_md5f):
        _old = open(_md5f).read().strip()
        if _old != _cur:
            shutil.rmtree(_bdir, ignore_errors=True)
            print(f"source changed ({_old[:8]} -> {_cur[:8]}); cleared build cache")
    os.makedirs(_bdir, exist_ok=True)
    with open(_md5f, "w") as _f:
        _f.write(_cur)
except Exception as _e:  # noqa: BLE001
    print(f"md5 gate skipped: {_e}")

CPP_SRC = """
#include <torch/extension.h>
"""

CUDA_SRC = f"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include "{HIPFILE}"

void check_routed_w13_launch(const char* operation) {{
  const hipError_t status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess, operation, " launch failed: ",
              hipGetErrorString(status));
}}

torch::Tensor w1_pack(torch::Tensor raw) {{
  TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kChar &&
              raw.is_contiguous() && raw.dim() == 3,
              "w1_pack needs contiguous int8 [E,N,K]");
  const int E = (int)raw.size(0);
  const int N = (int)raw.size(1);
  const int K = (int)raw.size(2);
  TORCH_CHECK(K % 32 == 0 && N % 16 == 0, "pack needs K%32==0, N%16==0");
  auto packed = torch::empty({{E, N / 16, K / 32, 64, 8}}, raw.options());
  launch_pack_routed_w1((const int8_t*)raw.data_ptr(),
                        (int8_t*)packed.data_ptr(), E, N, K,
                        at::cuda::getCurrentCUDAStream());
  check_routed_w13_launch("w1_pack");
  return packed;
}}

torch::Tensor w1_unpack(torch::Tensor packed, int64_t E, int64_t N, int64_t K) {{
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == torch::kChar &&
              packed.is_contiguous(), "w1_unpack needs contiguous HIP int8");
  auto raw = torch::empty({{E, N, K}}, packed.options());
  launch_unpack_routed_w1((const int8_t*)packed.data_ptr(),
                          (int8_t*)raw.data_ptr(), (int)E, (int)N, (int)K,
                          at::cuda::getCurrentCUDAStream());
  return raw;
}}

void routed_w13_out(
    torch::Tensor qA, torch::Tensor w1p, torch::Tensor a_scale,
    torch::Tensor w1_scale, torch::Tensor tok_ids, torch::Tensor exp_ids,
    torch::Tensor out) {{
  TORCH_CHECK(qA.is_cuda() && w1p.is_cuda() && a_scale.is_cuda() &&
              w1_scale.is_cuda() && tok_ids.is_cuda() && exp_ids.is_cuda() &&
              out.is_cuda(), "routed_w13 requires HIP tensors");
  TORCH_CHECK(qA.scalar_type() == torch::kChar &&
              w1p.scalar_type() == torch::kChar,
              "routed_w13 requires int8 qA and w1p");
  TORCH_CHECK(a_scale.scalar_type() == torch::kFloat &&
              w1_scale.scalar_type() == torch::kFloat,
              "routed_w13 requires fp32 scales");
  TORCH_CHECK(qA.is_contiguous() && w1p.is_contiguous() &&
              a_scale.is_contiguous() && w1_scale.is_contiguous() &&
              tok_ids.is_contiguous() && exp_ids.is_contiguous() &&
              out.is_contiguous(), "routed_w13 requires contiguous tensors");
  TORCH_CHECK(qA.get_device() == w1p.get_device() &&
              qA.get_device() == a_scale.get_device() &&
              qA.get_device() == w1_scale.get_device() &&
              qA.get_device() == tok_ids.get_device() &&
              qA.get_device() == exp_ids.get_device() &&
              qA.get_device() == out.get_device(),
              "routed_w13 tensors must share a device");
  TORCH_CHECK(qA.dim() == 2 && w1p.dim() == 5 && a_scale.dim() == 1 &&
              w1_scale.dim() == 2 && tok_ids.dim() == 1 &&
              exp_ids.dim() == 1 && out.dim() == 2,
              "routed_w13 tensor ranks");
  const int K = (int)qA.size(1);
  const int E = (int)w1_scale.size(0);
  const int N = (int)w1_scale.size(1);
  const int P = (int)out.size(0);
  TORCH_CHECK(K == 4096 && N == 384 && E == 192,
              "routed_w13 supports HY3 w13 [192,384,4096] only");
  TORCH_CHECK(w1p.sizes() == torch::IntArrayRef({{E, N / 16, K / 32, 64, 8}}),
              "routed_w13 packed w1 shape mismatch");
  TORCH_CHECK(a_scale.numel() == qA.size(0), "routed_w13 input scale shape");
  TORCH_CHECK(K % 32 == 0 && N % 16 == 0, "routed_w13 requires K%32==0, N%16==0");
  TORCH_CHECK(w1_scale.sizes() == torch::IntArrayRef({{E, N}}),
              "routed_w13 scale shape");
  TORCH_CHECK(tok_ids.scalar_type() == torch::kInt &&
              exp_ids.scalar_type() == torch::kInt,
              "routed_w13 pair ids must be int32");
  TORCH_CHECK(tok_ids.size(0) == P && exp_ids.size(0) == P, "pair list mismatch");
  TORCH_CHECK(out.size(1) == N && out.scalar_type() == torch::kBFloat16,
              "routed_w13 out must be [P, N] bf16");
  launch_routed_w13(
      (const int8_t*)qA.data_ptr(), (const int8_t*)w1p.data_ptr(),
      (const float*)a_scale.data_ptr(), (const float*)w1_scale.data_ptr(),
      (const int32_t*)tok_ids.data_ptr(), (const int32_t*)exp_ids.data_ptr(),
      (void*)out.data_ptr(), P, K, N, at::cuda::getCurrentCUDAStream());
  check_routed_w13_launch("routed_w13");
}}

void act_out(torch::Tensor w13, torch::Tensor qact, torch::Tensor a2) {{
  launch_silu_requant((const void*)w13.data_ptr(), (int8_t*)qact.data_ptr(),
                      (float*)a2.data_ptr(), (int)w13.size(0),
                      at::cuda::getCurrentCUDAStream());
}}

void w2_out(torch::Tensor qact, torch::Tensor w2p, torch::Tensor a2,
            torch::Tensor w2s, torch::Tensor exp, torch::Tensor weights,
            torch::Tensor pairout) {{
  launch_routed_w2((const int8_t*)qact.data_ptr(), (const int8_t*)w2p.data_ptr(),
                   (const float*)a2.data_ptr(), (const float*)w2s.data_ptr(),
                   (const int32_t*)exp.data_ptr(), (const float*)weights.data_ptr(),
                   (void*)pairout.data_ptr(), (int)qact.size(0),
                   (int)qact.size(1), (int)w2s.size(1),
                   at::cuda::getCurrentCUDAStream());
}}

void sum_out(torch::Tensor pairout, torch::Tensor out, int64_t topk) {{
  launch_routed_sum((const void*)pairout.data_ptr(), (void*)out.data_ptr(),
                    (int)out.size(0), (int)topk, (int)out.size(1),
                    at::cuda::getCurrentCUDAStream());
}}

void routed_full_out(
    torch::Tensor qA, torch::Tensor w1p, torch::Tensor w2p,
    torch::Tensor a1_scale, torch::Tensor w1_scale, torch::Tensor w2_scale,
    torch::Tensor tok_ids, torch::Tensor exp_ids, torch::Tensor topk_weights,
    torch::Tensor w13_out, torch::Tensor qact, torch::Tensor a2_scale,
    torch::Tensor pair_out, torch::Tensor out) {{
  const int M = (int)qA.size(0);
  const int K1 = (int)qA.size(1);
  const int P = (int)tok_ids.numel();
  const int N1 = (int)w1_scale.size(1);
  const int K2 = (int)qact.size(1);
  const int N2 = (int)w2_scale.size(1);
  const int topk = P / M;
  TORCH_CHECK(qA.scalar_type() == torch::kChar && qact.scalar_type() == torch::kChar,
              "full requires int8 qA/qact");
  TORCH_CHECK(tok_ids.scalar_type() == torch::kInt &&
              exp_ids.scalar_type() == torch::kInt,
              "full requires int32 pair ids");
  TORCH_CHECK(w13_out.sizes() == torch::IntArrayRef({{P, N1}}) &&
              qact.sizes() == torch::IntArrayRef({{P, K2}}) &&
              a2_scale.numel() == P && pair_out.sizes() == torch::IntArrayRef({{P, N2}}) &&
              out.sizes() == torch::IntArrayRef({{M, N2}}), "full workspace shape mismatch");
  TORCH_CHECK(N1 == 384 && K2 == 192 && N2 == 4096 && topk == 8,
              "full prototype only supports HY3 TP8 shape");
  launch_routed_full(
      (const int8_t*)qA.data_ptr(), (const int8_t*)w1p.data_ptr(),
      (const int8_t*)w2p.data_ptr(), (const float*)a1_scale.data_ptr(),
      (const float*)w1_scale.data_ptr(), (const float*)w2_scale.data_ptr(),
      (const int32_t*)tok_ids.data_ptr(), (const int32_t*)exp_ids.data_ptr(),
      (const float*)topk_weights.data_ptr(), (void*)w13_out.data_ptr(),
      (int8_t*)qact.data_ptr(), (float*)a2_scale.data_ptr(),
      (void*)pair_out.data_ptr(), (void*)out.data_ptr(), M, P, K1, N1, K2, N2,
      topk, at::cuda::getCurrentCUDAStream());
}}

TORCH_LIBRARY(routed_w13, m) {{
  m.def("w1_pack(Tensor raw) -> Tensor",
        torch::dispatch(c10::DispatchKey::CUDA, &w1_pack));
  m.def("w1_unpack(Tensor packed, int E, int N, int K) -> Tensor",
        torch::dispatch(c10::DispatchKey::CUDA, &w1_unpack));
  m.def("out(Tensor qA, Tensor w1p, Tensor a_scale, Tensor w1_scale, "
        "Tensor tok_ids, Tensor exp_ids, Tensor out) -> ()",
        torch::dispatch(c10::DispatchKey::CUDA, &routed_w13_out));
  m.def("act_out(Tensor w13, Tensor qact, Tensor a2) -> ()",
        torch::dispatch(c10::DispatchKey::CUDA, &act_out));
  m.def("w2_out(Tensor qact, Tensor w2p, Tensor a2, Tensor w2s, Tensor exp, "
        "Tensor weights, Tensor pairout) -> ()",
        torch::dispatch(c10::DispatchKey::CUDA, &w2_out));
  m.def("sum_out(Tensor pairout, Tensor out, int topk) -> ()",
        torch::dispatch(c10::DispatchKey::CUDA, &sum_out));
  m.def("full_out(Tensor qA, Tensor w1p, Tensor w2p, Tensor a1_scale, "
        "Tensor w1_scale, Tensor w2_scale, Tensor tok_ids, Tensor exp_ids, "
        "Tensor topk_weights, Tensor w13_out, Tensor qact, Tensor a2_scale, "
        "Tensor pair_out, Tensor out) -> ()",
        torch::dispatch(c10::DispatchKey::CUDA, &routed_full_out));
}}
"""

os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
os.environ["PYTORCH_ROCM_ARCH"] = "gfx928"
os.environ["TORCH_ROCM_ARCH"] = "gfx928"
os.environ.setdefault("MAX_JOBS", "4")

mod = cpp.load_inline(
    name="routed_w13",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=[],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx928", "-I/opt/dtk/include"],
    verbose=True,
)
print(f"BUILD_OK routed_w13 {mod.__file__}")
