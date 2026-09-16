#!/usr/bin/env python3
"""Build one TP8 shape .hip into a torch extension exposing
torch.ops.zth_<name>.gemm_out / pack_weight.

Usage: python3 build_extension.py <name> <path-to.hip>
"""
import os
import sys
import torch
import torch.utils.cpp_extension as cpp

NAME = sys.argv[1]
HIPFILE = os.path.abspath(sys.argv[2])

# torch's load_inline builds via ninja; the .hip is pulled in through an
# #include in the CUDA_SRC string, and ninja's depfile does NOT reliably track
# that .hip file. So a changed .hip silently reuses the previous build. Fix:
# clear the build dir whenever the .hip source hash changes.
try:
    import hashlib
    import shutil
    try:
        from torch.utils.cpp_extension import get_build_directory
        _bdir = get_build_directory(f"zth_{NAME}", verbose=False)
    except ImportError:  # torch 2.10 renamed it private
        from torch.utils.cpp_extension import _get_build_directory
        _bdir = _get_build_directory(f"zth_{NAME}", verbose=False)
    _md5f = os.path.join(_bdir, ".zth_src_md5")
    _hasher = hashlib.md5()
    _hasher.update(open(HIPFILE, "rb").read())
    _hpp = os.path.join(os.path.dirname(HIPFILE), "w8a8_m32_smallm.hpp")
    if os.path.exists(_hpp):
        _hasher.update(open(_hpp, "rb").read())
    _cur = _hasher.hexdigest()
    if os.path.exists(_md5f):
        _old = open(_md5f).read().strip()
        if _old != _cur:
            shutil.rmtree(_bdir, ignore_errors=True)
            print(f"source changed ({_old[:8]} -> {_cur[:8]}); cleared build cache")
    os.makedirs(_bdir, exist_ok=True)
    with open(_md5f, "w") as _f:
        _f.write(_cur)
except Exception as _e:  # noqa: BLE001  (md5 gating is best-effort)
    print(f"md5 gate skipped: {_e}")

CPP_SRC = """
#include <torch/extension.h>
#include <tuple>
torch::Tensor gemm_out_impl(
    torch::Tensor x_q, torch::Tensor packed, torch::Tensor x_scale,
    torch::Tensor w_scale, torch::Tensor workspace);
std::tuple<torch::Tensor, torch::Tensor> pack_weight_impl(
    torch::Tensor raw, torch::Tensor w_scale);
"""

CUDA_SRC = f"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <tuple>
#include "{HIPFILE}"

torch::Tensor gemm_out_impl(
    torch::Tensor x_q, torch::Tensor packed, torch::Tensor x_scale,
    torch::Tensor w_scale, torch::Tensor workspace) {{
  const int m = (int)x_q.size(0);
  const int k = (int)x_q.size(1);
  const int n = (int)w_scale.size(0);
  auto out = torch::empty({{m, n}},
                          x_q.options().dtype(torch::kBFloat16));
  launch_w8a8_gemm(
      (const int8_t*)x_q.data_ptr(),
      (const int8_t*)packed.data_ptr(),
      (const float*)x_scale.data_ptr(),
      (const float*)w_scale.data_ptr(),
      (void*)out.data_ptr(),
      (void*)workspace.data_ptr(),
      (int64_t)workspace.numel(),
      m, n, k,
      at::cuda::getCurrentCUDAStream());
  return out;
}}

std::tuple<torch::Tensor, torch::Tensor> pack_weight_impl(
    torch::Tensor raw, torch::Tensor w_scale) {{
  const int k = (int)raw.size(0);
  const int n = (int)raw.size(1);
  auto packed = torch::empty({{k * n}}, raw.options());
  auto packed_scale = torch::empty({{n}}, w_scale.options());
  launch_pack_w8a8_weight(
      (const int8_t*)raw.data_ptr(),
      (const float*)w_scale.data_ptr(),
      (int8_t*)packed.data_ptr(),
      (float*)packed_scale.data_ptr(),
      k, n,
      at::cuda::getCurrentCUDAStream());
  return {{packed, packed_scale}};
}}

TORCH_LIBRARY(zth_{NAME}, m) {{
  m.def("gemm_out(Tensor x_q, Tensor packed, Tensor x_scale, Tensor w_scale, Tensor workspace) -> Tensor",
        torch::dispatch(c10::DispatchKey::CUDA, &gemm_out_impl));
  m.def("pack_weight(Tensor raw, Tensor w_scale) -> (Tensor, Tensor)",
        torch::dispatch(c10::DispatchKey::CUDA, &pack_weight_impl));
}}
"""

os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
os.environ["PYTORCH_ROCM_ARCH"] = "gfx928"  # single-arch: du_mma.h guard needs __gfx928__
os.environ["TORCH_ROCM_ARCH"] = "gfx928"
os.environ.setdefault("MAX_JOBS", "4")

mod = cpp.load_inline(
    name=f"zth_{NAME}",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=[],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx928", "-I/opt/dtk/include"],
    verbose=True,
)
print(f"BUILD_OK {NAME}")
