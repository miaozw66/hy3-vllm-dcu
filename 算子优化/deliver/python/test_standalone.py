#!/usr/bin/env python3
"""Standalone numerical correctness for all 8 TP8 shapes.

Compares the HIP kernels against (a) triton_scaled_mm (the actual vLLM
original implementation) and (b) an exact int32 torch reference, using
ops.scaled_int8_quant for activations exactly like the vLLM path.

Usage (container):  python3 /home/hy3-TP8/entry-vllm/test_standalone.py
"""
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("ZTH_W8A8_CACHE", "/home/hy3-TP8/ext_cache")

import torch

sys.path.insert(0, "/home/hy3-TP8/entry-vllm")

NAMES = {
    "o_proj_m16": (16, 4096, 1024),
    "qkv_proj_m16": (16, 1280, 4096),
    "shared_down_proj_m16": (16, 4096, 192),
    "shared_gate_up_proj_m16": (16, 384, 4096),
    "o_proj_m4096": (4096, 4096, 1024),
    "qkv_proj_m4096": (4096, 1280, 4096),
    "shared_down_proj_m4096": (4096, 4096, 192),
    "shared_gate_up_proj_m4096": (4096, 384, 4096),
}


def triton_scaled_mm_ref(x_q, w_q, x_s, w_s, out_dtype):
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    return triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s,
                            out_dtype=out_dtype, bias=None)


def torch_ref(x_q, w_q, x_s, w_s):
    # float64 is EXACT for int8*int8 -> int32 accumulation
    acc = x_q.to(torch.float64) @ w_q.to(torch.float64)
    out = (acc.to(torch.float32) * x_s * w_s.t()).to(torch.bfloat16)
    return out


def main():
    torch.manual_seed(0)
    import zth_w8a8_ext
    failed = zth_w8a8_ext.load_all(verbose=True, include_standalone_only=True)
    assert not failed, f"failed to load: {failed}"

    from vllm import _custom_ops as vops
    results = []
    for name, (m, n, k) in NAMES.items():
        ops = getattr(torch.ops, f"zth_{name}")
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 3.0
        x_q, x_s, x_zp = vops.scaled_int8_quant(x.contiguous(), None, None,
                                                symmetric=True)
        assert x_zp is None
        w_q = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
        w_s = (torch.rand(n, 1, device="cuda") + 0.1).float()

        packed, pscale = ops.pack_weight(w_q.contiguous(), w_s.contiguous())
        ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        out = ops.gemm_out(x_q, packed, x_s, pscale, ws)
        torch.cuda.synchronize()

        ref_triton = triton_scaled_mm_ref(x_q, w_q, x_s, w_s.view(-1), x.dtype)
        ref_torch = torch_ref(x_q, w_q, x_s, w_s)

        d_t = (out.float() - ref_triton.float()).abs()
        d_r = (out.float() - ref_torch.float()).abs()
        tol = 1.0
        row = {
            "shape": (m, n, k),
            "hip_vs_triton_max": d_t.max().item(),
            "hip_vs_triton_mean": d_t.mean().item(),
            "hip_vs_triton_gt1": (d_t > tol).sum().item(),
            "hip_vs_torch_max": d_r.max().item(),
            "hip_vs_torch_mean": d_r.mean().item(),
            "hip_vs_torch_gt1": (d_r > tol).sum().item(),
        }
        results.append(row)
        print(row, flush=True)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
