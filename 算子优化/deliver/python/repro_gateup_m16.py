#!/usr/bin/env python3
"""验证 shared_gate_up_proj_m16 的 max_diff=13.8 是否为 bf16 输出精度限制。

假设：gate_up 某些 decode 激活值极大（输出值 ~1700-3500），此时 bf16
输出 1 ULP ≈ 0.39% 的绝对误差 > 1.0，validate 的 gt1 阈值被放大触发，
而非 kernel 逻辑 bug。

方法：固定权重 + 不同输入量级（×3/×30/×300/×1000），对比
  - zth kernel 输出
  - torch fp64 精确参考（int8*int8->int32 累加精确）
观察 hip_vs_torch_max / 输出量级 是否恒定在 ~0.4%（1 ULP bf16）。
"""
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault(
    "ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache")
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

NAME = "shared_gate_up_proj_m16"
M, N, K = 16, 384, 4096


def torch_ref(x_q, w_q, x_s, w_s):
    acc = x_q.to(torch.float64) @ w_q.to(torch.float64)
    return (acc.to(torch.float32) * x_s * w_s.t()).to(torch.bfloat16)


def triton_ref(x_q, w_q, x_s, w_s):
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )
    return triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s.view(-1),
                            out_dtype=torch.bfloat16, bias=None)


def main():
    import zth_w8a8_ext
    failed = zth_w8a8_ext.load_all(verbose=False, include_standalone_only=True)
    assert not failed, f"failed to load: {failed}"
    from vllm import _custom_ops as vops

    ops = getattr(torch.ops, f"zth_{NAME}")
    torch.manual_seed(0)

    # 真实范围权重：int8 全动态 + 实际 scale 量级
    w_q = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    w_s = (torch.rand(N, 1, device="cuda") * 0.5 + 0.05).float()
    packed, pscale = ops.pack_weight(w_q.contiguous(), w_s.contiguous())
    ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def compare(tag, x_q, x_s, w_q, w_s, packed, pscale, ws):
        out = ops.gemm_out(x_q, packed, x_s, pscale, ws)
        torch.cuda.synchronize()
        ref = torch_ref(x_q, w_q, x_s, w_s)
        triton = triton_ref(x_q, w_q, x_s, w_s)
        d_t = (out.float() - ref.float()).abs()
        d_tri_t = (triton.float() - ref.float()).abs()
        d_h_tri = (out.float() - triton.float()).abs()
        out_mean = out.float().abs().mean().item()
        out_max = out.float().abs().max().item()
        # 原始 int32 累加和量级（除以激活/权重量化 scale 还原）
        raw = out.abs() / (x_s.view(-1, 1).float() * w_s.t().float())
        raw_max = raw.max().item()
        print(f"{tag:>6} {out_mean:>10.1f} {out_max:>10.1f} "
              f"{d_t.max().item():>12.4f} {d_tri_t.max().item():>12.4f} "
              f"{d_h_tri.max().item():>12.4f} "
              f"{(d_t>1).sum().item():>6}  raw_max={raw_max:>12.0f}")

    print(f"{NAME} (m={M}, n={N}, k={K})")
    print(f"{'set':>6} {'out_mean':>10} {'out_max':>10} "
          f"{'hip_vs_torch':>12} {'tri_vs_torch':>12} {'hip_vs_tri':>12} {'gt1':>6}")
    for scale in [3.0, 30.0, 300.0, 1000.0]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * scale
        x_q, x_s, _ = vops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True)
        compare(f"x{scale:.0f}", x_q, x_s, w_q, w_s, packed, pscale, ws)

    # 真实偏置型 gate_up：权重全正均值高 + 激活单侧，原始部分和 > 2^24 (16.7M)，
    # 触发 Triton fp32 累加舍入；zth int32 累加仍精确。
    print("\n偏置型权重（部分和 > 2^24，触发 fp32 舍入）:")
    w_qb = torch.randint(60, 128, (K, N), dtype=torch.int8, device="cuda")
    w_sb = (torch.rand(N, 1, device="cuda") * 0.0004 + 0.0001).float()
    packed_b, pscale_b = ops.pack_weight(w_qb.contiguous(), w_sb.contiguous())
    for scale in [3.0, 30.0]:
        x = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5 + 8) * scale
        x_q, x_s, _ = vops.scaled_int8_quant(
            x.contiguous(), None, None, symmetric=True)
        compare(f"b{scale:.0f}", x_q, x_s, w_qb, w_sb, packed_b, pscale_b, ws)


if __name__ == "__main__":
    print("torch", torch.__version__)
    main()
