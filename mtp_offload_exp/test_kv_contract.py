#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eager_attention KV 契约单测（评审验证计划：小型随机张量验证三段契约）：
1. past=None 时返回 == 当前步 KV（prefill 恒等）
2. past 非空时返回 == cat(past, 当前步)（修复正确，pre-GQA 形状）
3. q_len=1 时掩码跳过，输出与手算 softmax 一致
4. GQA 场景：返回 KV 保持 Hkv 头数（pre-GQA 形状，未被 repeat 污染）
"""
import sys

import torch

sys.path.insert(0, ".")
from hy3_mtp_offload_infer import eager_attention

dev = "cpu"
torch.manual_seed(42)
ok = True


def check(name, cond, detail=""):
    global ok
    mark = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{mark}] {name} {detail}")


# ---- 1. past=None 恒等 ----
q = torch.randn(1, 4, 3, 8)
k = torch.randn(1, 2, 3, 8)
v = torch.randn(1, 2, 3, 8)
out, fk, fv = eager_attention(q, k, v)
check("past=None 返回 == 当前步 KV", torch.equal(fk, k) and torch.equal(fv, v),
      f"fk.shape={tuple(fk.shape)}")

# ---- 2. past 非空 == cat(past, 当前步) ----
past_k = torch.randn(1, 2, 5, 8)
past_v = torch.randn(1, 2, 5, 8)
out2, fk2, fv2 = eager_attention(q, k, v, past_k, past_v)
check("past 非空返回 == cat(past, 当前步)", torch.equal(fk2, torch.cat([past_k, k], dim=2))
      and torch.equal(fv2, torch.cat([past_v, v], dim=2)),
      f"fk2.shape={tuple(fk2.shape)} (期望 (1,2,8,8))")

# ---- 3. q_len=1 掩码跳过 + 手算 softmax ----
q1 = torch.randn(1, 4, 1, 8)
k3 = torch.randn(1, 2, 3, 8)
v3 = torch.randn(1, 2, 3, 8)
out3, _, _ = eager_attention(q1, k3, v3, scale=0.5)
# 手算：GQA repeat + 无掩码 softmax + 加权
k3r = k3.repeat_interleave(2, dim=1)
v3r = v3.repeat_interleave(2, dim=1)
scores = torch.matmul(q1, k3r.transpose(-1, -2)) * 0.5
probs = torch.softmax(scores, dim=-1)
ref = torch.matmul(probs, v3r)
check("q_len=1 输出 == 手算 softmax", torch.allclose(out3, ref, atol=1e-6),
      f"maxdiff={torch.max(torch.abs(out3 - ref)).item():.2e}")

# ---- 4. q_len>1 因果掩码与返回 KV 形状 ----
q4 = torch.randn(1, 4, 3, 8)
k4 = torch.randn(1, 2, 4, 8)
v4 = torch.randn(1, 2, 4, 8)
out4, fk4, fv4 = eager_attention(q4, k4, v4, scale=1.0)
check("q_len>1 返回 KV pre-GQA 头数", fk4.shape[1] == 2, f"fk4.shape={tuple(fk4.shape)}")

# ---- 5. 掩码正确性：query i 只可见 key <= i ----
q5 = torch.randn(1, 4, 2, 8)
k5 = torch.randn(1, 2, 2, 8)
v5 = torch.randn(1, 2, 2, 8)
out5, _, _ = eager_attention(q5, k5, v5)
k5r = k5.repeat_interleave(2, dim=1)
v5r = v5.repeat_interleave(2, dim=1)
scores5 = torch.matmul(q5, k5r.transpose(-1, -2))
mask5 = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
scores5 = scores5.masked_fill(mask5, float("-inf"))
probs5 = torch.softmax(scores5, dim=-1)
ref5 = torch.matmul(probs5, v5r)
check("q_len>1 掩码正确（query i 只见 key<=i）", torch.allclose(out5, ref5, atol=1e-6),
      f"maxdiff={torch.max(torch.abs(out5 - ref5)).item():.2e}")

# ---- 6. 确定性：两次调用逐位一致 ----
out6a, fk6a, fv6a = eager_attention(q, k, v, past_k, past_v)
out6b, fk6b, fv6b = eager_attention(q, k, v, past_k, past_v)
check("确定性（两次调用逐位一致）", torch.equal(out6a, out6b) and torch.equal(fk6a, fk6b))

print()
print("全部通过" if ok else "存在失败项")
sys.exit(0 if ok else 1)
