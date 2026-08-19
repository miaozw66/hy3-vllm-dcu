#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""黄金 KV 正确性校验（评审验证计划）：增量生成循环 == 全序列单趟重算。

对 [prompt, t1, t2, ..., tN] 做一次 80 层单趟前向（prefill 路径），逐位置
argmax 与增量循环产出的 token 序列逐位比对。修复后必须完全一致；修复前
（KV 覆盖 bug）token3 起注意力只剩单 key，必出垃圾。

用法：
  python3 verify_golden_kv.py <gen_token1,gen_token2,...> [--reuse-cache 无]
     例：python3 verify_golden_kv.py 5002,292,185

主判据 = 逐位置 argmax token 序列一致（对低裕度 ulp 翻转鲁棒）；不一致时
输出该位置 logits top-2 差值与 max-abs-diff 供人工判定（评审 2 issue 1）。
"""
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy3_mtp_offload_infer import (  # noqa: E402
    DEVICE,
    MODEL_DIR,
    PROMPT,
    WeightStore,
    build_cfg,
    decoder_layer_forward,
    load_layer_weights,
    log,
    rms_norm,
)

def main():
    if len(sys.argv) < 2:
        print("用法: python3 verify_golden_kv.py <token1,token2,...>")
        sys.exit(2)
    gen = [int(x) for x in sys.argv[1].split(",")]

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    cfg = build_cfg(config)
    store = WeightStore(MODEL_DIR)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)

    prompt_ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)[0]
    N = prompt_ids.shape[0]
    full_ids = torch.cat([prompt_ids, torch.tensor(gen, device=DEVICE)])
    M = full_ids.shape[0]
    log(f"全序列 {M} tokens = prompt {N} + gen {len(gen)}")

    embed_cpu = store.get("model.embed_tokens.weight")
    h = embed_cpu[full_ids.cpu()].to(DEVICE)
    positions = torch.arange(M, device=DEVICE)
    norm_w = store.get("model.norm.weight", DEVICE)

    t0 = time.time()
    for i in range(config["num_hidden_layers"]):
        w = load_layer_weights(store, i, cfg, config)
        h, _, _ = decoder_layer_forward(h, w, positions, None, store)
        for k2 in list(w.keys()):
            del w[k2]
        del w
        gc.collect()
        if (i + 1) % 20 == 0:
            log(f"层 {i+1}/80 完成 ({time.time()-t0:.1f}s)")
    h = rms_norm(h, norm_w, cfg["eps"])

    lm_head_cpu = store.get("lm_head.weight")
    lm_head_gpu = lm_head_cpu.to(torch.float32).to(DEVICE)
    logits = F.linear(h.to(torch.float32), lm_head_gpu)  # (M, 120832)
    golden = [int(logits[j].argmax().item()) for j in range(N, M)]
    log(f"单趟重算 argmax 序列: {golden}")

    # 自回归语义：位置 N+j 的输入是 gen[j]，其 argmax 预测 gen[j+1]
    # （增量循环中 round j+1 的 verify 恰在此位置验证 gen[j+1]）；
    # 序列末位（输入 gen[-1]）是「下一 token」预测，无增量对照。
    ok = True
    for j, g in enumerate(golden):
        pos = N + j
        if j < len(gen) - 1:
            r = gen[j + 1]
            if g == r:
                log(f"位置 {pos} (输入 gen[{j}]={gen[j]}): 增量 {r} == 重算 {g} ✓")
            else:
                ok = False
                top2 = torch.topk(logits[pos], 2)
                log(f"位置 {pos} (输入 gen[{j}]={gen[j]}): 增量 {r} != 重算 {g} ✗ "
                    f"(top1={top2.values[0].item():.3f}@{top2.indices[0].item()}, "
                    f"top2={top2.values[1].item():.3f}@{top2.indices[1].item()}, "
                    f"top1-top2={top2.values[0].item()-top2.values[1].item():.3f})")
        else:
            log(f"位置 {pos} (输入 gen[-1]={gen[-1]}): 续接预测 {g}（无增量对照）")
    log(f"黄金校验: {'通过（增量 == 全序列重算）' if ok else '存在不一致（见上）'}")
    store.close_all()
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
