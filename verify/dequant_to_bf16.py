#!/usr/bin/env python3
"""把 INT8 w8a8 (compressed-tensors) 权重反量化为 BF16，供对照测试。
用法: dequant_to_bf16.py <源模型目录> <层数> <输出目录>"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1]
NUM_LAYERS = int(sys.argv[2])
DST = sys.argv[3]
os.makedirs(DST, exist_ok=True)
keep_prefixes = ("model.embed_tokens.", "model.norm.", "lm_head.")

index_path = os.path.join(SRC, "model.safetensors.index.json")
if os.path.exists(index_path):
    idx = json.load(open(index_path))
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    need = set()
    for k, s in idx["weight_map"].items():
        m = layer_re.match(k)
        if (m and int(m.group(1)) < NUM_LAYERS) or k.startswith(keep_prefixes):
            need.add(s)
    shards = sorted(need)
    print(f"按 index 筛选：处理 {len(shards)} 个分片", flush=True)
else:
    shards = sorted(
        f for f in os.listdir(SRC) if re.match(r"model-\d+-of-\d+\.safetensors$", f)
    )

quant_keys = re.compile(
    r"^(model\.layers\.\d+\.(self_attn\.qkv_proj|self_attn\.o_proj|"
    r"mlp\.shared_mlp\.|mlp\.experts\.\d+\.)(gate_proj|up_proj|down_proj))\.weight$"
)

target_layer = re.compile(r"^model\.layers\.(\d+)\.")

result = {}
for shard in shards:
    path = os.path.join(SRC, shard)
    print(f"处理 {shard} ...", flush=True)
    with safe_open(path, framework="pt") as sf:
        for name in sf.keys():
            keep = False
            m = target_layer.match(name)
            if m and int(m.group(1)) < NUM_LAYERS:
                keep = True
            elif any(name.startswith(p) for p in keep_prefixes):
                keep = True
            if not keep:
                continue

            scale_name = name + "_scale"
            if quant_keys.match(name) and sf.get_slice(scale_name).get_shape() is not None:
                w = sf.get_tensor(name).to(torch.float32)
                s = sf.get_tensor(scale_name).to(torch.float32)
                # scale 形状 [out,1] 或 [1,in]，与权重广播对齐
                if s.dim() == 2 and s.shape[-1] == 1:
                    s = s.squeeze(-1)
                if s.dim() == 2 and s.shape[0] == 1:
                    s = s.squeeze(0)
                while s.dim() < w.dim():
                    s = s.unsqueeze(-1)
                w = w * s
                result[name] = w.to(torch.bfloat16)
                print(f"  反量化 {name} {tuple(w.shape)}", flush=True)
            elif sf.get_slice(name).get_dtype() in ("I8", "U8", "i8", "u8"):
                # 未匹配到 scale 的 INT 权重（理论不应出现），直接转
                result[name] = sf.get_tensor(name).to(torch.bfloat16)
                print(f"  !! 无 scale 的 INT 权重: {name}", flush=True)
            else:
                result[name] = sf.get_tensor(name)

save_file(result, os.path.join(DST, "model.safetensors"))
print(f"保存 {len(result)} 个张量到 {DST}/model.safetensors")

cfg = json.load(open(os.path.join(SRC, "config.json")))
cfg["num_hidden_layers"] = NUM_LAYERS
cfg.pop("compression_config", None)
with open(os.path.join(DST, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("config.json 已写入（去除压缩配置，num_hidden_layers=%d）" % NUM_LAYERS)
