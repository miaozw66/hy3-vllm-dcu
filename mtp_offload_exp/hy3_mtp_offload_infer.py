#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HY3 (Hy3-Channel-INT8-w8a8) 显存受限逐层加载推理 + MTP 投机解码脚本
================================================================
目标：在 K100AI DCU（单卡空闲显存仅 ~9GB）上，用「逐层加载 - 计算 - 立即释放」
的方式运行 80 层 MoE 模型（总权重约 280GB INT8），对 prompt「中国的首都是」
做带 MTP（Multi-Token Prediction）的多 token 生成，temperature=0 保证可复现。
默认 2 个 token（与 run3 [5002, 292] -> '北京。' 逐位可比），可用环境变量
HY3_NUM_GEN_TOKENS 配置（生成 token 总数，含 token1；9 对应在线链2 的 mt=9）。

关键设计：
1. 逐层 offload：每层权重按需从 safetensors 分片读取 -> 计算 -> 立即释放，
   显存峰值仅 ~1.5GB/层（远小于 9GB 预算）。
2. MoE 专家按需加载：router 每 token 选 top-8 专家（192 个中），只加载被
   选中的专家权重（~8×19MB），而不是整层 192 专家（~3.6GB）。
3. INT8 channel-wise 反量化（compressed-tensors 格式）：
   W_fp = W_int8.to(dtype) * scale，其中 scale 形状 (out_features, 1)。
4. MTP（num_nextn_predict_layers=1）：model.layers.80（enorm/hnorm/eh_proj +
   完整 decoder 块 + final_layernorm），语义参照 vllm hy_v3_mtp.py 的
   norm-residual 结构：
       a   = eh_proj(cat([enorm(E(t_i)), hnorm(h_i)]))
       h1  = input_layernorm(a)
       hA  = attention(h1)
       h2  = post_attention_layernorm(hA + a)      # norm-residual
       out = final_layernorm(h2 + moe(h2))
   位置 i 输入 [h_i, embed(t_i)] 同位置对齐 -> 预测 t_{i+1}（Patch E 对齐语义）。
5. 投机解码循环：每轮主模型 verify 得 gen[k]（KV 续接），MTP 单步
   （输入 [h_{N+k-1}, embed(gen[k-1])]，past = drafter_kv）得 draft_k，
   统计命中率。前向函数返回完整 KV（含 past），kv_cache 恒存完整历史。
6. embed/lm_head CPU 常驻，每轮仅转移所需行/瞬态副本；mtp_w 常驻 GPU。

依赖：torch, safetensors, transformers(AutoTokenizer)。
用法：python3 hy3_mtp_offload_infer.py   # 或 HY3_NUM_GEN_TOKENS=9 ...
"""

import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

MODEL_DIR = os.environ.get(
    "HY3_MODEL_DIR",
    "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master",
)
DEVICE = os.environ.get("HY3_DEVICE", "cuda:0")
PROMPT = "中国的首都是"
# 生成 token 总数（含 token1）；默认 2 与 run3 [5002,292]='北京。' 逐位可比，
# 9 对应在线链2 的 mt=9（mt = max_tokens，非 prompt 长度）
NUM_GEN_TOKENS = max(1, int(os.environ.get("HY3_NUM_GEN_TOKENS", "2")))
if NUM_GEN_TOKENS > 2000:
    print(f"[hy3-offload] 警告: HY3_NUM_GEN_TOKENS={NUM_GEN_TOKENS}，每 token ~235s，"
          f"预计 {NUM_GEN_TOKENS * 235 / 3600:.1f} 小时，显存预算内 KV 上限约 8700 token")
VERBOSE = os.environ.get("HY3_VERBOSE", "1") == "1"

DENSE_F32 = True   # 反量化到 fp32 计算（精度优先，朴素 fallback）
COMPUTE_DTYPE = torch.float32 if DENSE_F32 else torch.bfloat16


def log(msg):
    print(f"[hy3-offload] {msg}", flush=True)


def vram_mb():
    try:
        return torch.cuda.memory_allocated(DEVICE) / 2**20
    except Exception:
        return -1.0


def free_gpu(*objs):
    for o in objs:
        if o is not None:
            del o
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 权重加载：按 key 从 safetensors 分片随机读取（逐层按需）
# ---------------------------------------------------------------------------
class WeightStore:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.index = json.load(f)
        self.wm = self.index["weight_map"]
        self._open = {}

    def _handle(self, shard):
        if shard not in self._open:
            from safetensors import safe_open

            self._open[shard] = safe_open(
                os.path.join(self.model_dir, shard), framework="pt", device="cpu"
            )
        return self._open[shard]

    def close_all(self):
        for h in self._open.values():
            try:
                h.close()
            except Exception:
                pass
        self._open.clear()

    def has(self, name):
        return name in self.wm

    def get(self, name, device="cpu", dtype=None):
        """读取单个张量（CPU），可选转 dtype/device。"""
        h = self._handle(self.wm[name])
        t = h.get_tensor(name)
        if dtype is not None:
            t = t.to(dtype)
        if device != "cpu":
            t = t.to(device)
        return t


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def rms_norm(x, weight, eps=1e-5):
    """RMSNorm：x / sqrt(mean(x^2)+eps) * weight"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x * weight.to(torch.float32)).to(orig_dtype)


def qlinear(x, w_int8, w_scale, compute_dtype=COMPUTE_DTYPE):
    """INT8 per-channel 反量化线性：y = x @ (W_int8 * scale)^T
    scale 形状 (out, 1)，W_int8 形状 (out, in)。
    """
    w = w_int8.to(compute_dtype) * w_scale.to(compute_dtype)
    return F.linear(x.to(compute_dtype), w)


def rotary_embed(q, k, positions, theta=11158840.0, head_dim=128):
    """RoPE（default 类型，attention_scaling=1.0）"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    inv_freq = inv_freq.to(q.device)
    # positions: (N,) -> freqs: (N, head_dim/2)
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # (N, head_dim)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1,1,N,head_dim)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
    k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]
    q_rot = torch.cat((-q2, q1), dim=-1)
    k_rot = torch.cat((-k2, k1), dim=-1)
    q = (q * cos + q_rot * sin).to(q.dtype)
    k = (k * cos + k_rot * sin).to(k.dtype)
    return q, k


def eager_attention(q, k, v, past_k=None, past_v=None, scale=None):
    """朴素 attention（fallback 实现）。
    q: (B, H_q, N, D), k/v: (B, H_kv, N, D)
    返回 (attn_out, full_k, full_v)：
      attn_out (B, H_q, N, D)；
      full_k/full_v = cat 后的完整 KV（含 past），pre-GQA 形状
      (B, H_kv, N_total, D)，post-qk_norm/post-RoPE——即「kv_cache 恒存
      完整 KV」的前向契约，调用方存储返回值即存完整历史（past=None 时
      cat 为恒等，返回即当前步 KV）。"""
    B, Hq, N, D = q.shape
    Hkv = k.shape[1]
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    full_k, full_v = k, v  # 在 GQA repeat 之前捕获（repeat 会重绑形状）
    if Hkv < Hq:
        n_rep = Hq // Hkv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2))
    if scale is not None:
        scores = scores * scale
    # causal mask（q 序列内的每个 query 只看自己及之前的 key；past 部分全可见）
    q_len = q.shape[2]
    k_len = k.shape[2]
    if q_len > 1:
        mask = torch.triu(
            torch.ones(q_len, k_len, dtype=torch.bool, device=q.device), diagonal=k_len - q_len + 1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out, full_k, full_v


def silu(x):
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# 层前向（权重字典 w 已在 GPU 上，用完即由调用方释放）
# ---------------------------------------------------------------------------
def attention_forward(h, w, positions, past_k=None, past_v=None):
    """单层 attention。h: (N, 4096)。返回 (attn_out (N, 4096), full_k, full_v)，
    full_k/full_v 为含 past 的完整 KV（契约见 eager_attention）。"""
    N = h.shape[0]
    cfg = w["_cfg"]
    q = qlinear(h, w["q_proj"], w["q_proj_scale"])            # (N, 8192)
    k = qlinear(h, w["k_proj"], w["k_proj_scale"])            # (N, 1024)
    v = qlinear(h, w["v_proj"], w["v_proj_scale"])            # (N, 1024)
    q = q.view(N, cfg["n_heads"], cfg["head_dim"]).transpose(0, 1).unsqueeze(0)
    k = k.view(N, cfg["n_kv_heads"], cfg["head_dim"]).transpose(0, 1).unsqueeze(0)
    v = v.view(N, cfg["n_kv_heads"], cfg["head_dim"]).transpose(0, 1).unsqueeze(0)
    q = rms_norm(q, w["q_norm"], cfg["eps"])                  # qk_norm
    k = rms_norm(k, w["k_norm"], cfg["eps"])
    q, k = rotary_embed(q, k, positions, theta=cfg["rope_theta"], head_dim=cfg["head_dim"])
    attn_out, full_k, full_v = eager_attention(
        q, k, v, past_k, past_v, scale=cfg["head_dim"] ** -0.5
    )  # (B, Hq, N, D)
    attn_out = attn_out.transpose(1, 2).reshape(N, -1).contiguous()
    out = qlinear(attn_out, w["o_proj"], w["o_proj_scale"])
    return out, full_k, full_v


def moe_forward(h, w, store, layer_idx):
    """MoE 前向：router 选 top-8 专家，只加载被选中的专家权重。
    w 需含 router.gate(bf16)、expert_bias(fp32)、shared_mlp 权重、scale。
    """
    N = h.shape[0]
    cfg = w["_cfg"]
    # ---- router ----
    x = h.to(torch.float32)
    router_logits = F.linear(x, w["router_gate"].to(torch.float32))  # (N, 192)
    routing_weights = torch.sigmoid(router_logits)
    scores = routing_weights + w["expert_bias"].to(torch.float32)
    topk_idx = torch.topk(scores, cfg["top_k"], dim=-1, sorted=False).indices  # (N, 8)
    topk_weights = routing_weights.gather(1, topk_idx)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * cfg["router_scaling_factor"]

    routed = torch.zeros_like(h, dtype=torch.float32)
    # ---- 被选专家集合（按需加载，而非整层 192 专家）----
    chosen = torch.unique(topk_idx.reshape(-1)).tolist()
    # 每个 token 的 (专家, topk 位置, token 索引)
    flat_idx = topk_idx.reshape(-1)                    # (N*8,)
    token_ids = torch.arange(N, device=h.device).repeat_interleave(cfg["top_k"])
    for eid in chosen:
        mask = flat_idx == eid
        if not mask.any():
            continue
        tok_ids = token_ids[mask]
        wpos = (mask.nonzero(as_tuple=False).squeeze(-1) % cfg["top_k"])
        ew = load_expert_weights(store, layer_idx, eid)
        x_e = x[tok_ids]                               # (M, 4096) fp32
        gate = qlinear(x_e, ew["gate_proj"], ew["gate_proj_scale"])
        up = qlinear(x_e, ew["up_proj"], ew["up_proj_scale"])
        inter = silu(gate) * up
        down = qlinear(inter, ew["down_proj"], ew["down_proj_scale"])
        wgt = topk_weights[tok_ids, wpos].unsqueeze(-1)
        routed.index_add_(0, tok_ids, down * wgt)
        free_gpu(gate, up, inter, down, x_e, ew)

    # ---- shared expert ----
    shared = qlinear(x, w["shared_gate_proj"], w["shared_gate_proj_scale"])
    shared = qlinear(silu(shared) * qlinear(x, w["shared_up_proj"], w["shared_up_proj_scale"]),
                     w["shared_down_proj"], w["shared_down_proj_scale"])
    out = (routed + shared).to(h.dtype)
    free_gpu(routed, shared, x)
    return out


def dense_ffn_forward(h, w):
    x = h.to(COMPUTE_DTYPE)
    gate = qlinear(x, w["mlp_gate_proj"], w["mlp_gate_proj_scale"])
    up = qlinear(x, w["mlp_up_proj"], w["mlp_up_proj_scale"])
    down = qlinear(silu(gate) * up, w["mlp_down_proj"], w["mlp_down_proj_scale"])
    return down.to(h.dtype)


def decoder_layer_forward(h, w, positions, past_kv, store):
    """标准 transformer 层（主模型语义）：
    h1 = norm(x); hA = attn(h1); h2 = hA + x; out = mlp(norm(h2)) + h2
    返回 (out, new_k, new_v)——new_k/new_v 为含 past 的完整 KV
    （契约在 eager_attention，kv_cache 存储返回值即完整历史）。"""
    x = h
    layer_idx = w["_cfg"]["layer_idx"]
    h1 = rms_norm(x, w["input_layernorm"], w["_cfg"]["eps"])
    attn_out, new_k, new_v = attention_forward(h1, w, positions,
                                               past_kv[0] if past_kv else None,
                                               past_kv[1] if past_kv else None)
    h2 = attn_out + x
    if w["_cfg"]["is_dense"]:
        mlp_out = dense_ffn_forward(rms_norm(h2, w["post_layernorm"], w["_cfg"]["eps"]), w)
    else:
        mlp_out = moe_forward(rms_norm(h2, w["post_layernorm"], w["_cfg"]["eps"]), w, store, layer_idx)
    out = mlp_out + h2
    return out, new_k, new_v


# ---------------------------------------------------------------------------
# 权重组装
# ---------------------------------------------------------------------------
def build_cfg(config):
    return {
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "eps": config["rms_norm_eps"],
        "rope_theta": config["rope_parameters"]["rope_theta"],
        "top_k": config["num_experts_per_tok"],
        "router_scaling_factor": config["router_scaling_factor"],
    }


def load_layer_weights(store, layer_idx, cfg, config):
    """加载第 layer_idx 层的全部权重到 GPU。返回 dict 或 None（纯 dense/MoE 判定）。"""
    L = f"model.layers.{layer_idx}"
    w = {"_cfg": dict(cfg)}
    is_dense = layer_idx < config["first_k_dense_replace"]
    w["_cfg"]["is_dense"] = is_dense
    w["_cfg"]["layer_idx"] = layer_idx
    w["input_layernorm"] = store.get(f"{L}.input_layernorm.weight", DEVICE)
    w["post_layernorm"] = store.get(f"{L}.post_attention_layernorm.weight", DEVICE)
    # attention
    for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        w[p] = store.get(f"{L}.self_attn.{p}.weight", DEVICE)
        w[f"{p}_scale"] = store.get(f"{L}.self_attn.{p}.weight_scale", DEVICE)
    w["q_norm"] = store.get(f"{L}.self_attn.q_norm.weight", DEVICE)
    w["k_norm"] = store.get(f"{L}.self_attn.k_norm.weight", DEVICE)
    if is_dense:
        for p in ["gate_proj", "up_proj", "down_proj"]:
            w[f"mlp_{p}"] = store.get(f"{L}.mlp.{p}.weight", DEVICE)
            w[f"mlp_{p}_scale"] = store.get(f"{L}.mlp.{p}.weight_scale", DEVICE)
    else:
        w["router_gate"] = store.get(f"{L}.mlp.router.gate.weight", DEVICE)
        w["expert_bias"] = store.get(f"{L}.mlp.expert_bias", DEVICE)
        for p in ["gate_proj", "up_proj", "down_proj"]:
            w[f"shared_{p}"] = store.get(f"{L}.mlp.shared_mlp.{p}.weight", DEVICE)
            w[f"shared_{p}_scale"] = store.get(f"{L}.mlp.shared_mlp.{p}.weight_scale", DEVICE)
    return w


def load_expert_weights(store, layer_idx, eid):
    """按需加载单个专家权重（int8 + scale）。"""
    L = f"model.layers.{layer_idx}"
    w = {}
    for p in ["gate_proj", "up_proj", "down_proj"]:
        w[p] = store.get(f"{L}.mlp.experts.{eid}.{p}.weight", DEVICE)
        w[f"{p}_scale"] = store.get(f"{L}.mlp.experts.{eid}.{p}.weight_scale", DEVICE)
    return w


def load_mtp_weights(store, layer_idx, cfg, config):
    """加载 MTP 层（layers.80）全部权重：MTP 头 + 完整 decoder 块 + MoE。"""
    L = f"model.layers.{layer_idx}"
    w = load_layer_weights(store, layer_idx, cfg, config)  # 含 input_layernorm/attn/mlp/...
    w["enorm"] = store.get(f"{L}.enorm.weight", DEVICE)
    w["hnorm"] = store.get(f"{L}.hnorm.weight", DEVICE)
    w["eh_proj"] = store.get(f"{L}.eh_proj.weight", DEVICE)  # bf16 (4096, 8192)
    w["final_layernorm"] = store.get(f"{L}.final_layernorm.weight", DEVICE)
    return w


def mtp_layer_forward(h_in, prev_hidden, w, positions, store, past_kv=None):
    """MTP 层 forward（norm-residual 语义，来自 vllm hy_v3_mtp.py 判别实验结论）：
    训练语义（vllm eagle.py mtp 分支注释）：位置 i 输入 [h_i, embed(t_i)]
    （同位置对齐）→ 预测 t_{i+1}。逐位置：
      e = enorm(E(t_i))；p = hnorm(h_i)          # h_i 为主模型 final-norm 后 hidden
      a = eh_proj(cat([e, p], dim=-1))
      h1 = input_layernorm(a)
      hA = attention(h1)
      h2 = post_attention_layernorm(hA + a)     # norm-residual（非 Llama 残差）
      out = final_layernorm(h2 + moe(h2))
    位置 0 的 inputs_embeds 掩码为 0（vllm hy_v3_mtp.py L193-194 训练语义）。
    返回 (out, new_k, new_v)：new_k/new_v 为含 past 的完整 drafter KV
    （prefill 轮 past=None 时即全序列 KV，供调用方初始化 drafter_kv）。
    """
    cfg = w["_cfg"]
    e = rms_norm(h_in, w["enorm"], cfg["eps"])
    p = rms_norm(prev_hidden, w["hnorm"], cfg["eps"])
    # cat 前显式统一 dtype，消除 torch 版本相关的隐式类型提升依赖
    a = F.linear(torch.cat([e.to(COMPUTE_DTYPE), p.to(COMPUTE_DTYPE)], dim=-1),
                 w["eh_proj"].to(COMPUTE_DTYPE))
    a = a.to(torch.float32)
    h1 = rms_norm(a, w["input_layernorm"], cfg["eps"])
    pk = past_kv[0] if past_kv else None
    pv = past_kv[1] if past_kv else None
    attn_out, new_k, new_v = attention_forward(h1, w, positions, pk, pv)
    h2 = rms_norm(attn_out + a, w["post_layernorm"], cfg["eps"])
    if w["_cfg"]["is_dense"]:
        mlp_out = dense_ffn_forward(h2, w)
    else:
        mlp_out = moe_forward(h2, w, store, w["_cfg"]["layer_idx"])
    out = h2 + mlp_out
    return rms_norm(out, w["final_layernorm"], cfg["eps"]), new_k, new_v


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log(f"模型目录: {MODEL_DIR}")
    log(f"设备: {DEVICE}，空闲显存 {torch.cuda.mem_get_info(DEVICE)[0] / 2**30:.1f} GB")

    # ---- 加载 config / tokenizer ----
    from transformers import AutoTokenizer

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    log(f"模型: {config['architectures'][0]}, 层数 {config['num_hidden_layers']}, "
        f"MoE {config['num_experts']} experts, MTP layers {config['num_nextn_predict_layers']}")
    cfg = build_cfg(config)
    store = WeightStore(MODEL_DIR)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    input_ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)[0]
    N = input_ids.shape[0]
    log(f"prompt: {PROMPT!r} -> {N} tokens: {input_ids.tolist()}")

    # ---- 1. 逐层 prefill ----
    # embed/lm_head CPU 常驻（各 ~990MB 主机内存），每轮仅转移所需行/瞬态副本
    embed_cpu = store.get("model.embed_tokens.weight")  # bf16 (120832, 4096) CPU
    lm_head_cpu = store.get("lm_head.weight")           # bf16 (120832, 4096) CPU
    h = embed_cpu[input_ids.cpu()].to(DEVICE)           # (N, 4096)
    positions = torch.arange(N, device=DEVICE)
    kv_cache = {}  # layer_idx -> (k, v)，保留在显存（极小），恒存完整 KV

    for i in range(config["num_hidden_layers"]):
        t1 = time.time()
        w = load_layer_weights(store, i, cfg, config)
        h, new_k, new_v = decoder_layer_forward(h, w, positions, kv_cache.get(i), store)
        kv_cache[i] = (new_k, new_v)
        for k2 in list(w.keys()):
            del w[k2]
        del w
        gc.collect()
        if VERBOSE:
            log(f"层 {i:2d} 完成 ({time.time()-t1:.2f}s, 显存 {vram_mb():.0f} MB)")

    torch.cuda.empty_cache()  # 段末统一清理（层内仅 del，避免逐层同步开销）
    norm_w = store.get("model.norm.weight", DEVICE)  # 16KB，GPU 常驻供每轮复用
    h = rms_norm(h, norm_w, cfg["eps"])              # H_seq：每位置 final-norm 后 hidden

    # ---- 2. lm_head -> token1 ----
    lm_head_gpu = lm_head_cpu.to(torch.float32).to(DEVICE)  # 每轮瞬态转移
    logits1 = F.linear(h[-1].to(torch.float32), lm_head_gpu)  # (120832,)
    token1 = int(logits1.argmax().item())
    free_gpu(logits1)
    log(f"主模型 token1: {token1} = {tok.decode([token1])!r}")

    # ---- 3. MTP draft（投机解码语义，参照 vllm eagle.py mtp 分支）----
    # MTP 训练语义：位置 i 输入 [h_i, embed(t_i)]（同位置）→ 预测 t_{i+1}。
    # prefill 轮 MTP 输入 = 整个 prompt 序列 + 主模型 prefill hidden（final norm 后），
    # 最后位置（位置 N-1）的输出经共享 lm_head 得到 token1（t_N）的候选 draft。
    # 该轮前向同时产出 drafter KV（全序列，覆盖位置 0..N-1），作为后续
    # decode 轮单步 draft 的历史（采样位置输出因果依赖 KV<=i）。
    t1 = time.time()
    mtp_w = load_mtp_weights(store, config["num_hidden_layers"], cfg, config)  # 常驻 ~200MB
    emb_seq = embed_cpu[input_ids.cpu()].to(DEVICE)  # (N, 4096) prompt tokens
    emb_seq = emb_seq.clone()
    emb_seq[positions == 0] = 0                   # 位置 0 掩码（vllm hy_v3_mtp.py 训练语义）
    h_mtp, dk_k, dk_v = mtp_layer_forward(emb_seq, h, mtp_w, positions, store)
    drafter_kv = {config["num_hidden_layers"]: (dk_k, dk_v)}  # 位置 0..N-1 全 prompt KV
    free_gpu(emb_seq)
    logits_draft = F.linear(h_mtp[-1].to(torch.float32), lm_head_gpu)
    draft0 = int(logits_draft.argmax().item())
    free_gpu(h_mtp, logits_draft, h)
    drafts = [draft0]
    mtp_hit = draft0 == token1
    hits = [mtp_hit]
    log(f"MTP   draft(token1 候选): {draft0} = {tok.decode([draft0])!r} "
        f"| 与主模型 token1({token1}) 一致: {mtp_hit} ({time.time()-t1:.2f}s)")
    gen = [token1]

    # ---- 4. 生成主循环：每轮 verify 得 gen[k]，MTP 单步 draft 对照 ----
    # 裁决语义：round k（pos = N+k-1）先 verify 得 gen[k] 与 h_{N+k-1}
    # （final-norm 后 hidden），再以单位置 MTP（输入 [h_{N+k-1}, embed(gen[k-1])]，
    # past = drafter_kv）得 draft_k——h 与 emb 严格同位（先 draft 只能取上一轮
    # h，即 Patch E 错位模式）。位置严格连续递增（N, N+1, ...）。
    eos_id = config.get("eos_token_id")
    t_verify = 0.0
    t_mtp = 0.0
    peak_vram = vram_mb()
    for k in range(1, NUM_GEN_TOKENS):
        pos = N + k - 1
        t1 = time.time()
        emb_row = embed_cpu[[gen[-1]]].to(DEVICE)
        emb_keep = emb_row.clone()  # 80 层循环内 h_in 被覆盖，draft 输入须提前保留
        h_in = emb_row
        pos_t = torch.tensor([pos], device=DEVICE)
        for i in range(config["num_hidden_layers"]):
            w = load_layer_weights(store, i, cfg, config)
            h_in, new_k, new_v = decoder_layer_forward(h_in, w, pos_t, kv_cache.get(i), store)
            if new_k.shape[2] != N + k:  # KV 形状守卫：恒存完整历史
                raise RuntimeError(
                    f"KV 长度错误 layer {i}: {new_k.shape[2]} != {N + k}（覆盖回归）")
            kv_cache[i] = (new_k, new_v)
            for k2 in list(w.keys()):
                del w[k2]
            del w
            gc.collect()
        h_v = rms_norm(h_in, norm_w, cfg["eps"])  # final-norm 后 hidden（MTP 的 h_i）
        free_gpu(h_in)
        logits_k = F.linear(h_v.to(torch.float32), lm_head_gpu)
        gen_k = int(logits_k.argmax().item())
        gen.append(gen_k)
        t_verify += time.time() - t1
        log(f"主模型 token{k+1}: {gen_k} = {tok.decode([gen_k])!r} (pos {pos}, "
            f"{time.time()-t1:.2f}s)")
        if eos_id is not None and gen_k == eos_id:
            log(f"EOS 命中（{eos_id}），提前停止于第 {k+1} 个 token")
            break
        t2 = time.time()
        out_row, dk_k, dk_v = mtp_layer_forward(
            emb_keep, h_v, mtp_w, pos_t, store,
            drafter_kv[config["num_hidden_layers"]])
        if dk_k.shape[2] != N + k:  # drafter KV 形状守卫（同主 KV 契约）
            raise RuntimeError(
                f"drafter KV 长度错误: {dk_k.shape[2]} != {N + k}（覆盖回归）")
        drafter_kv = {config["num_hidden_layers"]: (dk_k, dk_v)}
        logits_draft_k = F.linear(out_row.to(torch.float32), lm_head_gpu)
        draft_k = int(logits_draft_k.argmax().item())
        hit_k = draft_k == gen_k
        drafts.append(draft_k)
        hits.append(hit_k)
        t_mtp += time.time() - t2
        free_gpu(emb_keep, h_v, out_row, logits_k, logits_draft_k)
        peak_vram = max(peak_vram, vram_mb())
        log(f"MTP   draft(token{k+1} 候选): {draft_k} = {tok.decode([draft_k])!r} "
            f"| 与主模型 token{k+1}({gen_k}) 一致: {hit_k} ({time.time()-t2:.2f}s)")
    free_gpu(lm_head_gpu)
    torch.cuda.empty_cache()

    # ---- 5. 结果 ----
    text = tok.decode(gen)
    n = len(gen)
    nd = len(hits)  # EOS 提前停止时最后一轮无 draft，draft 数 = 生成数 - 1
    hit_count = sum(hits)
    log("=" * 60)
    log(f"生成 token: {gen} -> 解码文本: {text!r}")
    log(f"MTP draft 命中: {hit_count}/{nd} ({100.0 * hit_count / nd:.1f}%)")
    if n < NUM_GEN_TOKENS:
        log(f"提前停止于第 {n} 个 token（EOS），未达 HY3_NUM_GEN_TOKENS={NUM_GEN_TOKENS}")
    log(f"耗时分解: 生成循环 verify 累计 {t_verify:.0f}s, MTP 累计 {t_mtp:.0f}s "
        f"(MTP 占循环 {(100.0 * t_mtp / (t_verify + t_mtp + 1e-9)):.1f}%)")
    log(f"峰值显存: {peak_vram:.0f} MB")
    log(f"总耗时 {time.time()-t0:.1f}s")

    store.close_all()
    # 语义验证
    if "北京" in text:
        log("语义验证: 通过（输出包含『北京』）")
    else:
        log("语义验证: 异常（未包含『北京』）")


if __name__ == "__main__":
    main()
