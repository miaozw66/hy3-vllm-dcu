"""
验证 vLLM HY3 推理 vs MetaInfer golden dump (TP=4)
逐层逐 dump point 计算余弦相似度

用法: torchrun --nproc_per_node=4 verify.py
"""
import glob
import json
import os

import safetensors
import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer

# 禁用 torch.compile 以避免 Triton INT8 kernel 的 dynamo 兼容性问题
dynamo.config.disable = True
dynamo.config.suppress_errors = True

from vllm.config import (
    VllmConfig,
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    LoadConfig,
)
from vllm.config.vllm import set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM, HYV3DecoderLayer

# ── 配置 ──────────────────────────────────────────────
MODEL_DIR = "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master"
GOLDEN_DIR = "/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/golden_dump"
PROMPT = "中国的首都是"
NUM_LAYERS = 65  # 完整验证：前 65 层（L0 稠密 + L1-64 MoE），根因已修复（路由 autocast 外）
# 退化层：最近一次运行中 05_mlp_out 余弦相似度 < 0.996 的层。
# 对这些层用逐 expert 循环（与 MetaInfer _compute_moe_from_loaded 同构）
# 替代 vLLM fused_experts kernel，验证「专家累加方式」是否为误差根因。
DEGRADED_LAYERS = {9, 13, 16, 21}
DUMP_POINTS = [
    "00_input",
    "01_input_layernorm",
    "02_attention_out",
    "03_attention_residual",
    "04_post_attention_layernorm",
    "05_mlp_out",
    "06_output",
]

# 全局变量，仅 rank 0 收集 hidden states
captured = {}
_golden = None
_saved_hidden_states = None  # 存储最后层的 hidden_states + residual 供后续批次使用
_ref_input_ids = None  # 参考输入 token ids（rank 0，CPU），用于 L0 embedding 诊断
_emb_weight_ref = None  # checkpoint 原始 embed_tokens.weight（CPU），用于 L0 embedding 诊断
_ckpt_index_cache = None  # {ckpt key: shard path}，专家/shared 权重定位索引


def get_ckpt_index():
    """构建 checkpoint key → shard 文件路径索引（一次性，全局缓存）。

    仅索引 expert 与 shared_mlp 权重 key（每层 forward 时按需补全
    EP 分片外缺失的专家权重，消除 all_reduce 求和顺序差异）。
    """
    global _ckpt_index_cache
    if _ckpt_index_cache is None:
        idx = {}
        for fname in sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors")):
            try:
                with safetensors.safe_open(fname, framework='pt') as f:
                    for k in f.keys():
                        if 'mlp.experts' in k or 'mlp.shared_mlp' in k:
                            idx[k] = fname
            except Exception:
                pass
        _ckpt_index_cache = idx
    return _ckpt_index_cache


def _ckpt_dequant(prefix: str, idx: dict, device):
    """从 checkpoint 读取 INT8 权重 + per-row scale 并 dequant 为 BF16。

    与 golden weight_loader._dequantize_int8 同构：
    (w_int8.float() * scale.float()).bfloat16()，scale [out, 1] 广播。
    """
    w_k = prefix + ".weight"
    s_k = prefix + ".weight_scale"
    with safetensors.safe_open(idx[w_k], framework='pt') as f:
        w = f.get_tensor(w_k)
        s = f.get_tensor(s_k)
    return (w.float() * s.float()).bfloat16().to(device)


def make_hooked_forward(layer_idx, layer_module):
    """替换 HYV3DecoderLayer.forward，在各阶段捕获 hidden states (仅 rank 0)

    手动实现注意力计算以绕过 vLLM 的 Attention 层（它需要 ForwardContext）。
    """
    attn = layer_module.self_attn
    orig_input_ln = layer_module.input_layernorm
    orig_post_ln = layer_module.post_attention_layernorm
    orig_mlp = layer_module.mlp
    rank = get_tensor_model_parallel_rank()
    ckpt_file_list = sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors"))
    ckpt_index = get_ckpt_index()

    # 缓存注意力参数以避免前向传播中重复的属性访问
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    head_dim = attn.head_dim
    q_size = attn.q_size
    kv_size = attn.kv_size
    scaling = attn.scaling
    use_qk_norm = attn.use_qk_norm
    q_norm = attn.q_norm if use_qk_norm else None
    k_norm = attn.k_norm if use_qk_norm else None
    rotary_emb = attn.rotary_emb
    qkv_proj = attn.qkv_proj
    o_proj = attn.o_proj

    def manual_attention(hidden_states, positions):
        """手动计算注意力：QKV 投影 + RoPE + SDPA + 输出投影
        使用 BF16 解量化权重，避免 INT8 量化误差累积。"""

        def dequant_weight(linear_layer):
            """解量化 INT8 线性层权重为 BF16"""
            w_int = linear_layer.weight.data
            w_scale = linear_layer.weight_scale.data
            scale_adj = w_scale
            while scale_adj.dim() < w_int.dim():
                scale_adj = scale_adj.unsqueeze(-1)
            # Handle transposed storage
            if scale_adj.shape[0] != w_int.shape[0] and scale_adj.shape[0] == w_int.shape[1]:
                w_int = w_int.T
            w_bf16 = (w_int.float() * scale_adj.float()).bfloat16()
            return w_bf16

        # QKV 投影 — 使用 BF16 解量化权重。
        # 与 golden 同构：3 个独立投影（合并 mm 的 cublas kernel 选择不同，
        # 累加顺序差异会产生 ~1e-5 级输出差异）。
        qkv_w_bf16 = dequant_weight(qkv_proj)
        h_bf16 = hidden_states.bfloat16()
        wq = qkv_w_bf16[:q_size]
        wk = qkv_w_bf16[q_size:q_size + kv_size]
        wv = qkv_w_bf16[q_size + kv_size:]
        q = torch.mm(h_bf16, wq.T)
        k = torch.mm(h_bf16, wk.T)
        v = torch.mm(h_bf16, wv.T)

        # QK 归一化 — 与 golden _apply_head_norm 同构（fp32 除法 + fp32 weight）
        if use_qk_norm:
            q = q.view(-1, num_heads, head_dim)
            k = k.view(-1, num_kv_heads, head_dim)

            def head_norm(x, nw):
                x_f32 = x.float()
                rms = torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + 1e-5)
                x_norm = x_f32 / rms
                return (x_norm * nw.float()).to(x.dtype)

            q = head_norm(q, q_norm.weight)
            k = head_norm(k, k_norm.weight)

        # 旋转位置编码 — 匹配 MetaInfer（GOLDEN-IN attn 实验证实 FP32 乘法
        # 偏离 golden 2.2e-5）：half-split 旋转 + cos/sin fp32 cache cast 到
        # BF16 后做 BF16 乘法（golden rope.apply 行为）。
        if not use_qk_norm:
            q = q.view(-1, num_heads, head_dim)
            k = k.view(-1, num_kv_heads, head_dim)

        def rope_mi(x):
            theta = 11158840.0
            inv_freq = 1.0 / (theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))
            required_len = int(positions.max().item()) + 1
            t = torch.arange(required_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[positions].to(dtype=x.dtype).unsqueeze(1)
            sin = emb.sin()[positions].to(dtype=x.dtype).unsqueeze(1)
            half = head_dim // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            cos1 = cos[..., :half]
            sin1 = sin[..., :half]
            x1_rot = x1 * cos1 - x2 * sin1
            x2_rot = x1 * sin1 + x2 * cos1
            return torch.cat([x1_rot, x2_rot], dim=-1)

        q = rope_mi(q)
        k = rope_mi(k)

        # 重塑为 (num_tokens, num_heads, head_dim) 用于 SDPA
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)

        # GQA: 如果 num_kv_heads < num_heads，则重复 K/V 头
        if num_kv_heads < num_heads:
            n_groups = num_heads // num_kv_heads
            k = k.unsqueeze(2).expand(-1, num_kv_heads, n_groups, head_dim)
            k = k.reshape(-1, num_heads, head_dim)
            v = v.unsqueeze(2).expand(-1, num_kv_heads, n_groups, head_dim)
            v = v.reshape(-1, num_heads, head_dim)

        # SDPA 期望 (batch, heads, seq, head_dim)
        # 当前 q/k/v 是 (seq, heads, head_dim)，需要 permute 为 (heads, seq, head_dim)
        q = q.permute(1, 0, 2).unsqueeze(0)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)

        # Scaled dot-product attention (causal for prefill, matching MetaInfer)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scaling, is_causal=True)

        # attn_out: (1, heads, seq, head_dim) → (seq, heads*head_dim)
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(-1, num_heads * head_dim)

        # 输出投影 — 使用 BF16 解量化权重 + all-reduce (RowParallelLinear)
        o_w_bf16 = dequant_weight(o_proj)
        output = torch.mm(attn_out.bfloat16(), o_w_bf16.T)
        dist.all_reduce(output, group=dist.group.WORLD)
        return output

    # ── Hook MoE forward 以捕获 shared/routed 分解 ──
    # 仅对 HYV3MoEFused 层（有 gate 属性）hook，Layer 0 是 HYV3FeedForward 跳过
    is_moe = hasattr(layer_module.mlp, 'gate')
    moe_captured = {}

    if is_moe:
        shared_mlp = layer_module.mlp.shared_mlp if hasattr(layer_module.mlp, 'shared_mlp') else None
        experts_layer = layer_module.mlp.experts  # FusedMoE module

        # 缓存路由参数（不占显存）
        gate_weight = layer_module.mlp.gate.weight.data.float()
        # 注意：vLLM 加载的 expert_bias 与 checkpoint 原始值有 ~2.78e-5 精度差异
        # （L22 诊断实测），会通过 sigmoid → renorm → 权重进入 MoE 输出。
        # golden 直接用 checkpoint 原始值（_load_fp_weight），这里也从
        # checkpoint 读取以与 golden 逐位一致。
        expert_bias = layer_module.mlp.expert_bias.data.float()
        bias_key = f"model.layers.{layer_idx}.mlp.expert_bias"
        for fname_b in ckpt_file_list:
            f_b = safetensors.safe_open(fname_b, framework='pt')
            if bias_key in f_b.keys():
                expert_bias = f_b.get_tensor(bias_key).float().to(
                    layer_module.mlp.expert_bias.device
                )
                if rank == 0:
                    print(f"  [BIAS L{layer_idx}] using checkpoint bias "
                          f"(dtype={f_b.get_tensor(bias_key).dtype})")
                break
        routed_scaling = experts_layer.routed_scaling_factor
        renormalize_flag = experts_layer.router.renormalize if hasattr(experts_layer.router, 'renormalize') else True

        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        def hooked_moe_forward(hidden_states):
            orig_shape = hidden_states.shape
            hidden_dim = hidden_states.shape[-1]
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            h_bf16 = hidden_states_flat.bfloat16()
            h_f32 = hidden_states_flat.float()

            # ── MetaInfer 风格路由 ──
            # 关键修复：路由 mm 必须在 autocast 外（fp32 精确）。verify 的
            # forward 在 torch.autocast(bfloat16) 内，fp32 mm 被降级为 BF16
            # 计算（logits 误差 ~7.7e-3，batch mm 甚至非确定 ~0.114），会改变
            # top-8 专家选择（与 MetaInfer 的精确 fp32 路由不一致）→ 2.8% off。
            # autocast(enabled=False) 下 fp32 mm 精确且确定（实测 8.3e-7）。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                router_logits = torch.mm(h_f32, gate_weight.T)
            scores = torch.sigmoid(router_logits + expert_bias.unsqueeze(0))
            raw_weights, raw_ids = torch.topk(scores, 8, dim=-1, sorted=True)

            if renormalize_flag:
                topk_weights = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                topk_weights = raw_weights
            topk_weights = topk_weights * routed_scaling
            topk_ids = raw_ids.to(torch.int32)

            # ── Shared expert: checkpoint 全量权重（golden SharedMLP 同构）──
            # vLLM 的 shared_mlp 是 TP 分片的（gate_up [768,4096] + down
            # [384,4096]），K 维分片 + all_reduce 的求和顺序与 golden 全量
            # [1536,4096]/[4096,1536] 不同 → 从 checkpoint 加载全量权重。
            shared_out = None
            if shared_mlp is not None:
                sh_base = f"model.layers.{layer_idx}.mlp.shared_mlp."
                sh_gate_w = _ckpt_dequant(sh_base + "gate_proj", ckpt_index, h_bf16.device)
                sh_up_w = _ckpt_dequant(sh_base + "up_proj", ckpt_index, h_bf16.device)
                sh_down_w = _ckpt_dequant(sh_base + "down_proj", ckpt_index, h_bf16.device)
                # golden SharedMLP.forward 同构：gate/up 独立投影 + F.linear
                s_gate = F.silu(F.linear(h_bf16, sh_gate_w))
                s_up = F.linear(h_bf16, sh_up_w)
                s_hidden = s_gate * s_up
                shared_out = F.linear(s_hidden, sh_down_w)

            # ── Routed experts: 即时 dequantize BF16 + MetaInfer routing ──
            w13_int = experts_layer.w13_weight.data
            w2_int = experts_layer.w2_weight.data
            w13_scale = experts_layer.w13_weight_scale.data
            w2_scale = experts_layer.w2_weight_scale.data

            w13_scale_sq = w13_scale
            while w13_scale_sq.dim() < w13_int.dim():
                w13_scale_sq = w13_scale_sq.unsqueeze(-1)
            w2_scale_sq = w2_scale
            while w2_scale_sq.dim() < w2_int.dim():
                w2_scale_sq = w2_scale_sq.unsqueeze(-1)

            w13_bf16 = (w13_int.float() * w13_scale_sq.float()).bfloat16()
            w2_bf16 = (w2_int.float() * w2_scale_sq.float()).bfloat16()

            # 收集本 batch 需要的全部专家（去重）。EP 分片下每 rank 只有 48 个
            # 本地专家（emap≥0），其他 rank 的专家从 checkpoint 补全 —— 这样
            # 每个 rank 都拥有 top-8 的全部权重，k=0..7 顺序累加与 golden
            # TP=1 完全同构，无需 all_reduce（all_reduce 的 rank 间求和顺序
            # 与 golden 顺序累加不同，是 L1 退化层 3.9e-4 差异的根因）。
            emap = experts_layer.expert_map
            needed = sorted({int(e) for e in topk_ids.flatten().tolist()})
            experts_w = {}
            for global_e in needed:
                local_e = emap[global_e].item()
                if local_e >= 0:
                    w1 = w13_bf16[local_e]          # [3072, 4096] = [gate; up]
                    experts_w[global_e] = (w1[:1536], w1[1536:], w2_bf16[local_e])
                else:
                    exp_base = f"model.layers.{layer_idx}.mlp.experts.{global_e}."
                    experts_w[global_e] = (
                        _ckpt_dequant(exp_base + "gate_proj", ckpt_index, h_bf16.device),
                        _ckpt_dequant(exp_base + "up_proj", ckpt_index, h_bf16.device),
                        _ckpt_dequant(exp_base + "down_proj", ckpt_index, h_bf16.device),
                    )

            # ── Routed experts: golden 完全同构逐 expert 计算（k=0..7 顺序累加）──
            # 与 MetaInfer _compute_expert + _compute_moe_from_loaded 逐行同构：
            #   gate = F.silu(F.linear(x, gate_proj))
            #   up   = F.linear(x, up_proj)
            #   token_output = token_output + F.linear(gate*up, down_proj) * score
            # fp32 token_out（golden 中 token_output 被 * score 提升为 fp32）。
            routed_out = torch.zeros(
                hidden_states_flat.shape[0], hidden_dim,
                device=h_bf16.device, dtype=torch.float32,
            )
            for t in range(topk_ids.shape[0]):
                token_out = torch.zeros(
                    1, hidden_dim, device=h_bf16.device, dtype=torch.float32
                )
                h_t = h_bf16[t:t + 1]
                for k in range(topk_ids.shape[1]):
                    global_e = topk_ids[t, k].item()
                    score = topk_weights[t, k]
                    gate_w, up_w, down_w = experts_w[global_e]
                    gate = F.silu(F.linear(h_t, gate_w))
                    up = F.linear(h_t, up_w)
                    token_out = token_out + F.linear(gate * up, down_w) * score
                routed_out[t:t + 1] = token_out

            routed_local_before_ar = routed_out.clone()
            shared_local_before_ar = shared_out.clone() if shared_out is not None else None

            # 无 all_reduce —— 每 rank 全量权重计算，结果相同（golden TP=1 同构）
            if shared_out is not None:
                final_hidden_states = (routed_out + shared_out).bfloat16()
            else:
                final_hidden_states = routed_out.bfloat16()

            # ── L1 深度诊断：输入 bitwise 一致时的 MoE 差异根因 ──
            # L1 是第一个真实断裂点（05_mlp_out cos=0.99961，输入 04 与 golden
            # bitwise 相同）。对比模型 routing 与 golden routing（checkpoint
            # gate/bias + golden 04 输入），并做 GOLDEN-IN MoE（golden 输入 +
            # golden routing + 同构专家计算）定位差异源。
            # 注意：计算块必须所有 rank 执行（含 all_reduce），
            # _golden 只在 rank 0 加载，不能作为块条件。
            if layer_idx == 1:
                g04 = torch.load(
                    os.path.join(GOLDEN_DIR, "layer_001", "04_post_attention_layernorm.pt"),
                    map_location=h_bf16.device,
                ).bfloat16().reshape(-1, hidden_dim)
                g_in_f = g04.float()
                g_gate = None
                g_bias = None
                for fname in ckpt_file_list:
                    f = safetensors.safe_open(fname, framework='pt')
                    if 'model.layers.1.mlp.router.gate.weight' in f.keys():
                        g_gate = f.get_tensor(
                            'model.layers.1.mlp.router.gate.weight').float().to(h_bf16.device)
                    if 'model.layers.1.mlp.expert_bias' in f.keys():
                        g_bias = f.get_tensor(
                            'model.layers.1.mlp.expert_bias').float().to(h_bf16.device)
                    if g_gate is not None and g_bias is not None:
                        break
                if rank == 0 and _golden is not None:
                    gw_diff = (gate_weight - g_gate).abs().max().item()
                    eb_diff = (expert_bias - g_bias).abs().max().item()
                    print(f"  [L1 DIAG] gate vs ckpt max_diff={gw_diff:.6e}, "
                          f"bias vs ckpt max_diff={eb_diff:.6e}")
                    ml_diff = (h_f32 - g_in_f).abs().max().item()
                    print(f"  [L1 DIAG] model 04 input vs golden 04: "
                          f"max_diff={ml_diff:.6e}")
                # golden routing：golden 输入 + checkpoint gate/bias（_route 同构）
                g_logits = torch.mm(g_in_f, g_gate.T)
                g_scores = torch.sigmoid(g_logits + g_bias.unsqueeze(0))
                g_raw_w, g_raw_ids = torch.topk(g_scores, 8, dim=-1)
                if renormalize_flag:
                    g_w = g_raw_w / (g_raw_w.sum(dim=-1, keepdim=True) + 1e-20)
                else:
                    g_w = g_raw_w
                g_w = g_w * routed_scaling
                g_ids = g_raw_ids.to(torch.int32)
                # GOLDEN-IN MoE：golden 输入 + golden routing + 同构专家循环
                g_routed = torch.zeros(
                    h_bf16.shape[0], hidden_dim,
                    device=h_bf16.device, dtype=torch.float32,
                )
                for t in range(g_ids.shape[0]):
                    g_token_out = torch.zeros(
                        1, hidden_dim, device=h_bf16.device, dtype=torch.float32
                    )
                    g_ht = g04[t:t + 1]
                    for k in range(g_ids.shape[1]):
                        global_e = g_ids[t, k].item()
                        local_e = emap[global_e].item()
                        if local_e < 0:
                            continue
                        score = g_w[t, k]
                        w1_e = w13_bf16[local_e]
                        half_g = w1_e.shape[0] // 2
                        gg = F.silu(F.linear(g_ht, w1_e[:half_g]))
                        uu = F.linear(g_ht, w1_e[half_g:])
                        g_token_out = g_token_out + F.linear(gg * uu, w2_bf16[local_e]) * score
                    g_routed[t:t + 1] = g_token_out
                dist.all_reduce(g_routed, group=dist.group.WORLD)
                if shared_out is not None:
                    g_full = (g_routed + shared_out).bfloat16()
                else:
                    g_full = g_routed.bfloat16()
                if rank == 0 and _golden is not None:
                    g05 = _golden[1]["05_mlp_out"].bfloat16().reshape(-1, hidden_dim).to(g_full.device)
                    print(f"  [L1 DIAG] GOLDEN-IN MoE (golden input+routing) "
                          f"vs golden 05: {cosine_similarity(g_full, g05):.8f}")
                    # routing 对比（模型 routing 已用 bitwise 相同的输入计算）
                    m_logits = torch.mm(h_f32, gate_weight.T)
                    ml_gd = (m_logits - g_logits).abs().max().item()
                    print(f"  [L1 DIAG] model logits vs golden logits: "
                          f"max_diff={ml_gd:.6e}")
                    m_set = set(topk_ids[0].tolist())
                    g_set = set(g_ids[0].tolist())
                    overlap = len(m_set & g_set)
                    print(f"  [L1 DIAG] top-8 overlap model vs golden: {overlap}/8")
                    if overlap < 8:
                        print(f"  [L1 DIAG] only model:  {sorted(m_set - g_set)}")
                        print(f"  [L1 DIAG] only golden: {sorted(g_set - m_set)}")
                        m9 = scores[0].topk(9).values
                        g9 = g_scores[0].topk(9).values
                        print(f"  [L1 DIAG] model scores 7th={m9[6].item():.10f} "
                              f"8th={m9[7].item():.10f} 9th={m9[8].item():.10f}")
                        print(f"  [L1 DIAG] gold  scores 7th={g9[6].item():.10f} "
                              f"8th={g9[7].item():.10f} 9th={g9[8].item():.10f}")

                    # ── fp64 参考计算：golden 输入 + golden 路由 + ckpt 全量权重 ──
                    # 数学上精确的 MoE 输出参考（BF16 权重值提升到 fp64 完全精确）。
                    # 若 golden 05 ≈ fp64 ref 而 hook 主路径差 2.8%，差异在 hook 的
                    # 权重/结构；若 golden 本身远离 fp64 ref，则 golden 计算与
                    # "纯数学"有结构差异。
                    g_needed = sorted({int(e) for e in g_ids.flatten().tolist()})
                    g_experts_ckpt = {}
                    for e in g_needed:
                        exp_base = f"model.layers.1.mlp.experts.{e}."
                        g_experts_ckpt[e] = (
                            _ckpt_dequant(exp_base + "gate_proj", ckpt_index, h_bf16.device),
                            _ckpt_dequant(exp_base + "up_proj", ckpt_index, h_bf16.device),
                            _ckpt_dequant(exp_base + "down_proj", ckpt_index, h_bf16.device),
                        )
                    # per-token 路由（golden _route 同构：[1,4096] mm）vs batch mm
                    pt_logits = []
                    for t in range(g04.shape[0]):
                        pt_logits.append(F.linear(g04[t:t + 1].float(), g_gate) + g_bias)
                    pt_logits = torch.cat(pt_logits, 0)
                    pt_scores = torch.sigmoid(pt_logits)
                    pt_raw_w, pt_raw_ids = torch.topk(pt_scores, 8, dim=-1)
                    pt_w = pt_raw_w / (pt_raw_w.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling
                    # fp64 参考：golden 输入 + per-token golden 路由 + ckpt 权重
                    ref_in = g04.double()
                    ref_routed = torch.zeros_like(ref_in)
                    for t in range(g_ids.shape[0]):
                        for k in range(g_ids.shape[1]):
                            e = g_ids[t, k].item()
                            score = pt_w[t, k].double()
                            gw, uw, dw = g_experts_ckpt[e]
                            r_gate = F.silu(F.linear(ref_in[t:t + 1], gw.double()))
                            r_up = F.linear(ref_in[t:t + 1], uw.double())
                            ref_routed[t:t + 1] += F.linear(r_gate * r_up, dw.double()) * score
                    ref_shared = F.linear(
                        F.silu(F.linear(ref_in, sh_gate_w.double())) * F.linear(ref_in, sh_up_w.double()),
                        sh_down_w.double(),
                    )
                    ref_full = (ref_routed + ref_shared).bfloat16()
                    # fp64 参考（混合权重：本地 vLLM + 远端 ckpt，与 hook 主路径同构）
                    ref_routed_mixed = torch.zeros_like(ref_in)
                    for t in range(g_ids.shape[0]):
                        for k in range(g_ids.shape[1]):
                            e = g_ids[t, k].item()
                            le = emap[e].item()
                            score = pt_w[t, k].double()
                            if le >= 0:
                                w1 = w13_bf16[le].double()
                                gw2, uw2, dw2 = w1[:1536], w1[1536:], w2_bf16[le].double()
                            else:
                                gw2, uw2, dw2 = (x.double() for x in g_experts_ckpt[e])
                            r_gate = F.silu(F.linear(ref_in[t:t + 1], gw2))
                            r_up = F.linear(ref_in[t:t + 1], uw2)
                            ref_routed_mixed[t:t + 1] += F.linear(r_gate * r_up, dw2) * score
                    # vLLM 本地专家权重 vs ckpt dequant（逐个专家 max_diff）
                    vllm_md = {}
                    for e in g_needed:
                        le = emap[e].item()
                        if le < 0:
                            continue
                        w1 = w13_bf16[le]
                        cg, cu, cd = g_experts_ckpt[e]
                        vllm_md[e] = max(
                            (w1[:1536].float() - cg.float()).abs().max().item(),
                            (w1[1536:].float() - cu.float()).abs().max().item(),
                            (w2_bf16[le].float() - cd.float()).abs().max().item(),
                        )
                    hook_full = final_hidden_states.reshape(-1, hidden_dim)
                    print(f"  [L1 DIAG] fp64 ref (ckpt 权重)   vs golden 05: {cosine_similarity(ref_full, g05):.8f}")
                    print(f"  [L1 DIAG] fp64 ref vs hook 主路径输出:        {cosine_similarity(ref_full, hook_full):.8f}")
                    ref_mixed_full = (ref_routed_mixed + ref_shared).bfloat16()
                    print(f"  [L1 DIAG] fp64 ref (混合权重)     vs golden 05: {cosine_similarity(ref_mixed_full, g05):.8f}")
                    print(f"  [L1 DIAG] fp64 ref (混合) vs fp64 ref (ckpt):   {cosine_similarity(ref_mixed_full, ref_full):.8f}")
                    rr = ref_routed.bfloat16()
                    print(f"  [L1 DIAG] hook routed vs ref routed: {cosine_similarity(routed_out.bfloat16().reshape(-1, hidden_dim), rr):.8f}")
                    rs = ref_shared.bfloat16()
                    sh_bit = torch.equal(
                        shared_out.bfloat16().reshape(-1, hidden_dim), rs.reshape(-1, hidden_dim)
                    )
                    print(f"  [L1 DIAG] hook shared vs ref shared bitwise: {sh_bit}")
                    md_ptb = (pt_logits - g_logits).abs().max().item()
                    print(f"  [L1 DIAG] per-token vs batch 路由 logits max_diff={md_ptb:.6e}")
                    print(f"  [L1 DIAG] per-token vs batch 路由 top-8 overlap: "
                          f"{len(set(pt_raw_ids[0].tolist()) & set(g_ids[0].tolist()))}/8")
                    # ── 探针：in-run fp32 mm 数值行为 ──
                    try:
                        print(f"  [L1 DIAG] in-run allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
                              f"fp32_precision={getattr(torch.backends.cuda.matmul, 'fp32_precision', 'n/a')} "
                              f"enabled={torch.is_autocast_enabled('cuda')}", flush=True)
                    except Exception:
                        pass
                    bt_r = torch.mm(g04.float(), g_gate.T) + g_bias
                    print(f"  [L1 DIAG] batch mm 两次 max_diff={(g_logits.float()-bt_r.float()).abs().max().item():.6e}", flush=True)
                    pt_r = torch.cat([F.linear(g04[t:t + 1].float(), g_gate) + g_bias for t in range(g04.shape[0])], 0)
                    print(f"  [L1 DIAG] per-token 两次 max_diff={(pt_logits.float()-pt_r.float()).abs().max().item():.6e}", flush=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                        bt_noac = torch.mm(g04.float(), g_gate.T) + g_bias
                        pt_noac = torch.cat([F.linear(g04[t:t + 1].float(), g_gate) + g_bias
                                             for t in range(g04.shape[0])], 0)
                    print(f"  [L1 DIAG] autocast外 pt vs batch max_diff={(bt_noac-pt_noac).abs().max().item():.6e}", flush=True)
                    print(f"  [L1 DIAG] logits 值域: pt max={pt_logits.abs().max().item():.4f} "
                          f"mean={pt_logits.abs().mean().item():.4f}", flush=True)
                    lg64 = torch.mm(g04.double(), g_gate.double().T) + g_bias.double()
                    print(f"  [L1 DIAG] fp64 vs batch    max_diff={(lg64.float()-g_logits.float()).abs().max().item():.6e}", flush=True)
                    print(f"  [L1 DIAG] fp64 vs per-token max_diff={(lg64.float()-pt_logits.float()).abs().max().item():.6e}", flush=True)

                    # ── 决定性对照：in-run v4_from 同构（精确路由 + ckpt 全量 BF16 +
                    #    顺序累加，与 standalone test_ac_v4 逐字节同构）──
                    # ≈0.99999404 → BF16 mm 环境无污染，hook 差异全部来自路由；
                    # ≈0.9996    → in-run 环境改变 BF16 mm 数值（standalone=0.99999404）
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                        ex_logits = torch.mm(g04.float(), g_gate.T) + g_bias
                    ex_scores = torch.sigmoid(ex_logits)
                    ex_raw_w, ex_raw_ids = torch.topk(ex_scores, 8, dim=-1)
                    ex_w = ex_raw_w / (ex_raw_w.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling
                    # 精确路由可能选降级路由外的专家，补齐权重
                    for e in sorted({int(v) for v in ex_raw_ids.flatten().tolist()}):
                        if e not in g_experts_ckpt:
                            exp_base = f"model.layers.1.mlp.experts.{e}."
                            g_experts_ckpt[e] = (
                                _ckpt_dequant(exp_base + "gate_proj", ckpt_index, g04.device),
                                _ckpt_dequant(exp_base + "up_proj", ckpt_index, g04.device),
                                _ckpt_dequant(exp_base + "down_proj", ckpt_index, g04.device),
                            )
                    v4_routed = torch.zeros(3, 4096, device=g04.device, dtype=torch.float32)
                    for t in range(3):
                        v4_token = torch.zeros(1, 4096, device=g04.device, dtype=torch.float32)
                        for k in range(8):
                            e = int(ex_raw_ids[t, k].item())
                            score = ex_w[t, k]
                            gw, uw, dw = g_experts_ckpt[e]
                            vg = F.silu(F.linear(g04[t:t + 1], gw))
                            vu = F.linear(g04[t:t + 1], uw)
                            v4_token = v4_token + F.linear(vg * vu, dw) * score
                        v4_routed[t:t + 1] = v4_token
                    v4_shared = F.linear(
                        F.silu(F.linear(g04, sh_gate_w)) * F.linear(g04, sh_up_w), sh_down_w)
                    v4_full = (v4_routed + v4_shared).bfloat16()
                    print(f"  [L1 DIAG] in-run v4 (精确路由+ckpt全量) vs golden 05: "
                          f"{cosine_similarity(v4_full, g05):.8f}", flush=True)
                    print(f"  [L1 DIAG] in-run v4 vs hook 主路径: "
                          f"{cosine_similarity(v4_full, hook_full):.8f}", flush=True)
                    # 路由分解：hook 路由（降级）vs 精确路由
                    print(f"  [L1 DIAG] ex_w vs hook topk_weights max_diff="
                          f"{(ex_w - topk_weights).abs().max().item():.6e}", flush=True)
                    print(f"  [L1 DIAG] ex_ids vs hook topk_ids 相同: "
                          f"{torch.equal(ex_raw_ids, topk_ids)}", flush=True)
                    # 路由影响量化：v4 结构 + hook 路由
                    # （hook 路由可能选 golden 路由外的专家，先补齐权重）
                    for e in sorted({int(v) for v in topk_ids.flatten().tolist()}):
                        if e not in g_experts_ckpt:
                            exp_base = f"model.layers.1.mlp.experts.{e}."
                            g_experts_ckpt[e] = (
                                _ckpt_dequant(exp_base + "gate_proj", ckpt_index, g04.device),
                                _ckpt_dequant(exp_base + "up_proj", ckpt_index, g04.device),
                                _ckpt_dequant(exp_base + "down_proj", ckpt_index, g04.device),
                            )
                    v4_hook = torch.zeros(3, 4096, device=g04.device, dtype=torch.float32)
                    for t in range(3):
                        v4_token = torch.zeros(1, 4096, device=g04.device, dtype=torch.float32)
                        for k in range(8):
                            e = int(topk_ids[t, k].item())
                            score = topk_weights[t, k]
                            gw, uw, dw = g_experts_ckpt[e]
                            vg = F.silu(F.linear(g04[t:t + 1], gw))
                            vu = F.linear(g04[t:t + 1], uw)
                            v4_token = v4_token + F.linear(vg * vu, dw) * score
                        v4_hook[t:t + 1] = v4_token
                    v4_hook_full = (v4_hook + v4_shared).bfloat16()
                    print(f"  [L1 DIAG] v4(hook路由) vs golden 05: "
                          f"{cosine_similarity(v4_hook_full, g05):.8f}", flush=True)
                    print(f"  [L1 DIAG] v4(精确) vs v4(hook路由): "
                          f"{cosine_similarity(v4_full, v4_hook_full):.8f}", flush=True)
                    # shared 分解：同输入同权重下 hook shared vs v4 shared
                    sh_hook = shared_out.bfloat16().reshape(-1, hidden_dim)
                    print(f"  [L1 DIAG] v4_shared vs hook shared cos: "
                          f"{cosine_similarity(v4_shared.bfloat16().reshape(-1, hidden_dim), sh_hook):.8f} "
                          f"bitwise={torch.equal(v4_shared.bfloat16().reshape(-1, hidden_dim), sh_hook)}", flush=True)
                    if vllm_md:
                        print(f"  [L1 DIAG] vLLM vs ckpt 专家权重 max_diff: {vllm_md}")
                    else:
                        print(f"  [L1 DIAG] 本 rank 无本地专家（全部远端）")

            # ── Layer 22 深度诊断：手动完整 MoE 计算（纯 PyTorch，零 vLLM kernel）──
            if layer_idx == 22:
                routed_manual = torch.zeros_like(routed_out, dtype=torch.bfloat16)
                routed_manual_f32 = torch.zeros_like(routed_out, dtype=torch.float32)
                # Also compute WITHOUT the router_weight factor for diagnostic
                routed_no_weight = torch.zeros_like(routed_out, dtype=torch.bfloat16)
                emap = experts_layer.expert_map
                for t in range(topk_ids.shape[0]):
                    for k in range(topk_ids.shape[1]):
                        global_e = topk_ids[t, k].item()
                        local_e = emap[global_e].item()
                        if local_e < 0:
                            continue
                        weight = topk_weights[t, k].float()
                        w1_e = w13_bf16[local_e]
                        w2_e = w2_bf16[local_e]
                        h_t = h_bf16[t:t+1]
                        gu = torch.mm(h_t, w1_e.T)
                        half_g = gu.shape[-1] // 2
                        g, u = gu[..., :half_g], gu[..., half_g:]
                        act = F.silu(g) * u
                        e_out = torch.mm(act, w2_e.T)
                        routed_manual[t] += weight * e_out.squeeze(0)
                        routed_no_weight[t] += 1.0 * e_out.squeeze(0)  # unweighted
                        # Float32
                        h_t_f32 = h_t.float()
                        w1_e_f32 = w1_e.float()
                        w2_e_f32 = w2_e.float()
                        gu_f32 = torch.mm(h_t_f32, w1_e_f32.T)
                        g_f32, u_f32 = gu_f32[..., :half_g], gu_f32[..., half_g:]
                        act_f32 = F.silu(g_f32) * u_f32
                        e_out_f32 = torch.mm(act_f32, w2_e_f32.T)
                        routed_manual_f32[t] += weight * e_out_f32.squeeze(0)
                manual_pre_ar_norm = routed_manual.float().norm().item()
                dist.all_reduce(routed_manual, group=dist.group.WORLD)
                dist.all_reduce(routed_manual_f32, group=dist.group.WORLD)
                dist.all_reduce(routed_no_weight, group=dist.group.WORLD)

                if shared_out is not None:
                    full_manual = shared_out + routed_manual
                    full_manual_f32 = shared_out.float() + routed_manual_f32
                else:
                    full_manual = routed_manual
                    full_manual_f32 = routed_manual_f32

                # ── 决定性实验：golden 输入 → 同构 MoE 计算（所有 rank 参与）──
                # 用 golden 的 04 输入 + 逐 expert 循环（与 golden 同构）。
                # ≈1.0 → MoE 计算完全对齐，误差 100% 来自输入路径；
                # 0.99x → MoE 内部仍有未对齐差异。
                g_in_bf16 = torch.load(
                    os.path.join(GOLDEN_DIR, "layer_022", "04_post_attention_layernorm.pt"),
                    map_location=h_bf16.device,
                ).bfloat16().reshape(-1, h_bf16.shape[-1])
                g_in_f32 = g_in_bf16.float()
                g_logits = torch.mm(g_in_f32, gate_weight.T)
                g_scores2 = torch.sigmoid(g_logits + expert_bias.unsqueeze(0))
                g_raw_w, g_raw_ids = torch.topk(g_scores2, 8, dim=-1, sorted=True)
                g_w2 = g_raw_w / (g_raw_w.sum(dim=-1, keepdim=True) + 1e-20)
                g_w2 = g_w2 * routed_scaling
                g_ids2 = g_raw_ids.to(torch.int32)
                g_routed = torch.zeros_like(h_bf16, dtype=torch.float32)
                for t in range(g_ids2.shape[0]):
                    g_token_out = torch.zeros(
                        1, h_bf16.shape[-1], device=h_bf16.device, dtype=torch.float32
                    )
                    g_ht = g_in_bf16[t:t + 1]
                    for k in range(g_ids2.shape[1]):
                        global_e = g_ids2[t, k].item()
                        local_e = emap[global_e].item()
                        if local_e < 0:
                            continue
                        weight = g_w2[t, k]
                        w1_e = w13_bf16[local_e]
                        hg = w1_e.shape[0] // 2
                        gg = F.silu(F.linear(g_ht, w1_e[:hg]))
                        uu = F.linear(g_ht, w1_e[hg:])
                        g_token_out = g_token_out + F.linear(gg * uu, w2_bf16[local_e]) * weight
                    g_routed[t:t + 1] = g_token_out
                dist.all_reduce(g_routed, group=dist.group.WORLD)
                # shared expert 也用 golden 输入重算（golden SharedMLP 同构，
                # checkpoint 全量权重，无 all_reduce）
                gs_gate = F.silu(F.linear(g_in_bf16, sh_gate_w))
                gs_up = F.linear(g_in_bf16, sh_up_w)
                g_shared = F.linear(gs_gate * gs_up, sh_down_w)
                g_full = (g_routed + g_shared).bfloat16()

                # ── 决定性实验 3：golden 01 输入 → hook attention → vs golden 02 ──
                # ≈1.0 → attention 计算对齐（误差来自上游累积/残差路径）；
                # 0.99x → attention 内部实现差异（RoPE 乘法 dtype、norm 乘法 dtype 等）。
                g01 = torch.load(
                    os.path.join(GOLDEN_DIR, "layer_022", "01_input_layernorm.pt"),
                    map_location=h_bf16.device,
                ).bfloat16().reshape(-1, h_bf16.shape[-1])
                pos_g = torch.arange(g01.shape[0], device=g01.device, dtype=torch.long)
                attn_g = manual_attention(g01, pos_g)
                g02 = torch.load(
                    os.path.join(GOLDEN_DIR, "layer_022", "02_attention_out.pt"),
                    map_location=h_bf16.device,
                )
                if rank == 0:
                    g02_bf16 = g02.bfloat16().reshape(-1, g02.shape[-1])
                    print(f"  [L22 DIAG] GOLDEN-IN attn vs golden: {cosine_similarity(attn_g, g02_bf16):.8f} "
                          f"(bitwise={torch.equal(attn_g.bfloat16(), g02_bf16)})")

                if rank == 0:
                    gate_shape = gate_weight.shape
                    bias_shape = expert_bias.shape
                    print(f"  [L22 DIAG] Gate weight shape: {gate_shape}, Expert bias shape: {bias_shape}")
                    if shared_mlp is not None:
                        gu_w = shared_mlp.gate_up_proj.weight.data
                        gu_s = shared_mlp.gate_up_proj.weight_scale.data
                        d_w = shared_mlp.down_proj.weight.data
                        d_s = shared_mlp.down_proj.weight_scale.data
                        print(f"  [L22 DIAG] Shared MLP gate_up_proj weight: {gu_w.shape}, scale: {gu_s.shape}")
                        print(f"  [L22 DIAG] Shared MLP down_proj weight: {d_w.shape}, scale: {d_s.shape}")
                    print(f"  [L22 DIAG] GPU local routed (BEFORE all-reduce) norm: {routed_local_before_ar.float().norm():.4f}")
                    if shared_local_before_ar is not None:
                        print(f"  [L22 DIAG] GPU local shared (BEFORE all-reduce) norm: {shared_local_before_ar.float().norm():.4f}")
                    print(f"  [L22 DIAG] AFTER all-reduce: shared_norm={shared_out.float().norm():.4f}, routed_norm={routed_out.float().norm():.4f}")
                    print(f"  [L22 DIAG] Manual routed pre-AR norm: {manual_pre_ar_norm:.4f}, post-AR norm: {routed_manual.float().norm():.4f}")
                    print(f"  [L22 DIAG] Manual routed (AR) norm: {routed_manual.float().norm():.4f}")
                    print(f"  [L22 DIAG] Manual UNWEIGHTED routed (AR) norm: {routed_no_weight.float().norm():.4f}")
                    print(f"  [L22 DIAG] Kernel routed vs Manual routed cos: {cosine_similarity(routed_out, routed_manual):.8f}")
                    print(f"  [L22 DIAG] BF16 manual vs F32 manual cos: {cosine_similarity(routed_manual, routed_manual_f32.bfloat16()):.8f}")
                    # Check the ratio: kernel routed (after AR) / manual routed (after AR)
                    ratio = (routed_out.float().norm() / (routed_manual.float().norm() + 1e-10)).item()
                    print(f"  [L22 DIAG] Kernel/Manual routed norm ratio: {ratio:.6f}")
                    if _golden is not None and layer_idx in _golden:
                        g22_mlp = _golden[layer_idx]["05_mlp_out"].to(routed_out.device)
                        g22_input = _golden[layer_idx]["04_post_attention_layernorm"].to(
                            hidden_states_flat.device)
                        print(f"  [L22 DIAG] MoE input  vs golden: {cosine_similarity(hidden_states_flat, g22_input):.8f}")
                        print(f"  [L22 DIAG] Manual MoE (BF16) vs golden: {cosine_similarity(full_manual, g22_mlp):.8f}")
                        print(f"  [L22 DIAG] Manual MoE (F32)  vs golden: {cosine_similarity(full_manual_f32.bfloat16(), g22_mlp):.8f}")
                        print(f"  [L22 DIAG] GOLDEN-IN loop MoE vs golden: {cosine_similarity(g_full, g22_mlp):.8f}")

                        # ── Checkpoint weight comparison ──
                        print(f"  [L22 DIAG] --- Checkpoint weight comparison ---")
                        ckpt_dir = MODEL_DIR
                        ckpt_files = sorted(glob.glob(f"{ckpt_dir}/model-*-of-*.safetensors"))
                        # Load expert 0 weights from checkpoint for layer 22
                        gate_key = "model.layers.22.mlp.experts.0.gate_proj.weight"
                        gate_scale_key = "model.layers.22.mlp.experts.0.gate_proj.weight_scale"
                        up_key = "model.layers.22.mlp.experts.0.up_proj.weight"
                        up_scale_key = "model.layers.22.mlp.experts.0.up_proj.weight_scale"
                        down_key = "model.layers.22.mlp.experts.0.down_proj.weight"
                        down_scale_key = "model.layers.22.mlp.experts.0.down_proj.weight_scale"

                        ckpt_gate_w = ckpt_gate_s = ckpt_up_w = ckpt_up_s = None
                        ckpt_down_w = ckpt_down_s = None
                        for fname in ckpt_files:
                            f = safetensors.safe_open(fname, framework='pt')
                            if gate_key in f.keys():
                                ckpt_gate_w = f.get_tensor(gate_key)
                                ckpt_gate_s = f.get_tensor(gate_scale_key)
                            if up_key in f.keys():
                                ckpt_up_w = f.get_tensor(up_key)
                                ckpt_up_s = f.get_tensor(up_scale_key)
                            if down_key in f.keys():
                                ckpt_down_w = f.get_tensor(down_key)
                                ckpt_down_s = f.get_tensor(down_scale_key)
                            if all(x is not None for x in [ckpt_gate_w, ckpt_up_w, ckpt_down_w]):
                                break

                        # Dequantize checkpoint weights: w_bf16 = int8 * scale → bf16
                        ckpt_gate_bf16 = (ckpt_gate_w.float() * ckpt_gate_s.float()).bfloat16()
                        ckpt_up_bf16 = (ckpt_up_w.float() * ckpt_up_s.float()).bfloat16()
                        ckpt_down_bf16 = (ckpt_down_w.float() * ckpt_down_s.float()).bfloat16()
                        # Combine gate + up → w13 (same as FusedMoE format)
                        ckpt_w13_bf16 = torch.cat([ckpt_gate_bf16, ckpt_up_bf16], dim=0)  # (3072, 4096)

                        # Model dequantized weights for expert 0
                        model_w13_e0 = w13_bf16[0]  # (3072, 4096)
                        model_w2_e0 = w2_bf16[0]    # (4096, 1536)

                        # Compare (move checkpoint weights to GPU)
                        ckpt_w13_gpu = ckpt_w13_bf16.to(model_w13_e0.device)
                        ckpt_w2_gpu = ckpt_down_bf16.to(model_w2_e0.device)
                        w13_cos = cosine_similarity(model_w13_e0.unsqueeze(0), ckpt_w13_gpu.unsqueeze(0))
                        w2_cos = cosine_similarity(model_w2_e0.unsqueeze(0), ckpt_w2_gpu.unsqueeze(0))
                        w13_maxdiff = (model_w13_e0.float() - ckpt_w13_gpu.float()).abs().max().item()
                        w2_maxdiff = (model_w2_e0.float() - ckpt_w2_gpu.float()).abs().max().item()
                        print(f"  [L22 DIAG] Model w13[0] vs Ckpt w13 (BF16): cos={w13_cos:.8f}, max_diff={w13_maxdiff:.6e}")
                        print(f"  [L22 DIAG] Model w2[0]  vs Ckpt w2  (BF16): cos={w2_cos:.8f}, max_diff={w2_maxdiff:.6e}")

                        # Direct MoE computation using CHECKPOINT weights
                        ckpt_w13_e0_gpu = ckpt_w13_gpu.to(h_bf16.device)
                        ckpt_w2_e0_gpu = ckpt_w2_gpu.to(h_bf16.device)
                        h_local = h_bf16.to(h_bf16.device)
                        gu_ckpt = torch.mm(h_local, ckpt_w13_e0_gpu.T)
                        half_g = gu_ckpt.shape[-1] // 2
                        g_ckpt, u_ckpt = gu_ckpt[..., :half_g], gu_ckpt[..., half_g:]
                        act_ckpt = F.silu(g_ckpt) * u_ckpt
                        out_ckpt = torch.mm(act_ckpt, ckpt_w2_e0_gpu.T)
                        # Same using model weights
                        gu_model = torch.mm(h_local, model_w13_e0.T)
                        g_model, u_model = gu_model[..., :half_g], gu_model[..., half_g:]
                        act_model = F.silu(g_model) * u_model
                        out_model = torch.mm(act_model, model_w2_e0.T)
                        single_exp_cos = cosine_similarity(out_ckpt, out_model)
                        print(f"  [L22 DIAG] Expert 0 output (Ckpt w13/w2 vs Model w13/w2): cos={single_exp_cos:.8f}")
                        # Also compute Expert 0 output vs golden
                        if g22_mlp is not None:
                            print(f"  [L22 DIAG] Expert 0 Ckpt output norm: {out_ckpt.float().norm():.4f}")
                            print(f"  [L22 DIAG] Expert 0 Model output norm: {out_model.float().norm():.4f}")

                        # ── Gate weight and expert_bias comparison (for routing diagnosis) ──
                        print(f"  [L22 DIAG] --- Routing parameter check ---")
                        gate_ckpt = None; bias_ckpt = None
                        for fname in ckpt_files:
                            f = safetensors.safe_open(fname, framework='pt')
                            for k in f.keys():
                                if k == 'model.layers.22.mlp.router.gate.weight':
                                    gate_ckpt = f.get_tensor(k).float()
                                if k == 'model.layers.22.mlp.expert_bias':
                                    bias_ckpt = f.get_tensor(k).float()
                            if gate_ckpt is not None and bias_ckpt is not None:
                                break
                        if gate_ckpt is not None:
                            gate_model = gate_weight.float()
                            gate_cos = cosine_similarity(gate_model.unsqueeze(0).cpu(), gate_ckpt.unsqueeze(0))
                            gate_maxdiff = (gate_model.cpu() - gate_ckpt).abs().max().item()
                            bias_cos = cosine_similarity(expert_bias.float().unsqueeze(0).cpu(), bias_ckpt.unsqueeze(0))
                            bias_maxdiff = (expert_bias.float().cpu() - bias_ckpt).abs().max().item()
                            print(f"  [L22 DIAG] Gate weight (model vs ckpt): cos={gate_cos:.8f}, max_diff={gate_maxdiff:.6e}")
                            print(f"  [L22 DIAG] Expert bias (model vs ckpt): cos={bias_cos:.8f}, max_diff={bias_maxdiff:.6e}")
                            # Compute routing from GOLDEN input + checkpoint gate/bias
                            g22_input_dev = h_bf16.device
                            g22_pre = _golden[layer_idx]["04_post_attention_layernorm"].to(g22_input_dev).float()
                            golden_logits = torch.mm(g22_pre.reshape(-1, g22_pre.shape[-1]), gate_ckpt.T.to(g22_input_dev))
                            golden_scores = torch.sigmoid(golden_logits + bias_ckpt.unsqueeze(0).to(g22_input_dev))
                            golden_topk_vals, golden_topk_ids = torch.topk(golden_scores, 8, dim=-1)
                            g_set = set(golden_topk_ids[0].tolist())
                            m_set = set(topk_ids[0].tolist())
                            overlap = len(g_set & m_set)
                            # Also compute golden routing weights AFTER renormalize+scale (same formula as model)
                            g_renorm = golden_topk_vals / (golden_topk_vals.sum(dim=-1, keepdim=True) + 1e-20)
                            g_renorm_scaled = g_renorm * experts_layer.routed_scaling_factor
                            model_scores_raw = scores[0][topk_ids[0]]  # raw sigmoid values from model
                            m_renorm = model_scores_raw / (model_scores_raw.sum() + 1e-20)
                            m_renorm_scaled = m_renorm * experts_layer.routed_scaling_factor
                            print(f"  [L22 DIAG] Golden renorm+scaled weights: {g_renorm_scaled.tolist()}")
                            print(f"  [L22 DIAG] Model  renorm+scaled weights: {m_renorm_scaled.tolist()}")
                            w_cos2 = cosine_similarity(
                                g_renorm_scaled.unsqueeze(0).cpu(),
                                m_renorm_scaled.float().unsqueeze(0).cpu())
                            print(f"  [L22 DIAG] Renorm+scaled weight cos: {w_cos2:.8f}")
                            print(f"  [L22 DIAG] Golden routing (from ckpt gate+bias+golden_input): top-8 overlap with Model={overlap}/8")
                            if overlap == 8:
                                # Routing IDs match, but do the weights match?
                                # Sort both by expert ID for comparison
                                g_sorted_idx = golden_topk_ids[0].argsort()
                                m_sorted_idx = topk_ids[0].argsort()
                                g_weights_sorted = golden_topk_vals[0][g_sorted_idx]
                                m_weights_sorted = topk_weights[0][m_sorted_idx]
                                weight_cos = cosine_similarity(
                                    g_weights_sorted.unsqueeze(0).cpu(),
                                    m_weights_sorted.float().unsqueeze(0).cpu())
                                print(f"  [L22 DIAG] Top-8 routing weight cos: {weight_cos:.8f}")
                                print(f"  [L22 DIAG] Golden weights: {g_weights_sorted.tolist()}")
                                print(f"  [L22 DIAG] Model  weights: {m_weights_sorted.tolist()}")
                                print(f"  [L22 DIAG] Routed scaling factor: {experts_layer.routed_scaling_factor}")
                            if overlap < 8:
                                print(f"  [L22 DIAG] Golden top-8: {golden_topk_ids[0].tolist()}")
                                print(f"  [L22 DIAG] Model  top-8: {topk_ids[0].tolist()}")
                                only_g = g_set - m_set
                                only_m = m_set - g_set
                                print(f"  [L22 DIAG] Only in golden: {sorted(only_g)}")
                                print(f"  [L22 DIAG] Only in model:  {sorted(only_m)}")

            if rank == 0:
                moe_captured["gate_logits"] = router_logits.detach().cpu()
                moe_captured["scores_mi_style"] = scores.detach().cpu()
                moe_captured["is_tuple"] = shared_out is not None
                if shared_out is not None:
                    moe_captured["shared_out"] = shared_out.detach().cpu()
                moe_captured["routed_out"] = routed_out.detach().cpu()

            result = final_hidden_states.view(orig_shape)
            if rank == 0:
                moe_captured["output"] = result.detach().cpu()
            return result

        layer_module.mlp.forward = hooked_moe_forward
    else:
        # Layer 0: Dense MLP (HYV3FeedForward) — BF16 解量化避免 INT8 误差累积
        gate_up_proj = layer_module.mlp.gate_up_proj
        down_proj = layer_module.mlp.down_proj

        def hooked_dense_mlp_forward(hidden_states):
            h_bf16 = hidden_states.bfloat16()

            # gate_up_proj: dequantize + BF16 matmul (ColumnParallelLinear)
            gu_w_int8 = gate_up_proj.weight.data
            gu_w_scale = gate_up_proj.weight_scale.data
            gu_scale_adj = gu_w_scale
            while gu_scale_adj.dim() < gu_w_int8.dim():
                gu_scale_adj = gu_scale_adj.unsqueeze(-1)
            if gu_scale_adj.shape[0] != gu_w_int8.shape[0] and gu_scale_adj.shape[0] == gu_w_int8.shape[1]:
                gu_w_int8 = gu_w_int8.T
            gu_w_bf16 = (gu_w_int8.float() * gu_scale_adj.float()).bfloat16()

            # golden DenseMLP.forward 同构：gate/up 独立投影 + F.linear
            half = gu_w_bf16.shape[0] // 2
            gate_out = F.silu(F.linear(h_bf16, gu_w_bf16[:half]))
            up_out = F.linear(h_bf16, gu_w_bf16[half:])
            act_out = gate_out * up_out

            # down_proj: dequantize + BF16 matmul + all-reduce (RowParallelLinear)
            d_w_int8 = down_proj.weight.data
            d_w_scale = down_proj.weight_scale.data
            d_scale_adj = d_w_scale
            while d_scale_adj.dim() < d_w_int8.dim():
                d_scale_adj = d_scale_adj.unsqueeze(-1)
            if d_scale_adj.shape[0] != d_w_int8.shape[0] and d_scale_adj.shape[0] == d_w_int8.shape[1]:
                d_w_int8 = d_w_int8.T
            d_w_bf16 = (d_w_int8.float() * d_scale_adj.float()).bfloat16()

            output = F.linear(act_out, d_w_bf16)
            dist.all_reduce(output, group=dist.group.WORLD)
            return output

        layer_module.mlp.forward = hooked_dense_mlp_forward

    def hooked_forward(positions, hidden_states, residual, idx=-1):
        # MetaInfer 的 00_input 是「残差合并后」的值 ——
        # 即上一层的 06_output（= hidden_states + residual）
        # 对于第 0 层 residual=None，00_input 就是嵌入输出本身。
        if rank == 0:
            if residual is not None:
                captured[layer_idx]["00_input"] = (hidden_states + residual).detach().cpu()
            else:
                captured[layer_idx]["00_input"] = hidden_states.detach().cpu()

        if residual is None:
            residual = hidden_states
            hidden_states = orig_input_ln(hidden_states)
        else:
            hidden_states, residual = orig_input_ln(hidden_states, residual)

        if rank == 0:
            captured[layer_idx]["01_input_layernorm"] = hidden_states.detach().cpu()

        # 手动注意力 — 绕过 vLLM 的 Attention 层，后者需要 ForwardContext
        hidden_states = manual_attention(hidden_states, positions)

        if rank == 0:
            captured[layer_idx]["02_attention_out"] = hidden_states.detach().cpu()

        attn_out = hidden_states
        attn_residual = attn_out + residual
        if rank == 0:
            captured[layer_idx]["03_attention_residual"] = attn_residual.detach().cpu()

        hidden_states, residual = orig_post_ln(attn_out, residual)

        if rank == 0:
            captured[layer_idx]["04_post_attention_layernorm"] = (
                hidden_states.detach().cpu()
            )

        # 保存 MLP 输入用于后续诊断
        mlp_input = hidden_states
        hidden_states = orig_mlp(hidden_states)

        if rank == 0:
            captured[layer_idx]["05_mlp_out"] = hidden_states.detach().cpu()
            output = hidden_states + residual
            captured[layer_idx]["06_output"] = output.detach().cpu()

            # ── bitwise 定位：每个 dump point 与 golden 逐位对比，找第一个断裂点 ──
            # 注意：torch.equal 对形状不同的张量直接返回 False（golden 是
            # [1, seq, hidden]，captured 是 [seq, hidden]），必须先 reshape 对齐。
            if _golden is not None and layer_idx in _golden:
                for pt in DUMP_POINTS:
                    g_pt = _golden[layer_idx][pt]
                    m_pt = captured[layer_idx][pt]
                    g_bf16 = g_pt.bfloat16().reshape(-1, g_pt.shape[-1])
                    m_bf16 = m_pt.bfloat16().reshape(-1, m_pt.shape[-1])
                    same = torch.equal(g_bf16, m_bf16)
                    cos_pt = cosine_similarity(m_pt, g_pt)
                    if (not same and layer_idx < 6) or layer_idx == 0:
                        print(f"  [BITWISE L{layer_idx}] {pt}: same={same} cos={cos_pt:.9f}")

                # ── L0 embedding 诊断：checkpoint 重算 vs vLLM captured vs golden ──
                # 三方对比定位 embedding 输出 bit 差异来源：
                #   captured == ref → vLLM 路径精确，差异来自 golden 侧
                #   ref == golden → vLLM 路径（mask/compress/all_reduce）引入 bit 差异
                if layer_idx == 0 and _ref_input_ids is not None:
                    global _emb_weight_ref
                    if _emb_weight_ref is None:
                        for fname in ckpt_file_list:
                            f = safetensors.safe_open(fname, framework='pt')
                            if 'model.embed_tokens.weight' in f.keys():
                                _emb_weight_ref = f.get_tensor(
                                    'model.embed_tokens.weight').bfloat16()
                                break
                        print(f"  [EMB DIAG] checkpoint embed_tokens.weight: "
                              f"{_emb_weight_ref.shape} {_emb_weight_ref.dtype}")
                    ref_emb = F.embedding(_ref_input_ids, _emb_weight_ref)  # [seq, 4096] CPU
                    m0 = captured[0]["00_input"]  # [seq, 4096] BF16 CPU
                    g0 = _golden[0]["00_input"]  # [1, seq, 4096] BF16 CPU
                    g0_flat = g0.reshape(-1, ref_emb.shape[-1])
                    print(f"  [EMB DIAG] captured vs ref:      "
                          f"cos={cosine_similarity(m0, ref_emb):.9f} "
                          f"bitwise={torch.equal(m0.bfloat16(), ref_emb.bfloat16())}")
                    print(f"  [EMB DIAG] golden   vs ref:      "
                          f"cos={cosine_similarity(g0_flat, ref_emb):.9f} "
                          f"bitwise={torch.equal(g0_flat.bfloat16(), ref_emb.bfloat16())}")
                    print(f"  [EMB DIAG] captured vs golden:   "
                          f"cos={cosine_similarity(m0, g0_flat):.9f} "
                          f"bitwise={torch.equal(m0.bfloat16(), g0_flat.bfloat16())}")
                    mf = m0.bfloat16().float()
                    gf = g0_flat.bfloat16().float()
                    neq = (mf != gf)
                    print(f"  [EMB DIAG] captured-vs-golden diff elements: "
                          f"{neq.sum().item()}/{neq.numel()}")
                    if neq.any():
                        rows = neq.any(dim=-1)
                        print(f"  [EMB DIAG] diff rows: {rows.sum().item()}/{rows.numel()}")
                        for r in rows.nonzero().flatten()[:6]:
                            n = neq[r].sum().item()
                            md = (mf[r] - gf[r]).abs().max().item()
                            print(f"  [EMB DIAG]   token {r.item()}: {n} diffs, "
                                  f"max_diff={md:.6e}")

            # ── 保存 MoE 内部诊断数据 ──
            captured[layer_idx]["_moe"] = dict(moe_captured)
            moe_captured.clear()

        # 若为最后层，保存输出以供后续批次继续
        if layer_idx == NUM_LAYERS - 1:
            global _saved_hidden_states
            _saved_hidden_states = (hidden_states.detach().cpu(), residual.detach().cpu())

        return hidden_states, residual

    layer_module.forward = hooked_forward


def load_golden():
    """加载 MetaInfer golden dump"""
    golden = {}
    for layer_idx in range(NUM_LAYERS):
        golden[layer_idx] = {}
        layer_dir = os.path.join(GOLDEN_DIR, f"layer_{layer_idx:03d}")
        for point in DUMP_POINTS:
            pt_file = os.path.join(layer_dir, f"{point}.pt")
            golden[layer_idx][point] = torch.load(pt_file, map_location="cpu")
    return golden


def cosine_similarity(a, b):
    a_f = a.float().reshape(-1, a.shape[-1])
    b_f = b.float().reshape(-1, b.shape[-1])
    a_n = torch.nn.functional.normalize(a_f, dim=-1)
    b_n = torch.nn.functional.normalize(b_f, dim=-1)
    return (a_n * b_n).sum(-1).mean().item()


def main():
    global _golden

    # ── 初始化分布式 ──
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=f"env://",
        local_rank=local_rank,
        backend="nccl",
    )

    # ── 配置 ──
    model_config = ModelConfig(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
        max_model_len=128,
        enforce_eager=True,  # 禁用 torch.compile，避免 Triton INT8 kernel 兼容性问题
    )

    # ── 关键: 覆盖 num_hidden_layers，只创建前 10 层 ──
    # 完整模型 280GB 专家权重无法放入 4×64GB GPU
    # 逐层加载策略 (MetaInfer 使用 GPU_EXPERT_CACHE_LAYERS=0 实现)
    # 验证只需要前 10 层，覆盖后每 GPU 仅需 ~12GB
    model_config.hf_config.num_hidden_layers = NUM_LAYERS
    model_config.hf_config.num_nextn_predict_layers = 0

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.90,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=world_size,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        is_moe_model=True,
    )
    load_config = LoadConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        load_config=load_config,
    )

    loader = get_model_loader(load_config)

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
        )

        if rank == 0:
            print("=" * 70)
            print("HY3 vLLM vs MetaInfer Golden Dump 验证 (TP=4)")
            print(f"Prompt: '{PROMPT}' (3 tokens)")
            print(f"Layers: 0 ~ {NUM_LAYERS-1} (first {NUM_LAYERS} of 80)")
            print("=" * 70)
            print("\n[1/3] Loading model (all 80 layers)...")

        model = loader.load_model(
            vllm_config=vllm_config, model_config=model_config
        )
        model.eval()

        # 在所有 rank 安装 hooks，但仅在 rank 0 收集 hidden states。
        # 所有 rank 必须使用相同的 hooked forward（手动注意力），
        # 因为原始 forward 中的 vLLM Attention 层需要 ForwardContext。
        if rank == 0:
            print(f"  Installing hooks on first {NUM_LAYERS} layers (all ranks)")
        for idx, layer in enumerate(model.model.layers):
            if isinstance(layer, HYV3DecoderLayer) and idx < NUM_LAYERS:
                if rank == 0:
                    captured[idx] = {}
                make_hooked_forward(idx, layer)

        # 所有 rank 执行推理之前，rank 0 预先加载 golden 数据供诊断用
        if rank == 0:
            print("\n[2/3] Loading golden dump for in-forward diagnostics...")
            _golden = load_golden()
            print(f"  Loaded {len(_golden)} layers of golden data")

            # ── 探针：模型加载完成后，fp32 mm 数值行为是否已改变 ──
            print("\n[PROBE] post-load fp32 mm behavior...", flush=True)
            print(f"  allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)
            try:
                print(f"  fp32_precision={torch.backends.cuda.matmul.fp32_precision}", flush=True)
            except AttributeError:
                print("  fp32_precision=<no attr>", flush=True)
            for k, v in sorted(os.environ.items()):
                if any(s in k.upper() for s in ("BLAS", "HIP", "TF32", "FP32", "MKL", "DETERMIN")):
                    print(f"  env {k}={v}", flush=True)
            pg04 = torch.load(os.path.join(GOLDEN_DIR, "layer_001", "04_post_attention_layernorm.pt"),
                              map_location=f"cuda:{local_rank}").bfloat16().reshape(-1, 4096)
            pg_gate = None
            pg_bias = None
            for fname in sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors")):
                pf = safetensors.safe_open(fname, framework='pt')
                pks = pf.keys()
                if 'model.layers.1.mlp.router.gate.weight' in pks:
                    pg_gate = pf.get_tensor('model.layers.1.mlp.router.gate.weight').to(f"cuda:{local_rank}")
                if 'model.layers.1.mlp.expert_bias' in pks:
                    pg_bias = pf.get_tensor('model.layers.1.mlp.expert_bias').to(f"cuda:{local_rank}")
            pbt = torch.mm(pg04.float(), pg_gate.float().T) + pg_bias.float()
            ppt = torch.cat([F.linear(pg04[t:t+1].float(), pg_gate.float()) + pg_bias.float()
                             for t in range(pg04.shape[0])], 0)
            print(f"  [PROBE] post-load pt vs batch max_diff={(pbt-ppt).abs().max().item():.6e}", flush=True)
            pbt2 = torch.mm(pg04.float(), pg_gate.float().T) + pg_bias.float()
            print(f"  [PROBE] post-load batch 两次 max_diff={(pbt-pbt2).abs().max().item():.6e}", flush=True)
            ppt2 = torch.cat([F.linear(pg04[t:t+1].float(), pg_gate.float()) + pg_bias.float()
                              for t in range(pg04.shape[0])], 0)
            print(f"  [PROBE] post-load per-token 两次 max_diff={(ppt-ppt2).abs().max().item():.6e}", flush=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pbt_ac = torch.mm(pg04.float(), pg_gate.float().T) + pg_bias.float()
                ppt_ac = torch.cat([F.linear(pg04[t:t+1].float(), pg_gate.float()) + pg_bias.float()
                                    for t in range(pg04.shape[0])], 0)
            print(f"  [PROBE] post-load autocast内 pt vs batch max_diff={(pbt_ac-ppt_ac).abs().max().item():.6e}", flush=True)
            print(f"  [PROBE] post-load plain vs autocast batch max_diff={(pbt-pbt_ac).abs().max().item():.6e}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(f"cuda:{local_rank}")
        # Flatten to 1D [num_tokens] — model expects 2D hidden states downstream,
        # and a 2D input_ids [1, seq] would produce 3D embeddings that break linear ops.
        input_ids = input_ids.view(-1)
        if rank == 0:
            global _ref_input_ids
            _ref_input_ids = input_ids.detach().cpu()
        positions = torch.arange(input_ids.shape[0], device=f"cuda:{local_rank}")

        if rank == 0:
            print("\n[3/4] Running prefill...")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with set_forward_context(
                    attn_metadata=None,
                    vllm_config=vllm_config,
                    num_tokens=int(input_ids.shape[0]),
                ):
                    _ = model(input_ids=input_ids, positions=positions)

        dist.barrier()

    # 仅 rank 0 做对比
    if rank == 0:
        print(f"  Done, captured {len(captured)} layers")

        # 保存中间状态供后续批次使用
        if _saved_hidden_states is not None and NUM_LAYERS < 80:
            hidden_final, residual_final = _saved_hidden_states
            torch.save(hidden_final, "intermediate_hidden_states.pt")
            torch.save(residual_final, "intermediate_residual.pt")
            print(f"  Saved intermediate states for layer {NUM_LAYERS} -> 80 continuation")
            print(f"    hidden_states: shape={hidden_final.shape}, norm={hidden_final.float().norm():.2f}")
            print(f"    residual:      shape={residual_final.shape}, norm={residual_final.float().norm():.2f}")

        print("\n[4/4] Computing similarity with golden dump...")
        golden = _golden

        print("\n" + "=" * 70)
        print("Cosine Similarity Results (per layer / per dump point)")
        print("=" * 70)

        hdr = f"{'Lyr':>4} |" + "|".join(f"{p[3:]:>8}" for p in DUMP_POINTS) + "| Mean"
        print(hdr)
        print("-" * len(hdr))

        all_sims = []
        for lyr in range(NUM_LAYERS):
            row_vals = []
            row_str = f" {lyr:03d} |"
            for pt in DUMP_POINTS:
                if pt in captured[lyr] and pt in golden[lyr]:
                    sim = cosine_similarity(captured[lyr][pt], golden[lyr][pt])
                    row_vals.append(sim)
                    all_sims.append(sim)
                    row_str += f" {sim:.6f}"
                else:
                    row_str += f" {'N/A':>8}"
            if row_vals:
                mean = sum(row_vals) / len(row_vals)
                row_str += f" | {mean:.6f}"
            print(row_str)

        # ── 跨层验证：残差传递方式差异 ──
        print("\n" + "=" * 70)
        print("Cross-layer consistency check (residual scheme diagnosis)")
        print("=" * 70)
        for lyr in range(NUM_LAYERS - 1):
            # vLLM 中，第 N 层的 05_mlp_out 应该等于第 N+1 层的 00_input
            # （因为残差不加在 hidden_states 中传递）
            if "05_mlp_out" in captured[lyr] and "00_input" in captured[lyr + 1]:
                sim_vllm_internal = cosine_similarity(
                    captured[lyr]["05_mlp_out"], captured[lyr + 1]["00_input"]
                )
                print(f"  vLLM L{lyr}/05_mlp_out vs L{lyr+1}/00_input: {sim_vllm_internal:.6f} "
                      f"(should be 1.0 — same tensor)")
            # 如果 MetaInfer 在层间传递时已经加了残差，
            # 那么 vLLM 第 N 层 06_output (= MLP+residual) 与
            # golden 第 N+1 层 00_input 的相似度应该较高
            if "06_output" in captured[lyr] and "00_input" in golden.get(lyr + 1, {}):
                sim_cross = cosine_similarity(
                    captured[lyr]["06_output"], golden[lyr + 1]["00_input"]
                )
                print(f"  vLLM L{lyr}/06_output vs golden L{lyr+1}/00_input: {sim_cross:.6f} "
                      f"(high → MetaInfer adds residual between layers)")
            if "05_mlp_out" in captured[lyr] and "00_input" in golden.get(lyr + 1, {}):
                sim_cross2 = cosine_similarity(
                    captured[lyr]["05_mlp_out"], golden[lyr + 1]["00_input"]
                )
                print(f"  vLLM L{lyr}/05_mlp_out vs golden L{lyr+1}/00_input: {sim_cross2:.6f}")

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        if all_sims:
            print(f"  Total points:    {len(all_sims)}")
            print(f"  Mean cos_sim:    {sum(all_sims)/len(all_sims):.6f}")
            print(f"  Min cos_sim:     {min(all_sims):.6f}")
            print(f"  Max cos_sim:     {max(all_sims):.6f}")
            lt99 = sum(1 for s in all_sims if s < 0.99)
            lt95 = sum(1 for s in all_sims if s < 0.95)
            lt90 = sum(1 for s in all_sims if s < 0.90)
            print(f"  Points < 0.99:   {lt99}")
            print(f"  Points < 0.95:   {lt95}")
            print(f"  Points < 0.90:   {lt90}")
        else:
            print("  No matching data points found!")

        # ── MoE 诊断：shared vs routed（已手动 all-reduce）──
        print("\n" + "=" * 70)
        print("MoE Diagnostic (with manual all-reduce fix)")
        print("=" * 70)
        for lyr in range(NUM_LAYERS):
            moe = captured[lyr].get("_moe", {})
            if not moe:
                continue
            print(f"\nLayer {lyr}:")
            print(f"  Shared expert active: {moe.get('is_tuple', 'N/A')}")
            if moe.get('is_tuple'):
                s = moe.get('shared_out')
                r = moe.get('routed_out')
                combined = moe['output']
                g = golden.get(lyr, {}).get('05_mlp_out')
                if g is not None and s is not None and r is not None:
                    print(f"  shared_out   vs golden: {cosine_similarity(s, g):.6f}")
                    print(f"  routed_out   vs golden: {cosine_similarity(r, g):.6f}")
                    print(f"  combined     vs golden: {cosine_similarity(combined, g):.6f}")
                    print(f"  Shared norm: {s.float().norm():.2f}, Routed norm: {r.float().norm():.2f}")

        # ── Gate logit 诊断：验证 MetaInfer 风格路由 ──
        print("\n" + "=" * 70)
        print("MoE Gate Routing Diagnostic (MetaInfer-style routing)")
        print("=" * 70)
        for lyr in range(NUM_LAYERS):
            moe = captured[lyr].get("_moe", {})
            logits = moe.get("gate_logits")
            if logits is None:
                continue

            # hook 中已使用 MetaInfer 风格路由
            scores = moe.get("scores_mi_style")
            if scores is None:
                continue

            # 对比 hook 内计算的路由 vs 离线 MetaInfer 计算
            layer = model.model.layers[lyr]
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
                continue
            gate_w = layer.mlp.gate.weight.data.float().cpu()
            gate_bias = layer.mlp.expert_bias.data.float().cpu()
            gate_input = captured[lyr]["04_post_attention_layernorm"]
            mi_logits = torch.mm(gate_input.float().cpu().reshape(-1, gate_input.shape[-1]), gate_w.T) + gate_bias
            mi_scores = torch.sigmoid(mi_logits)

            hook_scores = scores.float().cpu()
            score_diff = (hook_scores - mi_scores).abs().max().item()

            mi_topk = torch.topk(mi_scores, 8, dim=-1)[1]
            hook_topk = torch.topk(hook_scores, 8, dim=-1)[1]
            mi_set = set(mi_topk[0].tolist())
            hook_set = set(hook_topk[0].tolist())
            overlap = len(mi_set & hook_set)
            print(f"  Layer {lyr}: score max diff={score_diff:.2e}, top-8 overlap={overlap}/8")

    dist.barrier()


if __name__ == "__main__":
    main()
