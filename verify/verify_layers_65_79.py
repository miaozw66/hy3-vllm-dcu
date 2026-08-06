"""
层 65-79 纯手动验证（不依赖 vLLM model 类）
从 safetensors 直接读取权重、手动解量化、手动计算注意力和 MoE

用法: python3 verify_layers_65_79.py
"""
import glob
import os

import safetensors
import torch
import torch.nn.functional as F


# ── 配置 ──────────────────────────────────────────────
MODEL_DIR = "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master"
GOLDEN_DIR = "/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/golden_dump"
PROMPT = "中国的首都是"

START_LAYER = 65
END_LAYER = 80  # exclusive
NUM_LAYERS_TOTAL = 80

DUMP_POINTS = [
    "00_input", "01_input_layernorm", "02_attention_out",
    "03_attention_residual", "04_post_attention_layernorm",
    "05_mlp_out", "06_output",
]

# 模型参数
HIDDEN_SIZE = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
Q_SIZE = NUM_HEADS * HEAD_DIM  # 8192
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
MOE_INTERMEDIATE = 1536
NUM_EXPERTS = 192
TOP_K = 8
ROUTED_SCALING = 2.826
RENORMALIZE = True  # norm_topk_prob
USE_SIGMOID = True
USE_EXPERT_BIAS = True

# RoPE
ROPE_THETA = 11158840.0
MAX_POS = 262144

captured = {}
_golden = None


def cosine_similarity(a, b):
    a_f = a.float().reshape(-1, a.shape[-1])
    b_f = b.float().reshape(-1, b.shape[-1])
    a_n = F.normalize(a_f, dim=-1)
    b_n = F.normalize(b_f, dim=-1)
    return (a_n * b_n).sum(-1).mean().item()


def load_golden():
    golden = {}
    for layer_idx in range(START_LAYER, END_LAYER):
        golden[layer_idx] = {}
        layer_dir = os.path.join(GOLDEN_DIR, f"layer_{layer_idx:03d}")
        for point in DUMP_POINTS:
            pt_file = os.path.join(layer_dir, f"{point}.pt")
            golden[layer_idx][point] = torch.load(pt_file, map_location="cpu")
    return golden


def build_layer_to_file_map():
    files = sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors"))
    layer_to_file = {}
    for fname in files:
        f = safetensors.safe_open(fname, framework='pt')
        keys = list(f.keys())
        for k in keys:
            if 'layers.' in k:
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_to_file[int(parts[i + 1])] = fname
                        except ValueError:
                            pass
                        break
                break
    return layer_to_file


def dequant_weight(w_int, w_scale):
    """INT8 → BF16 解量化"""
    scale_adj = w_scale
    while scale_adj.dim() < w_int.dim():
        scale_adj = scale_adj.unsqueeze(-1)
    if scale_adj.shape[0] != w_int.shape[0] and scale_adj.shape[0] == w_int.shape[1]:
        w_int = w_int.T
    return (w_int.float() * scale_adj.float()).bfloat16()


def precompute_rope(positions, dim, theta=ROPE_THETA):
    """预计算 RoPE cos/sin"""
    n = positions.shape[0]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = freqs.to(positions.device)
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)
    return angles.cos(), angles.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """对 (seq, num_heads, head_dim) 的 x 应用标准 RoPE（half-split 约定）
    cos, sin: (seq, head_dim//2)"""
    # Standard RoPE convention: half-split
    cos = cos.unsqueeze(1)  # (seq, 1, head_dim//2)
    sin = sin.unsqueeze(1)  # (seq, 1, head_dim//2)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rms_norm(x, weight, eps=1e-5):
    """RMS LayerNorm"""
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + eps)
    return (x_f / rms * weight.float()).to(x.dtype)


# ── 全局权重缓存 ──
_weight_cache = {}  # {(layer_idx, key): tensor_on_cpu}


def preload_weights(layer_range, device_str='cpu'):
    """预加载指定层范围的所有权重到 CPU 内存缓存"""
    global _weight_cache
    files = sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors"))
    layer_prefixes = [f"model.layers.{i}." for i in layer_range]

    for fname in files:
        f = safetensors.safe_open(fname, framework='pt')
        for key in f.keys():
            for prefix in layer_prefixes:
                if key.startswith(prefix):
                    # Store on CPU to avoid GPU OOM
                    _weight_cache[key] = f.get_tensor(key)
                    break

    print(f"  Preloaded {len(_weight_cache)} weight tensors for layers "
          f"{layer_range[0]}-{layer_range[-1]} into CPU cache")


def _get_layer_weight(layer_idx, weight_key):
    """从缓存获取权重并移到 GPU"""
    full_key = f"model.layers.{layer_idx}.{weight_key}"
    t = _weight_cache.get(full_key)
    if t is None:
        raise KeyError(f"Weight not found: {full_key}")
    return t


def dequant_weight_from_cache(layer_idx, weight_key, device):
    """从缓存获取 INT8 权重 + scale 并解量化"""
    w_int = _get_layer_weight(layer_idx, weight_key)
    w_scale = _get_layer_weight(layer_idx, weight_key + "_scale")
    scale_adj = w_scale
    while scale_adj.dim() < w_int.dim():
        scale_adj = scale_adj.unsqueeze(-1)
    if scale_adj.shape[0] != w_int.shape[0] and scale_adj.shape[0] == w_int.shape[1]:
        w_int = w_int.T
    w_bf16 = (w_int.float() * scale_adj.float()).bfloat16()
    return w_bf16.to(device)


def load_attention_weights(layer_idx, device):
    """加载并解量化注意力的所有权重"""
    return {
        'q_proj': dequant_weight_from_cache(layer_idx, 'self_attn.q_proj.weight', device),
        'k_proj': dequant_weight_from_cache(layer_idx, 'self_attn.k_proj.weight', device),
        'v_proj': dequant_weight_from_cache(layer_idx, 'self_attn.v_proj.weight', device),
        'o_proj': dequant_weight_from_cache(layer_idx, 'self_attn.o_proj.weight', device),
        'q_norm': _get_layer_weight(layer_idx, 'self_attn.q_norm.weight').to(device),
        'k_norm': _get_layer_weight(layer_idx, 'self_attn.k_norm.weight').to(device),
    }


def load_norm_weights(layer_idx, device):
    return {
        'input_ln': _get_layer_weight(layer_idx, 'input_layernorm.weight').to(device),
        'post_ln': _get_layer_weight(layer_idx, 'post_attention_layernorm.weight').to(device),
    }


def load_shared_expert_weights(layer_idx, device):
    return {
        'gate_proj': dequant_weight_from_cache(layer_idx, 'mlp.shared_mlp.gate_proj.weight', device),
        'up_proj': dequant_weight_from_cache(layer_idx, 'mlp.shared_mlp.up_proj.weight', device),
        'down_proj': dequant_weight_from_cache(layer_idx, 'mlp.shared_mlp.down_proj.weight', device),
    }


def load_gate_weights(layer_idx, device):
    return {
        'gate': _get_layer_weight(layer_idx, 'mlp.router.gate.weight').float().to(device),
        'bias': _get_layer_weight(layer_idx, 'mlp.expert_bias').float().to(device),
    }


def load_expert_weights(layer_idx, expert_id, device):
    """加载单个专家的权重并解量化"""
    return {
        'gate_proj': dequant_weight_from_cache(layer_idx, f'mlp.experts.{expert_id}.gate_proj.weight', device),
        'up_proj': dequant_weight_from_cache(layer_idx, f'mlp.experts.{expert_id}.up_proj.weight', device),
        'down_proj': dequant_weight_from_cache(layer_idx, f'mlp.experts.{expert_id}.down_proj.weight', device),
    }


def manual_attention(hidden_states, attn_w, positions, cos, sin):
    """手动计算注意力"""
    h_bf16 = hidden_states.bfloat16()

    # QKV 投影
    q = torch.mm(h_bf16, attn_w['q_proj'].T)  # (seq, 8192)
    k = torch.mm(h_bf16, attn_w['k_proj'].T)  # (seq, 1024)
    v = torch.mm(h_bf16, attn_w['v_proj'].T)  # (seq, 1024)

    # Reshape + QK norm
    q = q.view(-1, NUM_HEADS, HEAD_DIM)
    k = k.view(-1, NUM_KV_HEADS, HEAD_DIM)

    # QK norm: RMSNorm (not L2 normalize)
    q_rms = torch.sqrt(torch.mean(q.float() * q.float(), dim=-1, keepdim=True) + 1e-5)
    k_rms = torch.sqrt(torch.mean(k.float() * k.float(), dim=-1, keepdim=True) + 1e-5)
    q = (q.float() / q_rms) * attn_w['q_norm'].float()
    k = (k.float() / k_rms) * attn_w['k_norm'].float()

    # RoPE
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    q, k = q.bfloat16(), k.bfloat16()

    # GQA: expand K/V
    if NUM_KV_HEADS < NUM_HEADS:
        n_groups = NUM_HEADS // NUM_KV_HEADS
        k = k.unsqueeze(2).expand(-1, NUM_KV_HEADS, n_groups, HEAD_DIM)
        k = k.reshape(-1, NUM_HEADS, HEAD_DIM)
        v = v.view(-1, NUM_KV_HEADS, HEAD_DIM)
        v = v.unsqueeze(2).expand(-1, NUM_KV_HEADS, n_groups, HEAD_DIM)
        v = v.reshape(-1, NUM_HEADS, HEAD_DIM)

    # SDPA
    q_sdpa = q.permute(1, 0, 2).unsqueeze(0)  # (1, heads, seq, head_dim)
    k_sdpa = k.permute(1, 0, 2).unsqueeze(0)
    v_sdpa = v.permute(1, 0, 2).unsqueeze(0)

    scaling = HEAD_DIM ** -0.5
    attn_out = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, scale=scaling, is_causal=True
    )
    attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(-1, Q_SIZE)

    # O-proj
    output = torch.mm(attn_out.bfloat16(), attn_w['o_proj'].T)
    return output


def compute_moe(hidden_states, shared_w, gate_w, layer_idx, device):
    """手动计算 MoE 层"""
    hidden_flat = hidden_states.view(-1, HIDDEN_SIZE)
    h_bf16 = hidden_flat.bfloat16()
    h_f32 = hidden_flat.float()

    # 路由: sigmoid(logits + bias) → topk → renormalize → scale
    router_logits = torch.mm(h_f32, gate_w['gate'].T)
    if USE_EXPERT_BIAS:
        scores = torch.sigmoid(router_logits + gate_w['bias'].unsqueeze(0))
    else:
        scores = torch.sigmoid(router_logits)

    raw_weights, raw_ids = torch.topk(scores, TOP_K, dim=-1, sorted=True)
    if RENORMALIZE:
        topk_weights = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        topk_weights = raw_weights
    topk_weights = topk_weights * ROUTED_SCALING

    # Shared expert: merge gate_proj + up_proj → gate_up_proj
    shared_out = None
    if shared_w is not None:
        gu_w = torch.cat([shared_w['gate_proj'], shared_w['up_proj']], dim=0)  # (3072, 4096)
        gu_out = torch.mm(h_bf16, gu_w.T)
        half = gu_out.shape[-1] // 2
        act = F.silu(gu_out[..., :half]) * gu_out[..., half:]
        shared_out = torch.mm(act, shared_w['down_proj'].T)

    # Routed experts: 收集唯一的专家 ID，加载权重，计算
    unique_experts = set(raw_ids.reshape(-1).tolist())
    routed_out = torch.zeros(h_bf16.shape[0], HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    for eid in unique_experts:
        e_weights = load_expert_weights(layer_idx, eid, device)
        # 合并 gate_proj + up_proj 为 w13
        w13 = torch.cat([e_weights['gate_proj'], e_weights['up_proj']], dim=0)  # (3072, 4096)
        w2 = e_weights['down_proj']  # (4096, 1536)

        # 找到使用该专家的 token 索引
        mask = (raw_ids == eid)  # (seq, topk)
        token_indices, k_indices = mask.nonzero(as_tuple=True)

        for ti, ki in zip(token_indices.tolist(), k_indices.tolist()):
            h_t = h_bf16[ti:ti+1]
            weight = topk_weights[ti, ki]
            gu_e = torch.mm(h_t, w13.T)
            half_e = gu_e.shape[-1] // 2
            gate_e = gu_e[..., :half_e]
            up_e = gu_e[..., half_e:]
            act_e = F.silu(gate_e) * up_e
            out_e = torch.mm(act_e, w2.T).squeeze(0)
            routed_out[ti] += weight * out_e

        del e_weights, w13, w2

    if shared_out is not None:
        return shared_out + routed_out
    return routed_out


def manual_mlp_dense(hidden_states, shared_w):
    """Dense MLP (not used for layers 65-79, here for completeness)"""
    h_bf16 = hidden_states.bfloat16()
    gu_w = torch.cat([shared_w['gate_proj'], shared_w['up_proj']], dim=0)
    gu_out = torch.mm(h_bf16, gu_w.T)
    half = gu_out.shape[-1] // 2
    act = F.silu(gu_out[..., :half]) * gu_out[..., half:]
    return torch.mm(act, shared_w['down_proj'].T)


def main():
    global _golden

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("=" * 70)
    print(f"HY3 Layers {START_LAYER}-{END_LAYER-1} Manual Verification (no vLLM)")
    print(f"Prompt: '{PROMPT}' (3 tokens)")
    print("=" * 70)

    # ── 加载中间状态 ──
    print(f"\n[1/3] Loading intermediate states from 65-layer run...")
    hidden_states = torch.load("intermediate_hidden_states.pt", map_location=device)
    residual = torch.load("intermediate_residual.pt", map_location=device)
    print(f"  hidden_states: shape={hidden_states.shape}, norm={hidden_states.float().norm():.2f}")
    print(f"  residual:      shape={residual.shape}, norm={residual.float().norm():.2f}")

    # ── Tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")[0]
    positions = torch.arange(input_ids.shape[0], device=device)
    num_tokens = input_ids.shape[0]

    # ── 预计算 RoPE ──
    cos, sin = precompute_rope(positions, HEAD_DIM, ROPE_THETA)
    cos = cos.to(device)
    sin = sin.to(device)

    # ── 加载 golden ──
    print("\n[2/3] Loading golden dump...")
    _golden = load_golden()
    print(f"  Loaded layers {START_LAYER}-{END_LAYER-1} of golden data")

    # ── 预加载所有权重到 CPU 缓存 ──
    print(f"\n[3/4] Preloading weights for layers {START_LAYER}-{END_LAYER-1}...")
    preload_weights(range(START_LAYER, END_LAYER))

    # ── 逐层计算 ──
    print(f"\n[4/4] Computing layers {START_LAYER}-{END_LAYER-1} manually...")

    for lyr in range(START_LAYER, END_LAYER):
        print(f"\n  Layer {lyr}...", end=" ", flush=True)

        captured[lyr] = {}

        # 00: input (before layer)
        captured[lyr]["00_input"] = (hidden_states + residual).detach().cpu()

        # input_layernorm
        norm_w = load_norm_weights(lyr, device)
        ln_out = rms_norm(hidden_states + residual, norm_w['input_ln'])
        captured[lyr]["01_input_layernorm"] = ln_out.detach().cpu()

        # Residual split (MetaInfer style)
        residual = hidden_states + residual
        hidden_states = ln_out

        # Attention
        attn_w = load_attention_weights(lyr, device)
        attn_out = manual_attention(hidden_states, attn_w, positions, cos, sin)
        captured[lyr]["02_attention_out"] = attn_out.detach().cpu()
        captured[lyr]["03_attention_residual"] = (attn_out + residual).detach().cpu()
        del attn_w

        # post_attention_layernorm
        ln2_out = rms_norm(attn_out + residual, norm_w['post_ln'])
        captured[lyr]["04_post_attention_layernorm"] = ln2_out.detach().cpu()

        # Residual split
        residual = attn_out + residual

        # MLP (MoE for layers 1+, layer 0 is dense)
        if lyr > 0:  # All layers 65-79 are MoE
            shared_w = load_shared_expert_weights(lyr, device)
            gate_w = load_gate_weights(lyr, device)
            mlp_out = compute_moe(ln2_out, shared_w, gate_w, lyr, device)
            del shared_w, gate_w
        else:
            shared_w = load_shared_expert_weights(lyr, device)
            mlp_out = manual_mlp_dense(ln2_out, shared_w)
            del shared_w

        captured[lyr]["05_mlp_out"] = mlp_out.detach().cpu()
        output = mlp_out + residual
        captured[lyr]["06_output"] = output.detach().cpu()

        # 更新 hidden_states 为 MLP 输出（residual 已在 post_ln 后正确设置，无需更改）
        hidden_states = mlp_out

        del norm_w, mlp_out
        torch.cuda.empty_cache()

        print(f"done. h={hidden_states.float().norm():.2f} r={residual.float().norm():.2f}", flush=True)

    # ── 对比 golden ──
    print("\n" + "=" * 70)
    print(f"Cosine Similarity Results (layers {START_LAYER}-{END_LAYER-1})")
    print("=" * 70)

    hdr = f"{'Lyr':>4} |" + "|".join(f"{p[3:]:>8}" for p in DUMP_POINTS) + "|  Mean  "
    print(hdr)
    print("-" * len(hdr))

    all_sims = []
    for lyr in range(START_LAYER, END_LAYER):
        row_vals = []
        row_str = f" {lyr:03d} |"
        for pt in DUMP_POINTS:
            if pt in captured.get(lyr, {}) and pt in _golden.get(lyr, {}):
                sim = cosine_similarity(captured[lyr][pt], _golden[lyr][pt])
                row_vals.append(sim)
                all_sims.append(sim)
                row_str += f" {sim:.6f}"
            else:
                row_str += f" {'N/A':>8}"
        if row_vals:
            row_str += f" | {sum(row_vals)/len(row_vals):.6f}"
        print(row_str)

    if all_sims:
        mean_sim = sum(all_sims) / len(all_sims)
        print(f"\n  Mean cos_sim: {mean_sim:.6f}")
        print(f"  Min: {min(all_sims):.6f}, Max: {max(all_sims):.6f}")

    # 保存结果供后续合并
    torch.save(captured, "captured_65_79.pt")
    print(f"\n  Saved captured data to captured_65_79.pt")


if __name__ == "__main__":
    main()
