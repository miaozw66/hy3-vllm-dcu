# MetaInfer 精简推理框架 vs vLLM 原版 — 全量异同对比

> 对比对象：
> - **MetaInfer**：`/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/`（生成当前 golden dump 的 TP=4 版本）
> - **vLLM**：`/data/mzw/vllm-hy3/vllm/`（HYV3 移植）+ `/data/fh/vllm-main/`（原版引擎注册点）
> - **对齐验证层**：`/data/mzw/vllm-hy3/verify.py`（hook 实现）与 `diff.txt`（验证发现的差异记录）
>
> 生成日期：2026-08-03

---

## 0. 一句话总结

MetaInfer 是**正确性优先、显存饥饿驱动**的纯 PyTorch 参考实现（INT8 权重加载即解量化为 BF16、逐层按需加载、逐 expert 循环计算、手写 KV cache 与 all-reduce）；vLLM 是**性能优先的融合 kernel 引擎**（INT8 权重驻留显存、Triton/CUDA 融合 kernel、EP/EPLB、PagedAttention、连续批处理）。两套代码数学上等价，但在**权重驻留形态、路由公式顺序、专家累加精度、归约方式、KV cache 机制、调度方式**六个层面存在实现级差异。

---

## 1. 框架定位与代码结构

| 维度 | MetaInfer | vLLM |
|------|-----------|------|
| 定位 | 单请求、单 batch=1 的精简推理；验证/移植参考实现 | 工业级推理引擎：连续批处理、paged attention、投机解码 |
| 计算原语 | 全部 PyTorch 算子（`F.linear`、`F.scaled_dot_product_attention`、`torch.mm`） | 融合 kernel（Triton INT8、PagedAttention、fused_experts）+ torch.compile（可禁用） |
| 代码形态 | 7 个 Python 文件，~1500 行 | 引擎 + 模型文件（hy_v3.py 707 行）+ 注册 + parser + MTP |
| 显存策略 | 逐层加载/释放 + 三级专家缓存（GPU INT8 → CPU LRU → NFS） | 全量驻留 + EP 权重过滤 |
| 批处理 | 无（batch=1，HTTP server + 请求队列） | 连续批处理调度器 |
| 精度取向 | 数值对齐官方（fp32 路由/norm），性能次要 | 性能优先（INT8 kernel、FP8 cache 可选项） |

**MetaInfer 文件清单（code/004/）：**

| 文件 | 职责 |
|------|------|
| `modeling/layers.py` | RMSNorm、RotaryEmbedding、Attention、DenseMLP、SharedMLP、MoELayer、TransformerBlock |
| `modeling/model.py` | HYV3ForCausalLM 模型图 + dump hooks + 逐层加载调度 |
| `weight_loader.py` | direct I/O 读 safetensors、INT8→BF16 解量化、三级专家缓存 |
| `model_config.py` | Hy3Config（从 config.json 读取）+ 显存预算计算 |
| `kv_cache.py` | 手写 PagedKVCache（156 行） |
| `tp_utils.py` | 手写 all_reduce / all_gather / all_to_all / broadcast |
| `server.py` | HTTP 推理服务 + Golden Dump 模式 |
| `runner.py` | tokenize → prefill → generation 循环 |
| `dump_golden.py` / `dump_golden_tp4.sh` | golden dump（TP=1 备用 / TP=4 实际使用） |

**vLLM 文件清单（vllm-hy3/vllm/）：**

| 文件 | 职责 |
|------|------|
| `model_executor/models/hy_v3.py` | HYV3ForCausalLM / HYV3Model / HYV3DecoderLayer / HYV3Attention / HYV3MoEFused / HYV3FeedForward |
| `model_executor/models/hy_v3_mtp.py` | HYV3MTP 投机解码（MTP 层） |
| `transformers_utils/configs/hy_v3.py` | HYV3Config(PretrainedConfig) |
| `tool_parsers/hy_v3_tool_parser.py` | XML tool_call 解析 |
| `reasoning/hy_v3_reasoning_parser.py` | `<think>` 推理解析 |
| INTEGRATION.md 记录 | registry.py / config.py / speculative.py / weight_utils.py 中的注册与 KV scale 重映射 |

---

## 2. 配置定义

两者都从同一份 `config.json` 读取架构参数，字段一致（hidden_size=4096、80 层、64 Q 头 / 8 KV 头、head_dim=128、192 experts、top-8、sigmoid 路由、router_scaling_factor=2.826、Layer 0 dense 等）。差异：

| 差异点 | MetaInfer Hy3Config | vLLM HYV3Config |
|--------|---------------------|-----------------|
| 来源 | 手写 dataclass，`from_model_dir` 手动解析 config.json + compression_config | 继承 `PretrainedConfig`，与 HF 生态集成 |
| 量化识别 | 解析 `compression_config` 判定 `w8a8_channel_int8` | 走 vLLM 的 QuantizationConfig（compressed-tensors） |
| 专家维度 | `moe_intermediate_size=1536` | `expert_hidden_dim=1536`（命名不同） |
| MTP | `total_layers = num_hidden_layers + num_nextn_predict_layers`，但直接跳过 layer 80 | 单独由 `HYV3MTP` 处理（见 §10） |
| 显存预算 | 手写 `gpu_cache_layers_safe()` 估算（64GB K100AI 假定） | vLLM 的 `gpu_memory_utilization` 机制 |
| 专用开关 | `enable_lm_head_fp32`、`enable_moe_fp32_combine`、`enable_attention_fp32_softmax` | 同为 config 字段，但 `enable_attention_fp32_softmax` 注释说明 eager 路径恒为 fp32，该字段为未来后端预留 |

---

## 3. 权重加载与量化（最根本的架构差异）

### 3.1 MetaInfer：加载即解量化，显存里全是 BF16

- **读取**：`_read_shard_tensors_direct` / `_read_shard_tensors_bulk` 用 direct I/O（`open→seek→read→torch.frombuffer`），不用 safetensors mmap（NFS 下 mmap 因页错误慢 8 倍）；bulk 模式一次顺序读完整个 data section 再切片
- **解量化时点**：加载时立即执行 `_dequantize_int8`：`(w_i8.float() * scale.float()).to(torch.bfloat16)`，**显存中不保留 INT8**
- **专家三级缓存**（`load_experts_bf16`）：
  1. GPU INT8 cache（`_gpu_expert_int8`，按层预缓存，层内保持 INT8 以省显存，使用时再解量化）
  2. CPU LRU cache（`_expert_cache`，INT8+scale 原样缓存，按字节数 LRU 驱逐）
  3. NFS 直接读取（不占 CPU 缓存）
  - 解量化在 GPU 上做（`to(device)` 后 dequant，~0.001s vs CPU ~5s/层）
- **串行化 NFS 预热**：rank 0 先读全部 shard（填 OS 页缓存），通过 `/tmp/hy3_gpu_cache_rank0_ready` flag 文件通知 rank 1-3 再读（避免 4× 并发 NFS 打满带宽）
- **权重分片**：按 `eid % tp_size` 归属 rank（TP=EP 合一）；TP 切分均为连续 block 切分（`[rank*out/tp : (rank+1)*out/tp]`）
- **global 权重**（embed/lm_head/norm）：每 rank 全复制（replicated），`to(dtype=torch.bfloat16)`

### 3.2 vLLM：INT8 驻留显存，量化 kernel 推理时计算

- **读取**：标准 vLLM 加载器（safetensors），流式加载
- **驻留形态**：INT8 权重保持 INT8 在显存，推理由 compressed-tensors 量化线性层（Triton kernel）直接算 int8 计算
- **EP 权重过滤**：`enable_ep_weight_filter=True` 时只加载本 rank 需要的专家（hy_v3.py 的 `load_weights` 中 `is_expert_weight` 分支跳过非本地专家）
- **KV cache scale 重映射**：`weight_utils.py:1588-1592` 将 `self_attn.q.scale` 和 `self_attn.{k,v}_cache.scale` 重命名挂到 FP8 cache 参数（支持 FP8 KV cache 的 checkpoint 布局）
- **堆叠参数映射**：qkv_proj / gate_up_proj 用 `stacked_params_mapping` 合并加载（`.q_proj/.k_proj/.v_proj → .qkv_proj`、`.gate_proj/.up_proj → .gate_up_proj`）；MetaInfer 则分别读 q/k/v、gate/up（保留独立张量）

### 3.3 数值上的一致性

解量化公式两边一致：`(int8 × scale) → bf16`（L22 深度诊断验证权重 cos=1.0、max_diff=0）。差异在于**解量化的时点和驻留形态**：MetaInfer 在加载时、vLLM 在 kernel 内——数学等价，数值行为不同（见 §11）。

---

## 4. 注意力模块

| 步骤 | MetaInfer | vLLM 原版 | 数值差异 |
|------|-----------|-----------|----------|
| Q/K/V 投影 | 独立 `F.linear(hidden, q_proj)` 等，BF16 | `QKVParallelLinear`（INT8 Triton kernel），q/k/v 拼接后 split | **INT8 vs BF16 计算路径**（verify.py 改用 BF16 后 attention_out 0.995→0.999+） |
| QK norm | 手写 `_apply_head_norm`：**float32** 内计算，乘 fp32 weight 后 cast 回 | vLLM `RMSNorm`（kernel 内部 fp32 累加） | 近似一致 |
| RoPE | 手写 rotate_half；cos/sin 缓存 **float32**，`cos[position_ids].to(dtype)` 后**乘法在输入 dtype（BF16）**；rope_type="default" | `get_rope(is_neox_style=True)`，vLLM 实现 | verify.py 改用 FP32 输入后 Mean 0.999230→0.999500（仍与 MetaInfer 的 BF16 乘法不同） |
| 注意力 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)`，全量张量 | `Attention` 层 → PagedAttention kernel（需 ForwardContext，走 KV cache） | 验证走全量 SDPA 时一致；走 cache 时见 §7 |
| O 投影 | `F.linear` + 手动 `tp_utils.all_reduce`（WORLD group） | `RowParallelLinear`（内含 all-reduce，归约到 hidden_size 维） | 数学等价 |
| KV 扩展 | `repeat_interleave(num_kv_groups, dim=1)`（GQA 8→32/64） | Attention kernel 内部处理（或 x==d 复制） | 等价 |

---

## 5. MoE 与路由（数值差异集中地）

### 5.1 路由公式（diff.txt bug #5，已由 hook 修复）

```
MetaInfer (layers.py _route):
    router_logits = F.linear(x_flat.float(), router_weight.float())   # 全程 fp32
    router_logits = router_logits + expert_bias.float()               # bias 在 sigmoid 之前
    scores = sigmoid(router_logits)
    topk_scores, topk_ids = topk(scores, 8)
    topk_scores = topk_scores / (sum + 1e-20)                          # renormalize
    topk_scores = topk_scores * router_scaling_factor                  # ×2.826

vLLM 原生 (FusedMoE, scoring_func="sigmoid", e_score_correction_bias):
    topk(sigmoid(logits) + e_score_correction_bias)                    # bias 加在 sigmoid 之后！
    weights = sigmoid(logits).gather(topk_idx)
```

- **bias 位置不同** → 专家选择 7-8/8 重叠、路由权重不同 → MoE 层 cos 被限制在 0.97-0.98
- vLLM 的 `renormalize=config.route_norm`、`routed_scaling_factor` 与 MetaInfer 一致
- verify.py hook 实现了 MetaInfer 风格路由并将自定义 topk_weights/topk_ids 传入 `fused_experts` → MoE 层 0.999+

### 5.2 专家计算路径

| 维度 | MetaInfer | vLLM |
|------|-----------|------|
| 形式 | **逐 token × 逐 expert 循环**：`token_out += F.linear(gate*up, down) * score` | `fused_experts` Triton kernel：8 专家一次性 batch 加权求和 |
| 累加精度 | 每步 `expert_out * score`（score 为 fp32）后 **cast 回 BF16 再累加** | 内部 **fp32 累加器**，最后 cast |
| router weight 应用 | 输出上乘（`F.linear(...) * score`） | `apply_router_weight_on_input=False` 时输出上乘 |
| 权重形态 | 独立 gate/up/down 三张 BF16 | w13（gate+up 拼接）与 w2 两张 |
| 归约 | 每 token 手动 `all_reduce`（WORLD） | `_reduce_output`（EP 模式下 no-op，**bug #4**，见下） |

### 5.3 EP 模式 all-reduce 缺失（diff.txt bug #4，核心问题）

- vLLM `DefaultMoERunner._reduce_output` 的 `tensor_model_parallel_all_reduce` 在 EP 模式下变为 no-op → shared/routed 输出只有 1/TP 的 partial 结果直接相加
- 影响：TP=4 时 MoE 层 05_mlp_out cos 从 0.66 随层数恶化到 0.03
- 修复：verify.py hook 手动 `dist.all_reduce(shared_out)` 和 `dist.all_reduce(routed_out)`（注意用 WORLD group，仅 TP=EP=WORLD 时正确；正确做法是 `get_tensor_model_parallel_group()`）
- MetaInfer 无此问题：它 TP=EP 合一，且始终手动 all_reduce

### 5.4 共享专家

| 维度 | MetaInfer | vLLM |
|------|-----------|------|
| 实现 | `SharedMLP`（SwiGlu，BF16 F.linear 链） | `HYV3FeedForward`（`reduce_results=False`），由 FusedMoE 内部管理（`n_shared_experts=1`） |
| 归约 | 自身 `tp_utils.all_reduce` | 依赖 FusedMoE 的归约路径（同样受 bug #4 影响） |
| 计算精度 | BF16 | INT8（quant_config 传入）——diff.txt #6 显示 shared INT8 vs BF16 cos>0.9996，影响极小 |

---

## 6. 归一化与位置编码

| 组件 | MetaInfer | vLLM | 差异 |
|------|-----------|------|------|
| RMSNorm | 手写：`x.float()` → rsqrt(variance+eps) → ×weight → cast 回 | vLLM RMSNorm（融合 kernel） | 均 fp32 内部计算；vLLM 支持 fused 残差形式 `norm(x, residual)` |
| QK norm | 手写 fp32 per-head norm | `RMSNorm(head_dim)` | 等价 |
| RoPE | 手写 rotate_half；cos/sin **fp32 缓存**；乘法在 BF16 | `get_rope(is_neox_style=True)` | 缓存均 fp32；**乘法精度路径不同**（vLLM 实现细节） |
| 残差结构 | `residual = hidden; hidden = norm(hidden); ...; hidden = residual + hidden`（显式） | `hidden_states, residual = input_layernorm(hidden_states, residual)`（fused） | 数学等价；MetaInfer 的 00_input dump 点是残差合并后的值 |

---

## 7. KV Cache（详见对话讨论，此处归档）

| 维度 | MetaInfer `PagedKVCache` | vLLM PagedAttention |
|------|---------------------------|---------------------|
| 布局 | `[num_blocks, num_kv_heads, block_size, head_dim]`（token 连续） | `[num_blocks, num_kv_heads, head_dim, block_size]`（x==d 转置，利于 kernel 合并访问） |
| block_size | 16 | 16（verify.py 配置） |
| 分配 | 手写 free list 线性分配（batch=1 greedy 专用） | BlockManager：多序列、块复用、前缀共享 COW |
| 写入 | Python 逐 slot 拷贝（`abs_pos // block_size` 计算后切片赋值） | kernel 内批量写入 |
| 读取 | `get()` 将各 block `torch.cat` 组装**连续张量**再交给 SDPA | kernel 直接消费 block table，不组装连续张量 |
| dtype | 固定 BF16 | `cache_dtype="auto"`，可 FP8（HYV3 有 KV scale 权重重映射支持） |
| 使用场景 | 仅 generation（`seq_len==1`）；**prefill 时 kv_cache=None，全量 SDPA** | Attention 层常态路径，prefill 也走 |
| 显存预算 | 手算 ~0.5GB（max_blocks=1024 或按需计算） | `gpu_memory_utilization` 自动估算 |
| 对 golden 验证 | **无数值影响**——golden 生成与 verify.py 的 prefill 均不经 cache | 同上 |

**推论**：generation 阶段若做逐 token 对比，KV cache 是独立差异源（dtype、读取路径、block 边界）。vLLM 启用 FP8 cache 时引入量化误差。

---

## 8. 分布式与显存管理

| 维度 | MetaInfer | vLLM |
|------|-----------|------|
| 通信 | 裸 `torch.distributed`（NCCL/RCCL）+ 手写 `tp_utils`（默认 WORLD group） | vLLM 分布式封装（TP/EP/PP 各自 group） |
| 并行方式 | TP 与 EP 合一（专家按 `eid % tp_size`） | TP + EP 分离 + **EPLB**（冗余专家：物理专家数 = 逻辑 + redundant，`enable_eplb=True`，`num_redundant_experts`） |
| PP | 无 | 支持（`make_layers` 按 PP 划分，IntermediateTensors 传递 hidden+residual） |
| 显存管理 | 逐层 `_load_layer` / `clear_gpu_working_set`；全局权重常驻 ~2GB；层工作集峰值 ~0.2GB；专家按需 | 全量常驻 + EP 权重过滤；cache 由 gpu_memory_utilization 控制 |
| NFS 优化 | direct I/O、bulk 读、rank0 预热页缓存、flag 文件串行化 | 无（假定本地盘/共享文件系统直接 mmap） |
| 通信次数 | 每层每 token：attention 1× + MoE 1× + shared 1× all_reduce | kernel 内归约（EP 模式有 bug） |

---

## 9. 推理流程与调度

### 9.1 MetaInfer（runner.py）

```
tokenize → forward_embed（全部前缀 token）
prefill：逐层循环 [ _load_layer(layer_idx); forward_layer_prefill ]
    - attention 对所有前缀 token 批量（一次）
    - MoE 层内逐 token（bound 专家数 ≤ 8/token，避免 union I/O 爆炸）
    - 每层结束 clear_gpu_working_set
generation：逐 token [ lm_head → argmax/top-p → forward_embed(单 token) → 逐层循环 ]
    - generation 走 KV cache（forward_layer_single：从 cache 取过去 K/V + 存当前）
```

特点：batch=1、HTTP server + 双队列（request/response）、贪婪解码默认、无连续批处理。

### 9.2 vLLM

调度器 + 连续批处理（多请求并发、动态增删序列）、前缀缓存、KV cache 管理、streaming 输出；prefill/decode 由引擎分派，模型侧只实现 `forward(input_ids, positions, intermediate_tensors)` + `compute_logits`。

### 9.3 模型接口差异

- MetaInfer：`forward_layer_prefill(layer_idx, hidden, position_ids, kv_cache, dump_dir)` / `forward_layer_single(...)` 按层手动调度，dump hooks 内嵌
- vLLM：`HYV3ForCausalLM.forward(input_ids, positions, intermediate_tensors, inputs_embeds)` 一次性走完整模型（`HYV3Model` 内部循环 `self.layers`），`compute_logits` 单独调用

---

## 10. MTP / Layer 80（投机解码）

| 维度 | MetaInfer | vLLM |
|------|-----------|------|
| Layer 80 | **完全不加载**：`TransformerBlock` 只建 0-79，80 跳过（与官方 HF `_keys_to_ignore_on_load_unexpected` 一致） | `HYV3MTP`（hy_v3_mtp.py，471 行）：完整实现 HYV3MultiTokenPredictor + HYV3MultiTokenPredictorLayer + HYV3SharedHead |
| 启用方式 | 无 | speculative config：`hy_v3` → `hy_v3_mtp` model_type 转换（speculative.py:505-509），`get_spec_layer_idx_from_weight_name` 过滤权重 |
| 功能 | 无 | 投机解码（multi-token prediction）、`sample` 接口、spec layer 权重重写（`_rewrite_spec_layer_name`、`_split_qkv_weight`） |

---

## 11. 数值精度差异全表（验证实验得出）

### 11.1 MetaInfer 自身修复过的 5 个精度 bug（hy3_framework_analysis.md）

1. RMSNorm 原用 BF16 → 改 fp32
2. RoPE cos/sin 缓存原用 BF16 → 改 fp32 缓存、使用时转换
3. Router linear + sigmoid 原 BF16 → 改 fp32
4. Layer 80 移除（不参与主 forward）
5. Router normalize → ×scale 顺序对齐官方

### 11.2 verify.py 验证中发现的 vLLM 侧问题（diff.txt）

| # | 问题 | 现象 | 修复 |
|---|------|------|------|
| 1 | 残差顺序差异 | 中间 hidden 语义不同（数学等价） | 文档记录，无修复 |
| 2 | SDPA 维度顺序错误 | 期望 (B,H,S,D) 传成 (B,S,H,D)，L0 attn_out ~0.80 | hook 内 permute 修复 |
| 3 | SDPA 缺 is_causal | attn_out 0.90→0.80 | 加 is_causal=True |
| 4 | **EP 模式缺 TP all-reduce** | MoE 层 cos 0.66→0.03（TP=4） | hook 手动 all_reduce（Mean 0.698→0.998） |
| 5 | **路由 bias 在 sigmoid 之后** | MoE 层卡在 0.97-0.98 | hook 实现 MetaInfer 路由（→0.999+） |
| 6 | Shared expert INT8 量化 | 额外精度损失 | hook 手动 BF16 解量化（Mean 0.994→0.9988） |
| 7 | shared INT8 vs BF16 | cos>0.9996 | 非瓶颈，记录 |
| 8 | fused_experts vs 手动 PyTorch | cos=0.999988 | 非瓶颈，记录 |
| 9 | 30 层验证（Attempt 2） | Mean 0.998232 | 见 #11 |
| 10 | L22 深度诊断 | 权重/路由/kernel 全对，根因为误差累积 | 记录 |
| 11 | Attempt 3：BF16 attention + Layer0 dense + FP32 RoPE | attention_out >0.999；Mean 0.999230→0.999500 | hook 内实施 |

### 11.3 当前仍存留的数值差异（0.9995 vs 0.9999 的根源）

1. **专家累加方式**：fused_experts（fp32 一次性累加）vs MetaInfer 逐 expert 循环（每步 cast BF16）
2. **RoPE 乘法精度**：verify.py 用 FP32 输入 vs MetaInfer 在 BF16 乘法（经验上 FP32 更接近 golden，vLLM 的 BF16 路径与之有别的差异）
3. **加法顺序**：`shared + routed`（verify.py）vs `routed + shared`（MetaInfer）
4. **INT8→BF16 解量化通道级 1 ULP 舍入**：理论上存在，实测权重 max_diff=0（非主因）

误差传播机制：每层 1e-5~1e-4 级舍入 → 残差累积 → 某层 hidden 差异被 sigmoid 路由放大（导数峰值 0.25）→ mlp_out 退化（L9: 0.996、L13: 0.988、L21: 0.978）→ 后续层偶发恢复。退化层位置随计算路径变化（非确定性模式）。

---

## 12. vLLM 独有的引擎集成组件（MetaInfer 没有）

| 组件 | 位置 | 作用 |
|------|------|------|
| 模型注册 | `model_executor/models/registry.py:139` | `"HYV3ForCausalLM": ("hy_v3", ...)` |
| MTP 注册 | `registry.py:642` | `"HYV3MTPModel": ("hy_v3_mtp", "HYV3MTP")` |
| Config 注册 | `transformers_utils/configs/__init__.py`、`config.py:109` | `hy_v3="HYV3Config"` |
| Reasoning parser | `reasoning/__init__.py:75-78` | `<think>` 推理解析 |
| Tool parser | `tool_parsers/__init__.py:85-88` | XML tool_call 解析 |
| KV scale 重映射 | `model_loader/weight_utils.py:1588-1592` | FP8 KV cache scale 名称 |
| MTP model_type 转换 | `config/speculative.py:505-509` | `hy_v3` → `hy_v3_mtp` |
| LoRA / PP 接口 | `SupportsLoRA`、`SupportsPP` | 引擎特性接入 |

---

## 13. 差异分类总结

**A. 语义等价、实现不同（无数值影响）**：残差结构、GQA KV 扩展、TP 切分方式、gate_up 拼接 vs 独立张量、O-proj 归约方式、模型接口形态（逐层 vs 整图）、embedding/lm_head 复制方式。

**B. 数学等价、数值行为不同（当前 0.9995 的根源）**：专家累加顺序与精度、RoPE 乘法精度、shared+routed 加法顺序、INT8 kernel vs BF16 路径。

**C. 公式不同（已被 hook 修复的真实 bug）**：路由 bias 位置（sigmoid 前后）、EP 模式 all-reduce 缺失。

**D. 纯工程差异（无数值影响）**：KV cache 布局/分配/读取方式（验证路径不经 cache）、权重驻留形态、NFS 优化、显存预算策略、批处理调度、MTP 支持与否、parser/注册等引擎集成。

---

## 附录：验证环境与结果

- 模型：`/data/model/hygon/Hy3-Channel-INT8-w8a8/.../snapshots/master`（INT8 w8a8 channel 量化）
- Golden：TP=4，80 层 × 7 点 = 560 文件，14MB，276s，prompt="中国的首都是"（3 tokens）
- 当前最佳结果（verify.py，NUM_LAYERS=23, TP=4）：Mean 0.999500 / Min 0.978417（L21）/ <0.99 共 2 点（L13: 0.9879、L21: 0.9784）
- 前 10 层：仅 L0 全部点 ≥0.9999；L1-L8 层输出 ≥0.9999 但 mlp_out 未达；L9 输出 0.999891、mlp_out 0.996016
- 日志：`run_verify_0803_0858.log`
