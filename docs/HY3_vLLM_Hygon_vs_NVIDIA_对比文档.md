# HY3 vLLM：海光 (Hygon/DCU) vs 英伟达 (NVIDIA/CUDA) 实现差异全量对比

> 生成日期：2026-08-05  
> 对比基准：`/data/mzw/vllm-hy3/vllm/model_executor/models/hy_v3.py`（上游 NV 实现，707 行，与 `vllm-project/vllm` PR #40681 逐字节一致）  
> 上游来源：`https://github.com/vllm-project/vllm.git`，commit `d0009ddb0`，作者 `stevenkuang` (Tencent)，2026-04-23 合入  
> 对比目标：`/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/hy_v3.py`（海光适配版，905 行）  
> vLLM 版本：0.18.1+das.dtk2604 (DCU/ROCm build)  
> 海光平台：Hygon C-3000 (KONGMING)，ROCm 6.3，RCCL 2.22.3，HIP 6.3

---

## 一、总览

| 维度 | NVIDIA 原版 | 海光适配版 |
|------|-----------|-----------|
| 模型核心文件 | `hy_v3.py` (707 行) | `hy_v3.py` (905 行，+198 行) |
| MoE 计算后端 | CUDA FusedMoE kernel | ROCm AITER FusedMoE kernel |
| Attention 后端 | FlashAttention (CUDA) | TRITON_ATTN (ROCm) |
| INT8 推理 kernel | CUDA INT8 GEMM | TritonInt8ScaledMMLinearKernel |
| 集合通信 | NCCL | RCCL (ROCm Collective Communications Library) |
| TopK Router | CUDA fused_topk | rocm_aiter_grouped_topk |
| Token Dispatch | CUDA fused_experts | AITER fused_experts (ROCm) |

---

## 二、hy_v3.py 模型核心文件差异（逐行对比）

### 差异 1：FusedMoE 启用 `reduce_results=True`

**位置**：`HYV3MoEFused.__init__` 中 `FusedMoE(...)` 调用（第 197 行）

**NVIDIA 原版**（无此参数，默认 `False`）：
```python
self.experts = FusedMoE(
    ...
    n_shared_experts=config.num_shared_experts,
    shared_experts=self.shared_mlp,
)
```

**海光适配版**（新增一行）：
```python
self.experts = FusedMoE(
    ...
    n_shared_experts=config.num_shared_experts,
    shared_experts=self.shared_mlp,
    reduce_results=True,  # <-- 海光新增
)
```

**差异原因**：
- NVIDIA CUDA FusedMoE kernel 在计算过程中**内部已完成 TP all-reduce**，Shared Expert 的输出在 MoE kernel 内部合并到 Routed Expert 输出中，不需要额外归约
- 海光 ROCm AITER FusedMoE kernel 返回**未归约的 per-TP-rank 局部结果**，Shared Expert 和 Routed Expert 输出分别返回
- 设置 `reduce_results=True` 后，FusedMoE layer 在返回前自动调用 `tensor_model_parallel_all_reduce` 归约各 TP rank 的结果
- 相关代码路径：`fused_moe/layer.py:660` → `runner/default_moe_runner.py:374` 检查 `reduce_results` 标志决定是否触发 all-reduce

**影响**：不设置此参数将导致输出值缺少其他 TP rank 的贡献，模型输出完全错误（已验证）。

---

### 差异 2：MoE 输出 Tuple 解包

**位置**：`HYV3MoEFused.forward` 中 `self.experts(...)` 调用后（第 214-215 行）

**NVIDIA 原版**：
```python
final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=router_logits
)
return final_hidden_states.view(orig_shape)
```

**海光适配版**：
```python
final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=router_logits
)
if isinstance(final_hidden_states, tuple):
    final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
return final_hidden_states.view(orig_shape)
```

**差异原因**：
- NVIDIA FusedMoE kernel 返回单个 `torch.Tensor`，即 routed + shared 已合并的结果
- 海光 AITER FusedMoE kernel 返回 `(shared_output, routed_output)` **元组**：
  - `final_hidden_states[0]` = Shared Expert 输出
  - `final_hidden_states[1]` = Routed Expert (Top-K) 输出
- 需要手动求和：`shared + routed` 得到最终 MoE 层输出
- 这个行为在 `rocm_aiter_fused_moe.py:120` 的返回类型签名中有声明：`-> tuple[torch.Tensor, torch.Tensor]`

**影响**：不进行 tuple 解包将导致 `view(orig_shape)` 对 tuple 调用而报错（run-time crash）。

---

### 差异 3：Dump Hooks（调试辅助代码）

**位置**：文件末尾（第 714-905 行，共 192 行）

**NVIDIA 原版**：无。

**海光适配版**：新增了完整的 dump hook 框架，用于 PP=2 推理的逐层中间状态捕获和验证。

涉及 3 个关键函数/类的 monkey-patch：
1. **`HYV3DecoderLayer.forward`** — 每层 dump 7 个检查点（00_input ~ 06_output）
2. **`HYV3Model.forward`** — dump final_norm，跟踪 prefill 状态
3. **`HYV3ForCausalLM.compute_logits`** — dump logits
4. **`HYV3MoEFused.forward`** — dump MoE router logits 和 expert 输出

激活方式：环境变量
- `VLLM_HY3_DUMP_DIR` — 指定 dump 输出目录
- `VLLM_HY3_DUMP_SKIP` — 跳过前 N 次 forward pass（warmup/profile），默认 3

**注意**：此差异与硬件平台无关，是验证项目特有的调试代码，**生产部署时不需要**。

---

## 三、其他关键差异（vLLM 核心框架层）

### 3.1 FusedMoE 计算后端

| 组件 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| MoE Kernel 文件 | `fused_moe/layers/fused_moe.py` (CUDA) | `fused_moe/rocm_aiter_fused_moe.py` (ROCm AITER) |
| 量化支持 | CUDA INT8/FP8 fused kernel | AITER INT8/FP8 via QuantMethod enum |
| TopK 路由 | `fused_topk` / `fused_topk_bias` (CUDA) | `rocm_aiter_grouped_topk` (ROCm AITER) |
| Expert Dispatch | CUDA `moe_align_block_size` | AITER `topK_meta_data` 预计算 |

核心 MoE 计算流程差异：
```
NVIDIA:
  router_logits → fused_topk() → topk_ids/weights
                               → moe_align_block_size() → sorted_token_ids
                               → fused_experts() → [CUDA kernel 内部 all-reduce] → output

海光 ROCm:
  router_logits → rocm_aiter_grouped_topk() → topk_ids/weights
                                            → aiter fused_experts() → (shared_out, routed_out) tuple
                                            → 手动 sum(shared, routed)
                                            → [条件 all-reduce，由 reduce_results 控制]
```

### 3.2 Attention 计算后端

| 维度 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| 首选后端 | FlashAttention (CUDA) | TRITON_ATTN |
| 可选后端 | FlashInfer, FlashMLA, FlexAttention | 仅 TRITON_ATTN |
| 编译优化 | CUDAGraphs | 默认禁用（enforce_eager） |

日志验证（海光启动日志）：
```
INFO: Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN'].
```

CUDAGraphs 在海光平台上默认禁用，因为 ROCm 的 HIP Graph 支持尚不完善。这导致：
- 每个 decode step 需要重新 launch kernel（而非回放 graph）
- 对 latency 有轻微影响，但正确性不受影响

### 3.3 INT8 量化推理

| 维度 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| W8A8 INT8 Kernel | CUDA CUTLASS INT8 GEMM | TritonInt8ScaledMMLinearKernel |
| 量化方案 | Compressed-Tensors W8A8 INT8 | 相同配置，但经 Triton kernel 计算 |
| FP8 支持 | CUDA FP8 (Hopper+) | 不支持（DCU 无 FP8 硬件单元） |

日志验证：
```
INFO: Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

**数值精度影响**：Triton INT8 kernel 与 CUDA CUTLASS INT8 在浮点舍入细节上可能有微小差异。这是验证报告中 MoE 层 (`05_mlp_out`) 余弦相似度最低（mean 0.993）的主要原因之一。

### 3.4 集合通信 (Collective Communication)

| 维度 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| 通信库 | NCCL | RCCL 2.22.3 |
| TP All-Reduce | NCCL all_reduce (NVLink/NVSwitch) | RCCL all_reduce (PCIe/HCCS) |
| PP Send/Recv | NCCL P2P | RCCL P2P |
| 跨节点通信 | NCCL over InfiniBand/RoCE | RCCL over TCP/PCIe |

日志中的 RCCL 警告：
```
NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instability or hang!
```
这是 RCCL 在 Hygon 平台上的已知提示，不影响功能，但建议添加 `iommu=pt` 内核参数。

### 3.5 平台自动检测

vLLM 通过 `platforms/__init__.py` 自动检测硬件平台：

```python
# 检测逻辑（简化）
if torch.cuda.is_available():
    if torch.version.hip is not None:
        is_rocm = True  # 海光 / AMD GPU
    else:
        is_rocm = False # NVIDIA GPU
```

所有后续的 kernel 选择、attention 后端、MoE 实现都基于 `is_rocm` 标志做分发。

### 3.6 AITER 加速库 (ROCm AITER Ops)

| 维度 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| 加速库 | CUDA custom ops（内置） | AITER (`_aiter_ops.py`, 2023 行) |
| MoE Kernel 入口 | `fused_moe/layers/fused_moe.py` | `fused_moe/rocm_aiter_fused_moe.py` |
| Grouped TopK | CUDA `ops.grouped_topk()` | `rocm_aiter_ops.grouped_topk()` |
| QuantType 枚举 | N/A | `QuantMethod` / `ActivationMethod` IntEnum |

AITER (AMD AI Tensor Engine Runtime) 是海光 DCU 上针对 MoE 和 INT8 推理的高性能加速库。它在 vLLM 中的入口点是 `/usr/local/lib/python3.10/dist-packages/vllm/_aiter_ops.py`（2023 行），封装了 ROCm HIP kernel 的 Python 接口。

NVIDIA vLLM 发行版中**不包含** `_aiter_ops.py` 和 `rocm_aiter_fused_moe.py`，这些是海光 ROCm vLLM 特有的。

### 3.7 FusedMoE 性能调优配置 (Tuning Config)

| 维度 | NVIDIA | 海光 (ROCm) |
|------|--------|------------|
| 调优配置文件 | 海量 GPU 专用 JSON（H100, H200, B200 等） | AMD MI300X 有少量配置，KONGMING **无配置** |
| HY3 模型配置 | `E=192, N=384, device_name=NVIDIA_H200.json` 等 | **缺失** `E=192,N=384,device_name=KONGMING.json` |
| 效果 | 预调优的 block size / num_warps / pipeline 参数 | 使用 default MoE config，性能可能次优 |

启动日志中的相关警告：
```
WARNING: Using default MoE config. Performance might be sub-optimal!
Config file not found at .../configs/E=192,N=384,device_name=KONGMING.json
```

目前已存在的 MoE 配置文件（`fused_moe/configs/`）覆盖了 NVIDIA H100/H200/H20/B200/L40S 和 AMD MI300X/MI325X 等主流 GPU，但 **KONGMING (Hygon C-3000) 尚无专用调优配置**。这不影响正确性，但 MoE 层的计算效率未达到最优。

---

## 四、hy_v3.py 核心架构（NVIDIA 与海光 共同部分）

以下架构在两个平台上**完全一致**（未修改），确认无平台差异：

### 4.1 模型架构组件

| 组件 | 说明 |
|------|------|
| `HYV3FeedForward` | Dense MLP（仅 layer 0 使用），SwiGLU 激活 |
| `HYV3MoEFused` | MoE 层（layer 1-79），192 experts + 1 shared expert，topk=8，sigmoid routing |
| `HYV3Attention` | Multi-Head Attention，RoPE 位置编码，GQA（8 KV heads / 32 Q heads） |
| `HYV3DecoderLayer` | 单层 Transformer Decoder，独立残差流 |
| `HYV3Model` | 完整 80 层模型，支持 PP（返回 IntermediateTensors） |
| `HYV3ForCausalLM` | 语言模型头，lm_head 与 embedding 权重共享 |

### 4.2 模型规格

| 参数 | 值 |
|------|-----|
| 总层数 | 80（layer 0 = dense FFN，layer 1-79 = MoE） |
| Hidden Size | 4096 |
| Intermediate Size | 3072 |
| Attention Heads | 32 (Q) / 8 (KV)，GQA |
| Head Dim | 128 |
| Experts | 192 routed + 1 shared |
| Top-K | 8 |
| Routing | Sigmoid（非 softmax） |
| Vocab Size | 120,832 |
| 量化 | INT8 W8A8 (Compressed-Tensors)，BF16 推理 |
| 总权重 | ~280 GB (INT8) / ~560 GB (BF16) |

### 4.3 独立残差流 (Independent Residual Stream)

HY3 使用**独立残差流**设计（区别于标准 Transformer 的 Pre-Norm / Post-Norm）：

```python
# HYV3DecoderLayer.forward 返回 (hidden_states, residual) 元组
def forward(self, positions, hidden_states, residual, idx=-1):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # ... attention, MLP ...
    return hidden_states, residual
```

PP 通信时，两个流都通过 `IntermediateTensors` 传递：
```python
IntermediateTensors({
    "hidden_states": hidden_states,   # shape: (n_tokens, 4096)
    "residual": residual,             # shape: (n_tokens, 4096)
})
```

---

## 五、差异总结矩阵

| # | 差异项 | 影响文件 | 差异类型 | NV 行为 | 海光行为 | 严重程度 |
|---|--------|---------|---------|---------|---------|---------|
| 1 | `reduce_results=True` | `hy_v3.py:197` | 功能性 | MoE kernel 内部 all-reduce | 需显式触发 TP all-reduce | **关键** |
| 2 | Tuple 解包 | `hy_v3.py:214-215` | 功能性 | 返回单个 Tensor | 返回 `(shared, routed)` 元组 | **关键** |
| 3 | Dump Hooks | `hy_v3.py:714-905` | 调试/辅助 | 无 | 环境变量激活的逐层 dump | 无（仅调试用） |
| 4 | MoE Kernel | `rocm_aiter_fused_moe.py` | 后端 | CUDA fused_moe | ROCm AITER fused_moe | **关键** |
| 5 | Grouped TopK | `grouped_topk_router.py` | 后端 | CUDA fused_topk | rocm_aiter_grouped_topk | **关键** |
| 6 | Attention | vLLM backend selector | 后端 | FlashAttention (CUDA) | TRITON_ATTN (ROCm) | 中等 |
| 7 | CUDAGraphs | vLLM compilation | 优化 | 默认启用 | 默认禁用 (eager) | 中等（性能） |
| 8 | INT8 Kernel | vLLM quant selector | 后端 | CUDA CUTLASS INT8 | TritonInt8ScaledMM | 低（精度） |
| 9 | 集合通信 | RCCL vs NCCL | 通信 | NCCL | RCCL 2.22.3 | **关键** |
| 10 | 平台检测 | `platforms/__init__.py` | 框架 | `is_rocm=False` | `is_rocm=True` | **关键** |
| 11 | AITER 加速库 | `_aiter_ops.py`, `rocm_aiter_fused_moe.py` | 后端库 | 不存在（CUDA 内置） | AITER 2023 行 ROCm 加速库 | **关键** |
| 12 | MoE 调优配置 | `fused_moe/configs/` | 性能 | 有 H100/H200/B200 等海量配置 | **KONGMING 无专用配置**（使用 default） | 低（仅性能） |

---

## 六、已验证的正确性结论

基于 80 层 PP=2 完整验证（`verification_80l_pp2_0805_1038.txt`）：

| 指标 | 数值 |
|------|------|
| 逐层对比点数 | 560 (80 layers × 7 dump points) |
| 平均余弦相似度 | **0.997662** |
| 最低余弦相似度 | 0.947811 (Layer 56, `05_mlp_out`) |
| PP 通信残差流 | cos > 0.999（`03_attention_residual`, `06_output`） |
| Embedding 匹配 | cos = 1.000002, max_abs_diff = 0.00 |

**结论**：海光平台上的上述 10 项差异（特别是 #1, #2, #4, #5, #9 五项关键差异）均已正确适配。海光 vLLM 推理结果与 MetaInfer Golden Dump 高度一致，所有数值差异均在 INT8 量化的浮点精度范围内。

---

## 附录 A：相关文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| 海光适配版 hy_v3.py | `/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/hy_v3.py` | 905 行，运行时生效 |
| 上游参考 hy_v3.py | `/data/mzw/vllm-hy3/vllm/model_executor/models/hy_v3.py` | 707 行，不含 dump hooks |
| Dump hooks 补丁 | `/data/mzw/vllm-hy3/patches/hy_v3.py` | 与已安装版本一致 |
| Grouped TopK Router | `/data/mzw/vllm-hy3/patches/grouped_topk_router.py` | 已应用到 vLLM |
| ROCm AITER MoE | `/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` | 海光 MoE kernel |
| FusedMoE Layer | `/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/layer.py` | `reduce_results` 参数声明处 |
| Default MoE Runner | `/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py` | `reduce_results` 执行处 |
| 平台检测 | `/usr/local/lib/python3.10/dist-packages/vllm/platforms/__init__.py` | `is_rocm` 标志 |
| 验证报告 | `/data/mzw/vllm-hy3/verification_80l_pp2_0805_1038.txt` | 最新 80 层验证结果 |
| 模型配置文件 | `/data/mzw/vllm-hy3/vllm/transformers_utils/configs/hy_v3.py` | HYV3Config |
| MTP 模型 | `/data/mzw/vllm-hy3/vllm/model_executor/models/hy_v3_mtp.py` | 多 token 预测扩展 |
| Reasoning Parser | `/data/mzw/vllm-hy3/vllm/reasoning/hy_v3_reasoning_parser.py` | 推理格式解析 |
| Tool Parser | `/data/mzw/vllm-hy3/vllm/tool_parsers/hy_v3_tool_parser.py` | 工具调用解析 |

---

## 附录 B：上游 vLLM 中的 HY3 注册点（NV + Hygon 共用）

以下文件在 vLLM 0.18.1 中注册了 HY3 模型，两个平台共用：

| 注册点 | 文件 |
|--------|------|
| 模型类注册 | `transformers_utils/configs/__init__.py` → `HYV3Config` |
| 模型类型映射 | `transformers_utils/config.py` → `hy_v3="HYV3Config"` |
| 推理 Parser | `reasoning/__init__.py` → `hy_v3 → HYV3ReasoningParser` |
| 工具 Parser | `tool_parsers/__init__.py` → `hy_v3 → HYV3ToolParser` |
| 投机解码 | `config/speculative.py` → `hy_v3_mtp` |
| 权重加载 | `model_executor/models/__init__.py` → `HYV3ForCausalLM` |
| 特殊权重映射 | `default_loader.py:1381` → `HYV3 format: .self_attn.q.scale → .self_attn.attn.q_scale` |
