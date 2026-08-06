# vLLM HYV3 推理框架移植

## 来源

从 `/data/fh/vllm-main/` 的 vLLM 源码中移植出的 HYV3ForCausalLM（Tencent HY3 大模型）推理框架支持代码。模型标识为 `tencent/Hy3-preview` / `tencent/Hy3-preview-Base`。

## 模型架构摘要

| 参数 | 值 |
|------|-----|
| 架构 | HYV3ForCausalLM (Dense + MoE Transformer Decoder) |
| 隐藏维度 | 4096 |
| 层数 | 80 decoder layers |
| 注意力头数 | 64 (Q) / 8 (KV) — GQA |
| Head Dim | 128 |
| QK Normalization | 是 |
| RoPE | theta=11,158,840, neox_style=True |
| Layer 0 | Dense SwiGLU MLP (intermediate=13,312) |
| Layer 1-79 | MoE (192 experts, top-8 sigmoid routing) |
| Expert Dim | 1,536 |
| Shared Expert | 1 per MoE layer |
| 词表大小 | 120,832 |
| 量化 | INT8 w8a8 channel-wise (compressed-tensors) |
| MTP | 1 nextn_predict layer (layer 80) |

## 文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `model_executor/models/hy_v3.py` | 707 | 核心模型: HYV3ForCausalLM, HYV3Model, HYV3DecoderLayer, HYV3Attention, HYV3FeedForward, HYV3MoEFused |
| `model_executor/models/hy_v3_mtp.py` | 471 | MTP 投机解码: HYV3MTP, HYV3MultiTokenPredictor, HYV3MultiTokenPredictorLayer |
| `transformers_utils/configs/hy_v3.py` | 185 | HYV3Config(PretrainedConfig) 模型配置类 |
| `tool_parsers/hy_v3_tool_parser.py` | 644 | XML-style Tool Call 解析器 (`<tool_call>` 标签) |
| `reasoning/hy_v3_reasoning_parser.py` | 141 | 推理过程解析器 (`<think>...</think>` 标签) |
| `tests/test_hy_v3_tool_parser.py` | 274 | Tool Parser 测试 |
| `tests/test_hy_v3_reasoning_parser.py` | 274 | Reasoning Parser 测试 |

## 支持的 vLLM 特性

- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Expert Parallelism + EPLB (Expert Parallel Load Balancing)
- LoRA
- MTP (Multi-Token Prediction) 投机解码
- FP8 KV Cache
- GPTQ / Compressed-Tensors 量化

## 注册位置

详见 `INTEGRATION.md`。在原 vLLM 中，HYV3 通过以下入口集成:
1. `model_executor/models/registry.py` — 模型架构注册
2. `transformers_utils/configs/` — 配置类注册
3. `reasoning/__init__.py` — 推理解析器注册
4. `tool_parsers/__init__.py` — Tool Call 解析器注册
5. `config/speculative.py` — MTP 架构配置
