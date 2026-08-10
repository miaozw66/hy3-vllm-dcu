# HYV3 vLLM 集成说明

本文档记录 HYV3 模型在原 vLLM 源码 (`/data/fh/vllm-main/`) 中的全部注册/集成位置。

## 1. 模型注册

**文件**: `vllm/model_executor/models/registry.py`

```python
# Line 139
"HYV3ForCausalLM": ("hy_v3", "HYV3ForCausalLM"),

# Line 642
"HYV3MTPModel": ("hy_v3_mtp", "HYV3MTP"),
```

## 2. Config 注册

**文件**: `vllm/transformers_utils/configs/__init__.py`

```python
# Line 42
"HYV3Config": "vllm.transformers_utils.configs.hy_v3",

# Line 114
"HYV3Config",
```

**文件**: `vllm/transformers_utils/config.py`

```python
# Line 109
hy_v3="HYV3Config",
```

## 3. Reasoning Parser 注册

**文件**: `vllm/reasoning/__init__.py`

```python
# Lines 75-78
"hy_v3": (
    "hy_v3_reasoning_parser",
    "HYV3ReasoningParser",
),
```

## 4. Tool Parser 注册

**文件**: `vllm/tool_parsers/__init__.py`

```python
# Lines 85-88
"hy_v3": (
    "hy_v3_tool_parser",
    "HYV3ToolParser",
),
```

## 5. Speculative Decoding 配置

**文件**: `vllm/config/speculative.py`

```python
# Line 51 - MTP architectures list
"hy_v3_mtp",

# Lines 505-509 - model type conversion
if hf_config.model_type == "hy_v3":
    hf_config.model_type = "hy_v3_mtp"
```

## 6. Weight Loading (KV cache scale remapping)

**文件**: `vllm/model_executor/model_loader/weight_utils.py`

```python
# Lines 1588-1592 - HYV3-specific KV cache scale name remapping
# Regex patterns for .self_attn.q.scale and .self_attn.{k,v}_cache.scale
```
