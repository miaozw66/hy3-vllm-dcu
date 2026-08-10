# HY3 双机 Pipeline Parallelism (PP=2) 通信方法

## 1. 概述

HY3 模型有 80 层 Decoder Layer，PP=2 将模型切成两半：
- **PP rank 0 (Node 0)**: Embedding + Layers 0~39 → 发送 `IntermediateTensors` 给 Node 1
- **PP rank 1 (Node 1)**: 接收 `IntermediateTensors` → Layers 40~79 + Final Norm + lm_head → 输出 logits

两节点间通过 **RCCL P2P send/recv** 传输 PP 边界数据。

HY3 使用**独立残差流**（separate residual stream），`HYV3DecoderLayer.forward()` 返回 `(hidden_states, residual)` 元组，两者都需跨 PP 边界传递。

---

## 2. 两种多机启动方式

### 2.1 Ray 方式（推荐用于生产）

```bash
# 依赖: Ray 集群已在两节点启动
# 仅需在一台机器上执行，Ray 自动调度

python -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
```

- **优点**: 只需单节点启动，Ray 自动管理多机进程
- **缺点**: 依赖 Ray 集群（`ray start --head` / `ray start --address=...`），日志分散在 Ray worker
- **适用**: 80 层完整模型

### 2.2 PyTorch 原生方式（推荐用于调试）

```bash
# 在两台机器上分别执行

# Node 0 (10.18.17.71):
python -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 10.18.17.71 \
  --master-port 29501 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5

# Node 1 (10.18.17.74):
python -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 10.18.17.71 \
  --master-port 29501 \
  ...
```

- **优点**: 无需 Ray，日志直出 stdout/stderr，`sitecustomize.py` dump 钩子直接生效
- **缺点**: 需在两台机器上分别启动，需手动管理进程
- **适用**: 4 层子模型快速调试

### 2.3 关键参数

| 参数 | 说明 |
|---|---|
| `--nnodes 2` | 总节点数 |
| `--node-rank 0/1` | 当前节点编号 |
| `--master-addr <IP>` | 主节点 IP（Node 0） |
| `--master-port 29501` | 主节点通信端口（PyTorch distributed init） |
| `--pipeline-parallel-size 2` | PP 切分数（= nnodes） |
| `--tensor-parallel-size 4` | 每机 TP 切分数（= GPU 数/节点） |

**注意**: `world_size = nnodes × TP = 2 × 4 = 8`，每节点 `local_world_size = TP = 4`。

---

## 3. PP 边界通信机制

### 3.1 PY3 Model forward 流程

```python
# hy_v3.py HYV3Model.forward()

def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    # === PP rank 0 (first rank) ===
    if get_pp_group().is_first_rank:
        hidden_states = self.embed_input_ids(input_ids)  # Embedding
        residual = None
    else:
        # === PP rank 1+ (后续 rank) ===
        hidden_states = intermediate_tensors["hidden_states"]  # 从上游接收
        residual = intermediate_tensors["residual"]

    # 遍历本 rank 负责的 layers
    for layer in self.layers[self.start_layer : self.end_layer]:
        hidden_states, residual = layer(positions, hidden_states, residual)

    # === 非最后一个 rank: 发送 IntermediateTensors 给下游 ===
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual,
        })

    # === 最后一个 rank: 最后处理 ===
    hidden_states = hidden_states + residual   # 合并残差
    hidden_states = self.norm(hidden_states)   # Final LayerNorm
    return hidden_states
```

### 3.2 IntermediateTensors 传输

PP 边界传输的是 `IntermediateTensors({"hidden_states": ..., "residual": ...})`，其底层通信：

```
Node 0 (PP rank 0)                    Node 1 (PP rank 1)
  layers 0~39 计算完毕                 等待接收
       |                                    |
  get_pp_group().send_tensor_dict()   get_pp_group().recv_tensor_dict()
       |  ──── RCCL P2P ────>              |
  (hidden_states, residual)           (hidden_states, residual)
                                           |
                                      layers 40~79 计算
```

- `send_tensor_dict` 将 dict 中各 tensor 序列化后通过 RCCL `send` 逐个发送
- `recv_tensor_dict` 接收并反序列化，恢复 `IntermediateTensors`
- 使用 **blocking send/recv**（RCCL 默认），确保数据完整性

### 3.3 独立残差流

HY3 不使用标准 Transformer 的 `x = x + attn(x) + mlp(x)` 残差累加，而是：

```python
# HYV3DecoderLayer.forward()
def forward(self, positions, hidden_states, residual):
    # 1. Input LayerNorm + RMSNorm (dual residual)
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # 2. Attention
    hidden_states = self.self_attn(positions, hidden_states)

    # 3. Post-Attention LayerNorm
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

    # 4. MLP / MoE
    hidden_states = self.mlp(hidden_states)

    return hidden_states, residual  # 两者分别返回，不合并
```

因此 PP 边界必须同时传递 `hidden_states` 和 `residual`，缺一不可。

---

## 4. RCCL 网络配置

```bash
export NCCL_SOCKET_IFNAME=eno1          # 指定网卡（NCCL/RCCL 通信走这个接口）
export NCCL_DEBUG=WARN                  # 日志级别（调试时可用 INFO）
export NCCL_IB_DISABLE=1               # 禁用 InfiniBand（走 TCP/IP over PCIe）
export HSA_FORCE_FINE_GRAIN_PCIE=1     # DCU 专有：强制 PCIe 细粒度通信
```

### 网络拓扑

```
10.18.17.71 (Node 0) ─── TCP/IP over PCIe (eno1) ─── 10.18.17.74 (Node 1)
     4×DCU GPU                                            4×DCU GPU
     TP rank 0,1,2,3                                      TP rank 0,1,2,3
```

PP 通信是**跨节点**的（PP rank 0 ↔ PP rank 1），TP 通信是**节点内**的（4 GPU 之间）。

---

## 5. 调试方法：sitecustomize.py Hook

### 5.1 原理

通过 `PYTHONPATH` 注入 `sitecustomize.py`，在 Python 启动时自动 monkey-patch `HYV3DecoderLayer.forward()` 和 `HYV3Model.forward()`，在每个 layer 的 7 个 dump point 保存中间 tensor。

### 5.2 环境变量

```bash
export PYTHONPATH=/data/mzw/vllm-hy3/submodel_debug:$PYTHONPATH
export VLLM_HY3_DUMP_DIR=/path/to/dump_dir       # 触发 dump（不设则跳过）
export VLLM_HY3_DUMP_SKIP=1                       # 跳过前 N 次 forward（默认1=跳过warmup）
```

### 5.3 Dump 文件结构

```
<dump_dir>/
├── layer_000/
│   ├── 00_input.pt               # hidden_states + residual（输入）
│   ├── 01_input_layernorm.pt     # input_layernorm 后
│   ├── 02_attention_out.pt       # self_attn 后
│   ├── 03_attention_residual.pt  # attention_out + residual
│   ├── 04_post_attention_layernorm.pt
│   ├── 05_mlp_out.pt             # MLP/MoE 后
│   └── 06_output.pt              # hidden_states + residual（最终输出）
├── layer_001/
│   └── ...
└── layer_039/                    # PP rank 0 最后一层
    └── 06_output.pt              # ≈ 发送给 Node 1 的 IntermediateTensors
```

**关键**: `layer_039/06_output.pt` = Node 0 的 `hidden_states + residual` = 跨 PP 边界发送的完整数据。

### 5.4 对比 Golden Dump

```bash
# 单机验证（所有层与 golden 对比）
python compare_dumps.py dumps/single_tp4_4l/ golden_dump/

# PP 边界验证（Node 0 的 layer_039/06_output 与 golden 对比）
python compare_dumps.py dumps/pp2_4l/ golden_dump/ --layers 0,1

# 双机 vs 单机（PP=2 结果与 TP=4 基准对比）
python compare_dumps.py dumps/pp2_4l/ dumps/single_tp4_4l/ --layers 2,3
```

---

## 6. 4 层子模型快速调试

完整 80 层模型加载需 ~17 分钟，调试迭代太慢。利用 Transformer 前馈性质：

> 子模型 layer N 的输出 == 完整模型 layer N 的输出（前 N-1 层输入相同、权重相同）

提取 4 层子模型（`extract_layers.py`），加载仅需 1-2 分钟：

```bash
# 提取连续前缀 4 层 (layer 0~3)
python submodel_debug/extract_layers.py \
  --model-dir /path/to/full_model \
  --out-dir submodel_debug/test4 \
  --layers 0,1,2,3

# PP=2 启动：每节点 2 层
# Node 0: ./start_vllm_4l_pp.sh 0
# Node 1: ./start_vllm_4l_pp.sh 1

# 推理测试
curl http://10.18.17.71:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "中国的首都是", "max_tokens": 16, "temperature": 0}'
```

PP=2 4 层模型的层分配：
- **Node 0**: Layers 0~1（PP rank 0）
- **Node 1**: Layers 2~3（PP rank 1），接收来自 Node 0 的 `IntermediateTensors`

---

## 7. 常见问题

### 7.1 RCCL 初始化超时

```bash
export NCCL_SOCKET_IFNAME=eno1  # 确保使用正确的网卡
export NCCL_DEBUG=INFO           # 查看详细通信日志
```

### 7.2 两侧 rank 不对称

确保两节点使用相同的 `--tensor-parallel-size` 和 `--pipeline-parallel-size`，总 GPU 数 = nnodes × TP。

### 7.3 Dump 为 warmup 数据

warmup 使用 dummy input（shape `[2048, 4096]`），实际推理 shape `[1, 3, 4096]`。设 `VLLM_HY3_DUMP_SKIP=1` 跳过 warmup。

### 7.4 双机间 NFS 共享

dump 目录需在两节点间共享（`/data` 为 NFS mount），否则 dump 文件分散在各节点本地。
