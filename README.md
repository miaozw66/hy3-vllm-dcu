# vllm-hy3: vLLM Adaptation for Tencent HY3 on Hygon DCU

将 Tencent HY3 大模型（`HYV3ForCausalLM`）的 vLLM 推理框架适配到海光 DCU（K100 / gfx928）平台，支持 TP=4/8 单机推理和 PP=2 双机流水线推理。

## 1. 模型架构摘要

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

## 2. 前置条件

### 2.1 硬件

- **GPU**: 8 × 海光 K100 AI (K500SM_AI / gfx928)，单卡 64 GiB 显存
- **单机部署**: 1 台 8 卡机器即可运行 TP=8 全量 80 层
- **双机部署**: 2 台机器（PP=2），每台建议 ≥ 4 卡，机器间通过 PCIe/RoCE 互联

### 2.2 软件栈（DTK和PyTorch需预装,暂时只打包了vLLM）

一个完整的海光 K100 环境需包含以下组件：

| 组件 | 说明 |
|------|------|
| **DTK 26.04** | 海光 DCU 工具链（含 HIP runtime、ROCm 兼容层、RCCL 通信库） |
| **PyTorch 2.10.0+das** | 海光定制版 PyTorch |
| **vLLM wheel** | 海光定制版 vLLM，包含 HY3 模型文件及全部修改 |

安装 vLLM wheel(本地实验的配置打包的)：

```bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

> **注意**：`pip install` 该 wheel 后，vLLM 即安装完毕，无需再单独安装其他 vLLM 包。仓库中的 `vllm/` 目录是上游源码快照，仅用于代码参考和 HY3 模型文件查阅。

### 2.3 模型权重

需要 HY3 INT8 W8A8 量化模型（compressed-tensors 格式），放置于所有节点可访问的路径（本地或 NFS）。

## 3. 快速开始

### 3.1 克隆仓库

```bash
git clone [<repo-url> vllm-hy3](https://github.com/miaozw66/hy3-vllm-dcu)
cd vllm-hy3
```

### 3.2 配置机器信息

编辑 **`deploy/env.sh`**，这是针对双机8卡的配置文件：

```bash
vim deploy/env.sh
```

关键配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `MASTER_ADDR` | 主节点 IP | `192.168.1.100` |
| `NODE1_IP` | 从节点 IP（单机留空） | `192.168.1.101` |
| `NIC` | 通信网卡名 | `eno1` |
| `DOCKER_NAME` | Docker 容器名（裸金属留空） | `""` |
| `MODEL_PATH` | 80 层完整模型路径 | `/data/models/hy3-int8` |
| `GPU_COUNT` | 每节点 GPU 数量 | `8` |

### 3.3 验证 RCCL 通信（单机）

```bash
python3 tools/test_rccl_single.py
```

预期输出：`SUCCESS: Single-node RCCL works!`

### 3.4 启动推理服务

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/hy3-model \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

### 3.5 测试推理

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5}'
```


## 5. 部署场景

### 5.1 单机 TP=8（8 卡，完整 80 层，可能有误，没在单机8卡启动过）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

如需更长的上下文，逐步增大 `--max-model-len` 并调低 `--gpu-memory-utilization`。

### 5.2 双机 PP=2（每机 4 卡，完整 80 层）

```bash
# 1. 确认 env.sh 中 NODE1_IP、DOCKER_NAME、NIC 配置正确
# 2. 启动双机 PP=2
bash deploy/run_pp2_80l.sh
```

此脚本会自动：
- 通过 SSH 在 Node 1 上启动 PP follower
- 在 Node 0 上启动 PP leader
- 等待服务就绪并输出健康检查命令

**NIAH 性能测试版（256K 上下文）：**

```bash
bash deploy/run_pp2_80l_niah.sh
```

### 5.3 调试模式（渐进增大 max-model-len）

```bash
bash deploy/run_debug_pp2.sh 8192    # 小上下文快速启动
bash deploy/run_debug_pp2.sh 32768   # 中等上下文
bash deploy/run_debug_pp2.sh 131072  # 大上下文
```

## 6. 双机 PP=2 环境要求

### 6.1 网络

- 两台机器需通过 PCIe 网络互联，RCCL 使用 TCP 通信
- 需要免密 SSH（Node 0 → Node 1）
- 模型路径需要在两台机器上一致（NFS 挂载或各自存放）

### 6.2 RCCL 验证（双机）

```bash
# 在 Node 0 上:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_multinode.py

# 在 Node 1 上:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_multinode.py
```

## 7. 性能测试

### 7.1 GPU 监控

```bash
# 监控 600 秒，每 10 秒采样
python3 tools/monitor_gpu.py 600 10

# 双机监控（通过环境变量配置节点列表）
MONITOR_NODES='[{"host":"192.168.1.100","type":"local","docker":""},
               {"host":"192.168.1.101","type":"remote","docker":""}]' \
python3 tools/monitor_gpu.py 600 10
```

### 7.2 NIAH (Needle in a Haystack) 测试

```bash
python3 benchmark/niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192,16384,32768,65536,131072,262144
```

## 8. 故障排查

### RCCL 初始化超时

1. 确认网卡名正确：`ip addr show` 查看实际网卡名，更新 `env.sh` 中 `NIC`
2. 确认防火墙未阻止端口：`MASTER_PORT` 需在 29500-29600 范围开放
3. 确认 `NCCL_IB_DISABLE=1` 已设置（不使用 InfiniBand）

### OOM (Out of Memory)

- 降低 `--gpu-memory-utilization`（如 0.75）
- 减小 `--max-model-len`
- 如果使用 PP=2，确认每张卡只加载 40 层

### 模型加载缓慢

添加 `--model-loader-extra-config '{"enable_multithread_load": true}'` 启用多线程加载（适用于 NFS）。

## 9. 移植到新机器 — 步骤清单

将本项目移植到另一台海光 K100 DCU 机器的完整步骤：

1. **确认硬件**: 8 × K100 AI (gfx928), 单卡 64 GiB
2. **确认软件栈**: DTK 26.04, PyTorch 2.10.0+das, vLLM 0.18.1+das 已安装
3. **克隆仓库**: `git clone <repo-url> && cd vllm-hy3`
4. **编辑配置**: 修改 `deploy/env.sh` 中的 IP、NIC、路径、Docker 容器名
5. **验证 RCCL**: `python3 tools/test_rccl_single.py`
6. **80 层单机 TP=8**: 按 5.1 节命令启动
7. **80 层双机 PP=2**（如需）: `bash deploy/run_pp2_80l.sh`

### 配置模板

复制 OpenCode 配置文件并填入你的模型路径：

```bash
cp configs/opencode.json.template configs/opencode.json
# 编辑 configs/opencode.json，将 <MODEL_PATH> 替换为实际路径
```

## 10. 项目文件结构

```
vllm-hy3/
├── README.md                     # 本文件
├── deploy/
│   ├── env.sh                    # ★ 集中式机器配置（移植时首先修改此文件）
│   ├── run_pp2_80l.sh            # 双机 PP=2 启动（dump 模式）
│   ├── run_pp2_80l_niah.sh       # 双机 PP=2 启动（NIAH 性能测试）
│   ├── run_debug_pp2.sh          # 双机 PP=2 调试（可变 max-len）
│   ├── run_pp2_ray_4l.sh         # 单机 PP=2 Ray 调试
│   ├── run_pp_dump.sh            # PP=2 带 dump
│   └── start_vllm_pp.sh          # 通用 PP 启动器
├── configs/
│   ├── opencode.json.template    # OpenCode 配置模板
│   └── moe_configs/              # MoE kernel 调优参数
│       └── E=192,N=384,device_name=KONGMING.json
├── reference/                    # 参考数据和调试工具
├── tools/
│   ├── test_rccl_single.py       # 单机 RCCL 测试
│   ├── test_rccl_multinode.py    # 双机 RCCL 测试
│   ├── test_rccl_direct.py       # 双机 RCCL 测试（mp.spawn 方式）
│   ├── monitor_gpu.py            # GPU 利用率监控
│   └── summarize_doc.py          # 文档总结测试脚本
├── benchmark/
│   └── niah_test.py              # NIAH 性能测试
├── vllm/                         # vLLM 0.18.1 源码 snapshot + HY3 模型文件
└── docs/                         # 移植过程文档（移植记录、调优指南等）
```

## 11. 已知问题与限制

1. **AITER CK kernel 暂未启用**: gfx928 不在上游 `_ON_GFX9` 列表，且 CK kernel 在 gfx928 上无预编译 `.co` 文件，当前部署默认 `VLLM_ROCM_USE_AITER=0`
2. **NFS 权重加载慢**: 多线程加载 (`enable_multithread_load`) 可缓解
3. **enforce-eager 禁用 CUDA Graph**: gfx928 上 HIP Graph 尚不稳定，暂时禁用
4. **MoE 调优配置**: `configs/moe_configs/` 中的 device_name=KONGMING 仅匹配海光卡；换 GPU 需重新生成
