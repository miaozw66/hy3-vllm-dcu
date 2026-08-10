# HY3 vLLM DCU 启动指南

在海光 K100 (gfx928) 平台上启动 HY3 模型的 vLLM 推理服务，支持单机 TP=8 和双机 PP=2。

# 1. 环境准备

确保机器已安装以下软件栈：

| 组件 | 说明 |
|------|------|
| DTK 26.04 | 海光 DCU 工具链（HIP runtime、ROCm 兼容层、RCCL） |
| PyTorch 2.10.0+das | 海光定制版 PyTorch |
| aiter / flash-attn / lightop / lmslim | 海光算子库（ROCm CK / FlashAttention / 融合算子） |

以上组件需由海光方面预装。确认安装成功后，继续以下步骤。

## 1.1 安装 vLLM wheel

```Bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

## 1.2 下载模型权重

```Bash
pip install modelscope
modelscope download --model hygon/Hy3-Channel-INT8-w8a8 --local_dir /data/model/Hy3-Channel-INT8-w8a8
```

> 模型 ~280GB（99 个 safetensors 分片），双机部署需放在两台机器都能访问的路径。

## 1.3 克隆仓库

```Bash
git clone https://github.com/miaozw66/hy3-vllm-dcu
cd vllm-hy3
```

# 2. 配置机器信息

编辑 `deploy/env.sh`，所有启动脚本都会读取此文件：

```Bash
vim deploy/env.sh
```

| 变量 | 说明 |
|------|------|
| `MASTER_ADDR` | 主节点 IP |
| `NODE1_IP` | 从节点 IP（单机留空） |
| `NIC` | 通信网卡名 |
| `DOCKER_NAME` | Docker 容器名（裸金属留空） |
| `MODEL_PATH` | 模型路径 |

# 3. 环境变量

`env.sh` 中已预设以下变量，启动脚本 `source env.sh` 后自动生效：

```Bash
# ── RCCL 通信 ──────────────────────────────────────────
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# ── 通用 ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

手动启动时，补充以下变量：

```Bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eno1          # 网卡名，按实际情况修改
export NCCL_DEBUG=WARN
export VLLM_ROCM_USE_AITER=0            # gfx928 暂不启用 CK 加速
export VLLM_TUNED_CONFIG_FOLDER=/path/to/vllm-hy3/configs/moe_configs
export MODEL_PATH=/data/model/Hy3-Channel-INT8-w8a8
```

# 4. 验证 RCCL 通信

```Bash
# 单机验证
python3 tools/test_rccl_single.py
# 预期输出：SUCCESS: Single-node RCCL works!

# 双机验证（两节点分别执行）
# Node 0:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 0
# Node 1:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 1
```

# 5. 启动服务

## 单机 TP=8（8 卡，80 层全量）

```Bash
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

首次启动建议 `--max-model-len 8192` 确认显存够用，再逐步增大。

## 双机 PP=2（每机 4 卡，80 层全量）

**前置条件**：Node 0 → Node 1 免密 SSH，模型路径两台机器一致。

```Bash
# 标准启动（max-model-len=262144, gpu-mem=0.90）
bash deploy/run_pp2_80l.sh

# 大上下文测试（256K，同上参数）
bash deploy/run_pp2_80l_niah.sh

# 可变上下文调试
bash deploy/run_debug_pp2.sh 8192     # 小上下文快速启动
bash deploy/run_debug_pp2.sh 131072   # 大上下文
```

> PP=2 启动脚本会自动：通过 SSH 在 Node 1 拉起 follower → 在 Node 0 拉起 leader → 等待服务就绪。

# 6. 测试推理

```Bash
# 健康检查
curl -s http://localhost:8000/health

# Completion
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5,"temperature":0.0}'

# Chat
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model":"hy3",
        "messages":[{"role":"user","content":"你好，帮我介绍南京"}],
        "max_tokens":100
    }'
```

# 7. 常见启动问题

| 问题 | 排查 |
|------|------|
| RCCL 初始化超时 | 检查网卡名、防火墙是否放行 29500-29600 端口 |
| OOM | 降 `--gpu-memory-utilization` 或 `--max-model-len` |
| 模型加载慢（NFS） | 加 `--model-loader-extra-config '{"enable_multithread_load":true}'` |
| PP=2 Node 1 连不上 | 确认免密 SSH、模型路径一致、Docker 容器名正确 |
| `import vllm` 报错 | 确认 DTK 26.04 和 PyTorch 2.10.0+das 已安装 |
