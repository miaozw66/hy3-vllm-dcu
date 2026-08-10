# HY3 vLLM DCU 启动指南

在海光 K100 (gfx928) 平台上启动 HY3 模型的 vLLM 推理服务，支持单机 TP=8 和双机 PP=2。

# 1. 环境和模型准备

```Bash
# 基础镜像（DTK 26.04 + PyTorch 2.10.0+das）
export IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260617

docker run -itd \
    --name hy3_vllm \
    --shm-size=200g \
    --privileged \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /data/model:/model:ro \
    $IMAGE \
    /bin/bash
```

进入容器后，下载模型并安装 vLLM wheel：

```Bash
# 下载 HY3 Channel-INT8-w8a8 模型（~280GB，99 个分片）
pip install modelscope
modelscope download --model hygon/Hy3-Channel-INT8-w8a8 --local_dir /data/model/Hy3-Channel-INT8-w8a8

# 安装 vLLM wheel（含 HY3 模型文件及全部修改）
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

# 2. 配置机器信息

编辑 `deploy/env.sh`，填入本机（和目标机器）的实际信息：

```Bash
vim deploy/env.sh
```

| 变量 | 说明 |
|------|------|
| `MASTER_ADDR` | 主节点 IP（单机填本机 IP） |
| `NODE1_IP` | 从节点 IP（单机留空） |
| `NIC` | 通信网卡名（`ip addr show` 查看） |
| `DOCKER_NAME` | Docker 容器名（裸金属部署留空） |
| `MODEL_PATH` | 模型路径 |

# 3. 设置环境变量

```Bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── RCCL 通信 ──────────────────────────────────────────
export NCCL_SOCKET_IFNAME=eno1          # 网卡名，按实际情况修改
export NCCL_IB_DISABLE=1                # K100 不用 InfiniBand
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# ── AITER ──────────────────────────────────────────────
export VLLM_ROCM_USE_AITER=0            # gfx928 暂不启用 CK 加速

# ── 通用 ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

# ── 模型路径 ──────────────────────────────────────────
export MODEL_PATH=/data/model/Hy3-Channel-INT8-w8a8
```

> **海光 CPU 绑核**（可选，Intel CPU 不需要）：
> ```Bash
> export VLLM_NUMA_BIND=1
> export VLLM_RANK0_NUMA=0; export VLLM_RANK1_NUMA=1
> export VLLM_RANK2_NUMA=2; export VLLM_RANK3_NUMA=3
> export VLLM_RANK4_NUMA=4; export VLLM_RANK5_NUMA=5
> export VLLM_RANK6_NUMA=6; export VLLM_RANK7_NUMA=7
> ```

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
# 标准启动
bash deploy/run_pp2_80l.sh

# 大上下文测试（256K）
bash deploy/run_pp2_80l_niah.sh

# 可变上下文调试
bash deploy/run_debug_pp2.sh 8192     # 小上下文快速启动
bash deploy/run_debug_pp2.sh 131072   # 大上下文
```

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
