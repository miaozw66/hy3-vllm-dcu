# HY3 vLLM DCU 部署指南

海光 K100 (gfx928) 平台上使用 vLLM 部署 Tencent HY3 大模型（`HYV3ForCausalLM`），支持 INT8 W8A8 量化推理，TP=4/8 单机、PP=2 双机。

# 环境和模型准备

```Bash
# 基础镜像（包含 DTK 26.04 + PyTorch 2.10.0+das）
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

模型使用海光 K100 适配版 HY3 Channel-INT8-w8a8（compressed-tensors 格式，99 个分片，~280GB）。

```Bash
pip install modelscope

# HY3 Channel-INT8-w8a8（推荐）
modelscope download --model hygon/Hy3-Channel-INT8-w8a8 --local_dir /data/model/Hy3-Channel-INT8-w8a8
```

安装 vLLM wheel（包含 HY3 模型文件和全部修改）：

```Bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

# 准备环境变量

```Bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── RCCL 通信 ──────────────────────────────────────────
# 网卡名（通过 ip addr 确认实际名称）
export NCCL_SOCKET_IFNAME=eno1
# 禁用 InfiniBand（K100 使用 PCIe/RoCE TCP 通信）
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# RCCL 调优参数
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# ── AITER CK 加速 ─────────────────────────────────────
# gfx928 上 CK kernel 无预编译 .co 文件，当前禁用
export VLLM_ROCM_USE_AITER=0

# ── MoE 调优 ──────────────────────────────────────────
export VLLM_TUNED_CONFIG_FOLDER=/path/to/vllm-hy3/configs/moe_configs

# ── 通用 ──────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

# ── 模型路径 ──────────────────────────────────────────
export MODEL_PATH=/data/model/Hy3-Channel-INT8-w8a8
```

# 启动 vLLM

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

> **注意**：单机 TP=8 未充分验证，首次启动建议先用较小的 `--max-model-len` 确认显存够用后再逐步增大。

## 双机 PP=2（每机 4 卡，80 层全量）

**前置条件**：Node 0 → Node 1 免密 SSH，模型路径两台机器一致（NFS 或各自存放）。

编辑 `deploy/env.sh`，填入两台机器的 IP 和网卡名，然后：

```Bash
# 标准启动（max-model-len=8192，带 dump）
bash deploy/run_pp2_80l.sh

# NIAH 性能测试版（max-model-len=262144）
bash deploy/run_pp2_80l_niah.sh

# 调试模式（可变 max-model-len）
bash deploy/run_debug_pp2.sh 8192     # 小上下文快速启动
bash deploy/run_debug_pp2.sh 32768    # 中等上下文
bash deploy/run_debug_pp2.sh 131072   # 大上下文
```

PP=2 模式下 Node 0 为 leader，Node 1 为 follower。启动后通过 SSH 自动在 Node 1 上拉起 follower 进程。

# 验证 RCCL 通信

```Bash
# 单机验证
python3 tools/test_rccl_single.py
# 预期输出：SUCCESS: Single-node RCCL works!

# 双机验证（Node 0）
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 0

# 双机验证（Node 1）
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 1
```

# 测试推理

```Bash
# 健康检查
curl -s http://localhost:8000/health

# Completion 请求
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hy3",
        "prompt": "中国的首都是",
        "max_tokens": 5,
        "temperature": 0.0
    }' | python3 -m json.tool

# Chat 请求
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hy3",
        "messages": [
            {"role": "user", "content": "你好，帮我介绍一下南京"}
        ],
        "max_tokens": 100
    }' | python3 -m json.tool
```

# 性能测试

```Bash
# 安装 bench 依赖
pip install ray[default]

# 基准测试（随机数据）
vllm bench serve \
    --model hy3 \
    --tokenizer $MODEL_PATH \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --max-concurrency 10 \
    --num-prompts 10 \
    --host localhost \
    --port 8000 \
    --random-range-ratio 0 \
    --request-rate 100 \
    --metric-percentiles 95,99 \
    --temperature 0 \
    --seed 42 \
    --save-result \
    --result-dir output_hy3
```

# 性能监控

```Bash
# GPU 利用率监控（600 秒，每 10 秒采样）
python3 tools/monitor_gpu.py 600 10

# 双机监控
MONITOR_NODES='[{"host":"192.168.1.100","type":"local","docker":""},
               {"host":"192.168.1.101","type":"remote","docker":""}]' \
python3 tools/monitor_gpu.py 600 10
```

# 故障排查

## RCCL 初始化超时

1. 确认网卡名正确：`ip addr show` 查看实际网卡名，更新 `deploy/env.sh` 中 `NIC`
2. 确认防火墙未阻止端口（`MASTER_PORT` 需在 29500-29600 范围开放）
3. 确认 `NCCL_IB_DISABLE=1` 已设置（海光 K100 不使用 InfiniBand）

## OOM (Out of Memory)

- 降低 `--gpu-memory-utilization`（如 0.75）
- 减小 `--max-model-len`
- PP=2 模式下确认每卡只加载 40 层

## 模型加载缓慢

添加多线程加载（适用于 NFS）：

```Bash
--model-loader-extra-config '{"enable_multithread_load": true}'
```

## PP=2 双机启动问题

1. 确认 Node 0 到 Node 1 免密 SSH 正常：`ssh -o StrictHostKeyChecking=no $NODE1_IP hostname`
2. 确认模型路径在两台机器上一致
3. Docker 部署时确认 `env.sh` 中 `DOCKER_NAME` 正确；裸金属部署时留空
4. PP=2 的 `kv_cache_coordinator` 和 `collective_rpc` 修复已内置于 vLLM wheel 中，无需额外 monkey-patch

---

## 模型信息

| 参数 | 值 |
|------|-----|
| 架构 | HYV3ForCausalLM (Dense + MoE Transformer Decoder) |
| 隐藏维度 | 4096 |
| 层数 | 80（Layer 0 Dense + Layer 1-79 MoE + Layer 80 MTP） |
| 注意力头数 | 64 (Q) / 8 (KV) — GQA，Head Dim 128 |
| MoE | 192 experts, top-8 sigmoid routing, shared expert ×1 |
| 量化 | INT8 W8A8 channel-wise (compressed-tensors) |
| 词表大小 | 120,832 |
| 模型大小 | ~280GB（99 个 safetensors 分片） |
| RoPE | theta=11,158,840, neox_style=True |
