# HY3 vLLM DCU 启动指南

在海光 K100 (gfx928) 平台上启动 HY3 模型的 vLLM 推理服务。

# 前置条件

- 8 × 海光 K100 AI (gfx928)，单卡 64 GiB 显存
- DTK 26.04、PyTorch 2.10.0+das、海光算子库（aiter / flash-attn / lightop / lmslim）已安装
- 双机 PP=2 需 Node 0 → Node 1 免密 SSH

# 1. 安装 vLLM

```Bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

# 2. 克隆仓库

```Bash
git clone https://github.com/miaozw66/hy3-vllm-dcu
cd vllm-hy3
```

# 3. 下载模型

```Bash
pip install modelscope
modelscope download --model hygon/Hy3-Channel-INT8-w8a8 --local_dir /data/model/Hy3-Channel-INT8-w8a8
```

# 4. 修改配置

编辑 `deploy/env.sh`，填入实际机器的 IP、网卡名、模型路径：

```Bash
vim deploy/env.sh
```

需要修改的变量：

- `MASTER_ADDR` — 主节点 IP
- `NODE1_IP` — 从节点 IP（单机留空）
- `NIC` — 通信网卡名
- `DOCKER_NAME` — Docker 容器名（裸金属留空）
- `MODEL_PATH` — 模型路径

# 5. 验证 RCCL

```Bash
# 单机
python3 tools/test_rccl_single.py

# 双机（两节点分别执行）
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 0   # Node 0
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 1   # Node 1
```

# 6. 启动服务

**单机 TP=8**（8 卡，80 层全量）：

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

**双机 PP=2**（每机 4 卡，80 层全量）：

```Bash
bash deploy/run_pp2_80l.sh
```

脚本会自动处理 Node 1 SSH 拉起、环境变量设置（RCCL 参数、AITER 开关、网卡名等）。其他变体：

```Bash
bash deploy/run_pp2_80l_niah.sh       # 256K 上下文，用于 NIAH 测试
bash deploy/run_debug_pp2.sh 8192     # 自定义 max-model-len
```

# 7. 测试

```Bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5}'
```
