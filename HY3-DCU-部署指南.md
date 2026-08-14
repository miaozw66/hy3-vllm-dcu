# HY3 vLLM DCU 启动指南

在海光 K100 (gfx928) 平台上启动 HY3 模型的 vLLM 推理服务（单机 8 卡 TP=8）。

# 前置条件

- 8 × 海光 K100 AI (gfx928)，单卡 64 GiB 显存
- DTK 26.04、PyTorch 2.10.0+das、海光算子库（aiter / flash-attn / lightop / lmslim）已安装

# 1. 安装 vLLM

```Bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

# 2. 克隆仓库

```Bash
git clone -b tp8 https://github.com/miaozw66/hy3-vllm-dcu
cd hy3-vllm-dcu
```

# 3. 下载模型

```Bash
pip install modelscope
modelscope download --model hygon/Hy3-Channel-INT8-w8a8 --local_dir /data/model/Hy3-Channel-INT8-w8a8
```

# 4. 修改配置

编辑 `deploy/env.sh`，填入实际机器的模型路径：

```Bash
vim deploy/env.sh
```

需要修改的变量：

- `MODEL_PATH` — 80 层完整模型路径
- `SUBMODEL_PATH` — 4 层调试子模型（可留空）
- `GPU_COUNT` — GPU 数量（默认 8）
- `DEBUG_MODE` — 1=全量 INFO 日志（排查问题）；0=安静模式（默认，仅 WARNING 及以上）
- `DOCKER_NAME` — Docker 容器名（容器环境留空）

# 5. 验证 RCCL

```Bash
python3 tools/test_rccl_single.py
```

预期输出：`SUCCESS: Single-node RCCL works!`

# 6. 启动服务

```Bash
bash deploy/run_tp8_single_80l.sh                    # CUDA graph 模式（默认，安静日志）
DEBUG_MODE=1 bash deploy/run_tp8_single_80l.sh       # 全量 INFO 日志（排查问题用）
MODE=eager bash deploy/run_tp8_single_80l.sh         # enforce-eager 模式
```

# 7. 测试

```Bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/models/Hy3-Channel-INT8-w8a8","prompt":"中国的首都是","max_tokens":5}'
```
