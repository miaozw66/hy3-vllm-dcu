#!/bin/bash
# 启动 vLLM 服务，接入 7 个已提速的 INT8 GEMM HIP 算子（TP8, Hy3 w8a8）。
# 用户原始启动参数 + 仅注入 patch 环境（无 --enforce-eager / 无 --max-num-batched-tokens）。
# 使用方式（容器内）:
#   bash /home/hy3-TP8/entry-vllm/start_user.sh            # use 模式（默认，真跑 HIP）
#   ZTH_W8A8_MODE=validate bash .../start_user.sh           # validate 模式（对比记 diff，返回原实现）
#   ZTH_W8A8_MODE=off   bash .../start_user.sh              # 纯原实现基线（零注入开销）
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/usr/local/lib/:/usr/local/lib64/:/opt/mpi/lib:/opt/hwloc/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
export PYTHONPATH=/home/hy3-TP8/entry-vllm:${PYTHONPATH:-}
export ZTH_W8A8_MODE=${ZTH_W8A8_MODE:-use}
export ZTH_W8A8_CACHE=${ZTH_W8A8_CACHE:-/home/hy3-TP8/ext_cache}

MODEL_PATH=${MODEL_PATH:-/models/Hy3-Channel-INT8-w8a8}
LOG=${LOG:-/workspace/hy-w8a8.log}
PORT=${PORT:-30001}

pkill -f "vllm serve --trust-remote-code --host 0.0.0.0 --port $PORT" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8

nohup vllm serve \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name hy3 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  --no-async-scheduling \
  -O1 \
  > "$LOG" 2>&1 &
echo "SERVER_PID=$!"
echo "log: $LOG  mode: $ZTH_W8A8_MODE"
