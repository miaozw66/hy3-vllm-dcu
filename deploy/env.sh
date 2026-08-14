# ============================================================
# Machine Configuration for vllm-hy3
#
# 移植到新机器时，只需修改此文件即可。所有启动脚本都会 source 它。
# Edit this file once per deployment. All launch scripts source it.
# ============================================================

# --- 项目根目录 (Project root, absolute path) ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# Docker 配置 (留空则为裸金属部署；本机为容器环境，无 docker)
# ============================================================
DOCKER_NAME=

# ============================================================
# 模型路径 (Model paths)
# ============================================================
# 完整 80 层 INT8 模型 (本地 nvme 只读挂载)
MODEL_PATH=/models/Hy3-Channel-INT8-w8a8
# 4 层调试子模型 (用于快速验证)；本机无此文件，留空
SUBMODEL_PATH=

# ============================================================
# MoE 调优配置目录 (MoE tuning config)
# ============================================================
MOE_CONFIG_DIR=${PROJECT_ROOT}/configs/moe_configs

# ============================================================
# 运行时目录 (Runtime directories)
# ============================================================
DUMP_DIR=${PROJECT_ROOT}/dumps
LOG_DIR=${PROJECT_ROOT}/logs

# ============================================================
# GPU 配置
# ============================================================
# 每节点 GPU 数量 (本机 8 卡)
GPU_COUNT=8

# ============================================================
# 日志控制 (Logging)
# ============================================================
# 调试模式：1=输出全量 INFO 日志（排查问题用）；0=默认安静模式，仅 WARNING 及以上
DEBUG_MODE=0

# ============================================================
# RCCL / NCCL 通信调优参数
# ============================================================
export RCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=4
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# ============================================================
# 通用环境变量
# ============================================================
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# RPC timeout: first inference triggers torch.compile which can take minutes
export VLLM_RPC_TIMEOUT=1800000
# EngineCore→Worker execute_model/sample_tokens timeout (seconds, default 300)
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
# NCCL heartbeat timeout: prevent watchdog killing workers during long torch.compile
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
