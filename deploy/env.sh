# ============================================================
# Machine Configuration for vllm-hy3
#
# 移植到新机器时，只需修改此文件即可。所有启动脚本都会 source 它。
# Edit this file once per deployment. All launch scripts source it.
# ============================================================

# --- 项目根目录 (Project root, absolute path) ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
# 网络配置 (Network)
# ============================================================
# 主节点 IP (PP rank 0)
MASTER_ADDR=10.18.17.71
# 从节点 IP (PP rank 1)，单机部署时留空
NODE1_IP=10.18.17.74
# 网卡名称 — 用于 NCCL/RCCL 通信
NIC=eno1

# ============================================================
# Docker 配置 (留空则为裸金属部署)
# ============================================================
DOCKER_NAME=mmh_qwen_opt

# ============================================================
# 模型路径 (Model paths)
# ============================================================
# 完整 80 层 INT8 模型 (NFS 或本地路径)
MODEL_PATH=/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master
# 4 层调试子模型 (用于快速验证)
SUBMODEL_PATH=${PROJECT_ROOT}/reference/submodel_debug/test4

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
# 每节点 GPU 数量
GPU_COUNT=4

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
