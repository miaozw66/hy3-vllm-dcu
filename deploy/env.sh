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
MASTER_ADDR=127.0.0.1
# 从节点 IP (PP rank 1)，单机部署时留空
NODE1_IP=
# 网卡名称 — 用于 NCCL/RCCL 通信
NIC=

# ============================================================
# Docker 配置 (留空则为裸金属部署)
# ============================================================
DOCKER_NAME=

# ============================================================
# 模型路径 (Model paths)
# ============================================================
# 完整 80 层 INT8 模型 (NFS 或本地路径)
MODEL_PATH=/models/Hy3-Channel-INT8-w8a8
# 4 层调试子模型 (用于快速验证)
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
# 每节点 GPU 数量
GPU_COUNT=8

# ============================================================
# RCCL / NCCL 通信调优参数
# ------------------------------------------------------------
# A/B（2026-08-24，benchmark/BOTTLENECK_ALLREDUCE_20260824.md）：单流解码
# 瓶颈是每层 MoE allreduce 的 latency。NCCL_PROTO=LL + NCCL_ALGO=Tree 把
# 单流 tpot 从 77.4ms 压到 63.0ms（-18%），LL 为主贡献（-16%）、Tree 单用
# 反而慢。conc16 吞吐基本不变（LL 是延迟协议，高并发下收益被摊薄）。
#
# 注意：两行用 ${VAR:-default} 守卫，允许上层脚本（A/B 驱动）先 export
# NCCL_PROTO/NCCL_ALGO 再 source 本文件即注入对照协议（如 Simple/Ring），
# 不覆盖时保持 LL/Tree 默认。跨 run 数值对比（含 A/B）双方协议必须一致，
# 否则 W8A8 INT8 归约顺序漂移会翻转近并列 logits（2026-08-20 教训）。
# ============================================================
export RCCL_BUFFSIZE="${RCCL_BUFFSIZE:-8388608}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export NCCL_ALGO="${NCCL_ALGO:-Tree}"

# ============================================================
# 通用环境变量
# ============================================================
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
# RPC timeout: first inference and rank-local MTP loading can take over 30 minutes
# on NFS. Milliseconds.
export VLLM_RPC_TIMEOUT=7200000
# EngineCore→Worker execute_model/sample_tokens timeout in seconds.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Distributed collective timeout used by both node launch commands, in seconds.
export DISTRIBUTED_TIMEOUT_SECONDS=7200
# NCCL heartbeat must exceed the longest model-loading collective.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
