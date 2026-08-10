# HY3 vLLM Docker 镜像
#
# 基于海光 DTK 26.04 基础镜像，安装 HY3 定制版 vLLM
# 模型权重通过 -v 挂载，不打包进镜像（~280GB）
#
# 构建前确认：
#   1. 基础镜像 TAG 是否正确（联系海光确认）
#   2. dist/ 目录下有最新的 .whl 文件
#
# 构建：
#   docker build -t hy3-vllm:latest .
#
# 运行（单机 8 卡）：
#   docker run -itd \
#       --name hy3 \
#       --privileged \
#       --network=host \
#       --ipc=host \
#       --device=/dev/kfd \
#       --device=/dev/mkfd \
#       --device=/dev/dri \
#       --group-add video \
#       --cap-add=SYS_PTRACE \
#       --security-opt seccomp=unconfined \
#       --shm-size=200g \
#       -v /path/to/Hy3-Channel-INT8-w8a8:/model:ro \
#       hy3-vllm:latest

FROM harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.18.1-ubuntu22.04-dtk26.04-py3.10-20260617

# 安装 HY3 定制版 vLLM wheel
COPY dist/vllm-0.18.1+das.dtk2604.hy3-*.whl /tmp/
RUN pip install /tmp/vllm-0.18.1+das.dtk2604.hy3-*.whl && \
    rm /tmp/vllm-*.whl

# 部署脚本和配置
COPY deploy/   /workspace/vllm-hy3/deploy/
COPY tools/    /workspace/vllm-hy3/tools/
COPY configs/  /workspace/vllm-hy3/configs/

WORKDIR /workspace/vllm-hy3

# 默认单机 TP=8 启动
# 双机 PP=2 请用：docker exec hy3 bash deploy/run_pp2_80l.sh
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/model", \
     "--tensor-parallel-size", "8", \
     "--trust-remote-code", \
     "--enforce-eager", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.85", \
     "--port", "8000"]
