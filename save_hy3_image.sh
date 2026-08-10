#!/bin/bash
# HY3 vLLM Docker 镜像打包脚本
#
# 在宿主机上运行此脚本，将 mmh_qwen_opt 容器保存为可分发镜像
#
# 用法：
#   1. SSH 到宿主机（worker24）
#   2. bash save_hy3_image.sh
#
# 输出：
#   镜像 hy3-vllm:latest 和 tar 包 hy3-vllm.tar.gz

set -e

CONTAINER=mmh_qwen_opt
IMAGE=hy3-vllm:latest
TARFILE=hy3-vllm.tar

echo "=== 1. 将容器保存为镜像 ==="
docker commit $CONTAINER $IMAGE
echo "镜像已创建: $IMAGE"
echo ""

echo "=== 2. 导出为 tar（可选，用于分发） ==="
docker save -o $TARFILE $IMAGE
echo "导出完成: $TARFILE ($(du -h $TARFILE | cut -f1))"
echo "压缩中..."
gzip -f $TARFILE
echo "压缩完成: $TARFILE.gz ($(du -h $TARFILE.gz | cut -f1))"
echo ""

echo "=== 3. 推送到 Harbor（可选） ==="
echo "如需推送到镜像仓库供其他机器拉取："
echo "  docker tag $IMAGE harbor.sourcefind.cn:5443/dcu/hy3-vllm:latest"
echo "  docker push harbor.sourcefind.cn:5443/dcu/hy3-vllm:latest"
echo ""

echo "=== 分发方式 ==="
echo "方式 A — tar 包（离线环境）："
echo "  scp $TARFILE.gz 目标机器:/tmp/"
echo "  ssh 目标机器 'gunzip -c /tmp/$TARFILE.gz | docker load'"
echo ""
echo "方式 B — 镜像仓库（在线环境）："
echo "  docker pull harbor.sourcefind.cn:5443/dcu/hy3-vllm:latest"
echo ""

echo "=== 其他机器运行 ==="
echo "docker run -itd \\"
echo "    --name hy3 \\"
echo "    --privileged \\"
echo "    --network=host \\"
echo "    --ipc=host \\"
echo "    --device=/dev/kfd \\"
echo "    --device=/dev/mkfd \\"
echo "    --device=/dev/dri \\"
echo "    --group-add video \\"
echo "    --cap-add=SYS_PTRACE \\"
echo "    --security-opt seccomp=unconfined \\"
echo "    --shm-size=200g \\"
echo "    -v /path/to/Hy3-Channel-INT8-w8a8:/model:ro \\"
echo "    $IMAGE"
echo ""
echo "进入容器后："
echo "  cd /workspace/vllm-hy3"
echo "  vim deploy/env.sh          # 修改 IP、网卡"
echo "  bash deploy/run_pp2_80l.sh # 启动服务"
