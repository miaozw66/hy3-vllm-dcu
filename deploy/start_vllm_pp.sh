#!/bin/bash
# Generic PP launcher with auto-detected network interface
#
# Usage:
#   bash start_vllm_pp.sh <node_rank> [additional vLLM args...]
#   bash start_vllm_pp.sh 0
#   bash start_vllm_pp.sh 1 --max-model-len 131072
set -e

source "$(dirname "$0")/env.sh"

# Auto-detect the network interface for the node's primary IP
NODE_IP=$(python3 -c "
import socket, fcntl, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('$MASTER_ADDR', 1))
src_ip = s.getsockname()[0]
s.close()
for ifname in socket.if_nameindex():
    name = ifname[1]
    try:
        ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = socket.inet_ntoa(fcntl.ioctl(ss.fileno(), 0x8915, struct.pack('256s', name[:15].encode()))[20:24])
        ss.close()
        if addr == src_ip:
            print(name)
            break
    except:
        pass
")
echo "Detected interface: $NODE_IP"

export NCCL_SOCKET_IFNAME=$NODE_IP
export NCCL_DEBUG=WARN

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 \
  --node-rank "$1" \
  --master-addr $MASTER_ADDR \
  --master-port 29501 \
  --trust-remote-code \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser hy_v3 \
  ${@:2}
