# HY3 80 层 Pipeline Parallelism 双机部署指南

**目标**：两台 K500SM_AI (4×64GiB) 通过 PP=2 加载完整 80 层模型处理长序列

**节点信息**：

| 角色 | IP | 主机名 | GPU |
|------|-----|--------|-----|
| Node 0 (Head) | 10.18.17.71 | worker24 | 4 × K500SM_AI 64GiB |
| Node 1 (Worker) | 10.18.17.74 | worker24 | 4 × K500SM_AI 64GiB |

**环境**：Docker 容器，vLLM 0.18.1+das.dtk2604，DCU (Hygon)，RCCL 2.22.3

**架构**：

```
Node 0 (10.18.17.71)                    Node 1 (10.18.17.74)
┌───────────────────────────┐           ┌───────────────────────────┐
│ GPU0  GPU1  GPU2  GPU3    │           │ GPU0  GPU1  GPU2  GPU3    │
│  TP=4 (层内切分)           │           │  TP=4 (层内切分)           │
│                           │  hidden   │                           │
│ Embedding                 │  states   │                           │
│ 层 0 ──→ 层 39            │ ──P2P──→  │ 层 40 ──→ 层 79           │
│                           │           │ Final Norm + lm_head      │
│ 权重 ~39GB / GPU          │           │ 权重 ~39GB / GPU          │
│ KV Cache 余量 ~25GB / GPU │           │ KV Cache 余量 ~25GB / GPU │
└───────────────────────────┘           └───────────────────────────┘
```

---

## 步骤 0：确认都在容器内操作

**Ray 直接装在容器里**，不要装宿主机。vLLM、Python、GPU 驱动都在容器内，Ray 必须在同一环境运行。

```bash
# 确认当前在容器里（不是宿主机）
echo $HOSTNAME && python3 -c "import torch; print('vLLM OK:', torch.cuda.device_count(), 'GPUs')"
```

## 步骤 1：安装 Ray

```bash
# === 两台机器都在容器内执行 ===
pip install ray -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 如果清华源没有对应 DCU 架构的 wheel，去掉 `-i` 参数用默认源。

**网络注意**：容器需要能访问外网（或内网 PyPI 镜像）。如果容器网络受限，先在宿主机下载 wheel 再拷贝进容器：
> ```bash
> # 宿主机
> pip download ray -d /tmp/ray_pkgs/
> # 拷贝进容器后
> pip install /tmp/ray_pkgs/*.whl
> ```

**验证**：

```bash
python3 -c "import ray; print(ray.__version__)"
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 步骤 2：网络连通性验证

Node 0 (10.18.17.71) 上执行：

```bash
python3 -c "
import socket
targets = [('10.18.17.74', 22), ('10.18.17.74', 6379)]
for host, port in targets:
    try:
        s = socket.create_connection((host, port), timeout=3)
        print(f'{host}:{port} -> OK')
        s.close()
    except Exception as e:
        print(f'{host}:{port} -> FAIL ({e})')
"
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 步骤 3：启动 Ray 集群

### 3a. Node 0 启动 Head

```bash
# 10.18.17.71
ray stop -f   # 先停掉旧的（如果有）
ray start --head \
    --node-ip-address=10.18.17.71 \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --num-gpus=4
```

### 3b. Node 1 加入 Worker

```bash
# 10.18.17.74
ray stop -f
ray start --address='10.18.17.71:6379' \
    --node-ip-address=10.18.17.74 \
    --num-gpus=4
```

### 3c. 验证集群

```bash
# Node 0 上执行
ray status
```

预期输出：

```
Node status
---------------------------------------------------------------
Active:
 1 node_xxx 10.18.17.71  (4 GPUs)
 1 node_xxx 10.18.17.74  (4 GPUs)
---------------------------------------------------------------
Total: 2 nodes, 8 GPUs
```

再验证资源：

```bash
python3 -c "
import ray
ray.init(address='auto', ignore_reinit_error=True)
print('Cluster resources:', ray.cluster_resources())
print('Nodes:', len(ray.nodes()))
print('Total GPUs:', ray.cluster_resources().get('GPU', 0))
ray.shutdown()
"
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 步骤 4: RCCL 多机通信配置

两台都要设置：

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_PROTO=simple
export HSA_FORCE_FINE_GRAIN_PCIE=1

# 确认环境变量生效
env | grep NCCL
```

> 如果不知道网卡名，先跑：
> ```bash
> python3 -c "
> import socket
> s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
> s.connect(('10.18.17.74', 1))
> ip = s.getsockname()[0]
> print(f'Source IP: {ip}')
> s.close()
> "
> ```
> 然后在 `/proc/net/fib_trie` 或 `ip addr` 中查对应网卡。没有 `ip` 命令就用 Python：
> ```bash
> python3 -c "
> import socket, fcntl, struct
> def get_ifname(ip):
>     for ifname in socket.if_nameindex():
>         name = ifname[1]
>         try:
>             s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
>             addr = socket.inet_ntoa(fcntl.ioctl(
>                 s.fileno(), 0x8915,
>                 struct.pack('256s', name[:15].encode())
>             )[20:24])
>             if addr.startswith(ip.split('.')[0]):
>                 print(f'{name} -> {addr}')
>         except:
>             pass
> get_ifname('10.18.17.71')
> "
> ```

**状态**：□ 未执行  □ 成功  □ 失败 _______ 网卡名：_______

---

## 步骤 5: vLLM 多机通信测试

先不启动完整推理，只跑一个通信测试确认 RCCL 跨节点正常：

```bash
# Node 0 上执行
python3 << 'EOF'
import ray
import torch
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "10.18.17.71"
os.environ["MASTER_PORT"] = "29500"
os.environ["WORLD_SIZE"] = "8"
os.environ["RANK"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"

ray.init(address="auto")

@ray.remote(num_gpus=1)
class NCCLTest:
    def __init__(self, rank):
        self.rank = rank
        torch.cuda.set_device(0)
        
    def init_and_test(self):
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://10.18.17.71:29500",
            world_size=8,
            rank=self.rank,
        )
        t = torch.ones(1024).cuda() * (self.rank + 1)
        dist.all_reduce(t)
        expected = sum(range(1, 9))
        ok = t[0].item() == expected
        print(f"Rank {self.rank}: all_reduce result={t[0].item()}, expected={expected}, OK={ok}")
        dist.destroy_process_group()
        return ok

# 创建 8 个 actor（每台机器 4 个 GPU）
tasks = [NCCLTest.options(
    resources={"node:10.18.17.71": 0.001} if i < 4 else {"node:10.18.17.74": 0.001}
).remote(i) for i in range(8)]

results = ray.get([t.init_and_test.remote() for t in tasks])
print(f"All ranks passed: {all(results)}")
ray.shutdown()
EOF
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 步骤 6：启动 vLLM 推理服务

```bash
# Node 0 上执行
python -m vllm.entrypoints.openai.api_server \
    --model /data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 步骤 7：推理测试

```bash
# 发送请求测试
curl http://10.18.17.71:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy3",
    "prompt": "中国的首都是",
    "max_tokens": 10,
    "temperature": 0
  }'
```

**状态**：□ 未执行  □ 成功  □ 失败 _______

---

## 故障排查

### Ray 启动失败
```bash
# 检查日志
ray status --verbose
# 查看 dashboard
curl http://10.18.17.71:8265/api/nodes
# 确认防火墙/端口
python3 -c "import socket; s=socket.create_connection(('10.18.17.74',6379),timeout=3); print('OK'); s.close()"
```

### RCCL 通信失败
```bash
# 检查 NCCL 调试输出
export NCCL_DEBUG=TRACE
# 确认网卡正确
export NCCL_SOCKET_IFNAME=eth0  # 可能需要改成 bond0 / ens 等
# 如果一直超时，尝试降级
export NCCL_PROTO=simple
export NCCL_ALGO=ring
```

### vLLM 模型加载 OOM
```bash
# 降低显存使用率
--gpu-memory-utilization 0.75
# 减少 max-model-len
--max-model-len 4096
# 尝试 PP=4
--pipeline-parallel-size 4 --tensor-parallel-size 2
```

### PP 通信慢
```bash
# PP 只在层边界传一次 hidden_states，本身不会成为瓶颈
# 如果延迟异常，检查：
# 1. 两台机器是否同交换机
# 2. 是否有防火墙/安全组拦截 GPU Direct RDMA
# 3. NCCL_PROTO=simple 降级为 TCP 后的带宽
```

---

## 备选方案：手动 PP（不依赖 Ray）

如果 vLLM + Ray 在 DCU 上不兼容，写一个手动流水线脚本：

```python
# 基本思路：
# Node 0: 加载 config(num_hidden_layers=40) 的模型前半
# Node 1: 加载 config(num_hidden_layers=40) 的模型后半（embedding 替换为 identity）
# Node 0 跑完 layer 39 后，用 sockets 把 hidden_states 发给 Node 1
# Node 1 从 layer 40 开始继续推理
```

此方案需要的修改：
1. 拆分 safetensors 为前半/后半两份（按层分组）
2. 两份 config.json（num_hidden_layers=40）
3. Node 1 的 embedding 层作为 identity（直接透传 Node 0 的 hidden_states）
4. 用 Python socket / HTTP 传递 hidden_states（只有 3×4096×2 bytes ≈ 24KB）

这个备选没有 Ray/RCCL 依赖，两台独立运行，更稳定但吞吐量减半。

---

## 环境快照

```
Date: 2026-08-03
Model: Hy3-Channel-INT8-w8a8 (80 layers, 192 experts, topk=8)
vLLM: 0.18.1+das.dtk2604
Python: (填实际版本)
DCU Driver: (填实际驱动版本)
NCCL/RCCL: 2.22.3
GPU: K500SM_AI × 8 (2 nodes × 4 GPUs)
```
