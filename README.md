# vllm-hy3: vLLM Adaptation for Tencent HY3 on Hygon DCU

将 Tencent HY3 大模型（`HYV3ForCausalLM`）的 vLLM 推理框架适配到海光 DCU（K100 / gfx928）平台，支持 TP=4/8 单机推理和 PP=2 双机流水线推理。

## 1. 模型架构摘要

| 参数 | 值 |
|------|-----|
| 架构 | HYV3ForCausalLM (Dense + MoE Transformer Decoder) |
| 隐藏维度 | 4096 |
| 层数 | 80 decoder layers |
| 注意力头数 | 64 (Q) / 8 (KV) — GQA |
| Head Dim | 128 |
| QK Normalization | 是 |
| RoPE | theta=11,158,840, neox_style=True |
| Layer 0 | Dense SwiGLU MLP (intermediate=13,312) |
| Layer 1-79 | MoE (192 experts, top-8 sigmoid routing) |
| Expert Dim | 1,536 |
| Shared Expert | 1 per MoE layer |
| 词表大小 | 120,832 |
| 量化 | INT8 w8a8 channel-wise (compressed-tensors) |
| MTP | 1 nextn_predict layer (layer 80) |

## 2. 前置条件

### 2.1 硬件

- **GPU**: 8 × 海光 K100 AI (K500SM_AI / gfx928)，单卡 64 GiB 显存
- **单机部署**: 1 台 8 卡机器即可运行 TP=8 全量 80 层
- **双机部署**: 2 台机器（PP=2），每台建议 ≥ 4 卡，机器间通过 PCIe/RoCE 互联

### 2.2 软件栈（DTK和PyTorch需预装,暂时只打包了vLLM）

一个完整的海光 K100 环境需包含以下组件：

| 组件 | 说明 |
|------|------|
| **DTK 26.04** | 海光 DCU 工具链（含 HIP runtime、ROCm 兼容层、RCCL 通信库） |
| **PyTorch 2.10.0+das** | 海光定制版 PyTorch |
| **vLLM wheel** | 海光定制版 vLLM，包含 HY3 模型文件 + AITER CK 算子 + 全部修改 |

安装 vLLM wheel(本地实验的配置打包的)：

```bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

> **注意**：`pip install` 该 wheel 后，vLLM 即安装完毕，无需再单独安装其他 vLLM 包。仓库中的 `vllm/` 目录是上游源码快照，仅用于代码参考和 HY3 模型文件查阅。

### 2.3 模型权重

需要 HY3 INT8 W8A8 量化模型（compressed-tensors 格式），放置于所有节点可访问的路径（本地或 NFS）。

## 3. 快速开始

### 3.1 克隆仓库

```bash
git clone [<repo-url> vllm-hy3](https://github.com/miaozw66/hy3-vllm-dcu)
cd vllm-hy3
```

### 3.2 配置机器信息

编辑 **`deploy/env.sh`**，这是针对双机8卡的配置文件：

```bash
vim deploy/env.sh
```

关键配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `MASTER_ADDR` | 主节点 IP | `192.168.1.100` |
| `NODE1_IP` | 从节点 IP（单机留空） | `192.168.1.101` |
| `NIC` | 通信网卡名 | `eno1` |
| `DOCKER_NAME` | Docker 容器名（裸金属留空） | `""` |
| `MODEL_PATH` | 80 层完整模型路径 | `/data/models/hy3-int8` |
| `GPU_COUNT` | 每节点 GPU 数量 | `8` |

### 3.3 验证 RCCL 通信（单机）

```bash
python3 tools/test_rccl_single.py
```

预期输出：`SUCCESS: Single-node RCCL works!`

### 3.4 启动推理服务

#### 单机 TP=8（推荐 8 卡配置）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/hy3-model \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

#### 单机 TP=4（4 层子模型快速验证）

```bash
bash deploy/run_tp4_single_4l.sh
```

### 3.5 测试推理

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5}'
```

## 4. 启用 AITER CK 加速

gfx928（K100）不被上游 vLLM 的 `_ON_GFX9` 列表识别，导致 AITER CK 加速 kernel 被静默禁用。
使用 `patch_gfx928.py` 修复此问题：

**方法 1: 在启动脚本最前面 import**

```python
import patch_gfx928  # noqa: F401  — 必须在所有 vllm import 之前
```

**方法 2: 通过 PYTHONSTARTUP 自动注入**

```bash
export PYTHONSTARTUP=/path/to/vllm-hy3/patch_gfx928.py
```

**AITER 环境变量参考：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_ROCM_USE_AITER` | `0` | 总开关，设为 1 启用全部 CK 加速 |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | CK INT8 GEMM (Linear 层) |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | CK Fused MoE |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `1` | CK RMSNorm |
| `VLLM_ROCM_USE_AITER_MHA` | `1` | CK Flash Attention |

完整 AITER 选项见 `vllm/vllm/envs.py` 第 101-114 行。

> **注意**: CK kernel 在 gfx928 上无预编译 `.co` 文件，首次运行会触发 JIT 编译，可能导致超时或崩溃。建议先用小 max-model-len 跑一次 warmup。

## 5. 部署场景

### 5.1 单机 TP=8（8 卡，完整 80 层）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

如需更长的上下文，逐步增大 `--max-model-len` 并调低 `--gpu-memory-utilization`。

### 5.2 双机 PP=2（每机 4 卡，完整 80 层）

```bash
# 1. 确认 env.sh 中 NODE1_IP、DOCKER_NAME、NIC 配置正确
# 2. 启动双机 PP=2
bash deploy/run_pp2_80l.sh
```

此脚本会自动：
- 通过 SSH 在 Node 1 上启动 PP follower
- 在 Node 0 上启动 PP leader
- 等待服务就绪并输出健康检查命令

**NIAH 性能测试版（启用 AITER + 256K 上下文）：**

```bash
bash deploy/run_pp2_80l_niah.sh
```

### 5.3 调试模式（渐进增大 max-model-len）

```bash
bash deploy/run_debug_pp2.sh 8192    # 小上下文快速启动
bash deploy/run_debug_pp2.sh 32768   # 中等上下文
bash deploy/run_debug_pp2.sh 131072  # 大上下文
```

## 6. 双机 PP=2 环境要求

### 6.1 网络

- 两台机器需通过 PCIe 网络互联，RCCL 使用 TCP 通信
- 需要免密 SSH（Node 0 → Node 1）
- 模型路径需要在两台机器上一致（NFS 挂载或各自存放）

### 6.2 RCCL 验证（双机）

```bash
# 在 Node 0 上:
RCCL_MASTER_ADDR=<主节点IP> bash tools/test_rccl_multinode.sh 0

# 在 Node 1 上:
RCCL_MASTER_ADDR=<主节点IP> bash tools/test_rccl_multinode.sh 1
```

### 6.3 PP=2 需额外安装的修复

双机 PP=2 时，PP follower 节点的 `kv_cache_coordinator` 可能为空导致崩溃。
需要使用 `reference/submodel_debug/sitecustomize.py` 中的 monkey-patch：

```bash
export PYTHONPATH=/path/to/vllm-hy3/reference/submodel_debug:$PYTHONPATH
```

此文件已在 `.gitignore` 中取消忽略，clone 后会包含。

## 7. 调试与 Dump

### 7.1 启用逐层 Dump

设置环境变量后，每层 7 个关键位置（input、attention、MLP 等）的 tensor 会被保存：

```bash
export VLLM_HY3_DUMP_DIR=/tmp/dumps
export VLLM_HY3_DUMP_SKIP=2  # 跳过前 2 次 forward（warmup）
```

### 7.2 与 MetaInfer Golden Dump 对比验证

```bash
# 对比 PP=2 全量 80 层的 dump 与 golden 参考数据
VLLM_DUMP_DIR=/path/to/vllm/dumps \
VLLM_GOLDEN_DUMP_DIR=/path/to/golden/dumps \
python3 verify/compare_80l_full.py
```

## 8. 性能测试

### 8.1 GPU 监控

```bash
# 监控 600 秒，每 10 秒采样
python3 tools/monitor_gpu.py 600 10

# 双机监控（通过环境变量配置节点列表）
MONITOR_NODES='[{"host":"192.168.1.100","type":"local","docker":""},
               {"host":"192.168.1.101","type":"remote","docker":""}]' \
python3 tools/monitor_gpu.py 600 10
```

### 8.2 NIAH (Needle in a Haystack) 测试

```bash
python3 benchmark/niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192,16384,32768,65536,131072,262144
```

## 9. 故障排查

### AITER 未生效

```bash
python3 -c "import patch_gfx928; from vllm.platforms.rocm import on_gfx9; print('on_gfx9():', on_gfx9())"
# 应输出: on_gfx9(): True
```

### RCCL 初始化超时

1. 确认网卡名正确：`ip addr show` 查看实际网卡名，更新 `env.sh` 中 `NIC`
2. 确认防火墙未阻止端口：`MASTER_PORT` 需在 29500-29600 范围开放
3. 确认 `NCCL_IB_DISABLE=1` 已设置（不使用 InfiniBand）

### OOM (Out of Memory)

- 降低 `--gpu-memory-utilization`（如 0.75）
- 减小 `--max-model-len`
- 如果使用 PP=2，确认每张卡只加载 40 层

### 模型加载缓慢

添加 `--model-loader-extra-config '{"enable_multithread_load": true}'` 启用多线程加载（适用于 NFS）。

### PP=2 follower 节点崩溃

确认已设置 `PYTHONPATH` 包含 `reference/submodel_debug/sitecustomize.py`（参见 6.3 节）。

## 10. 移植到新机器 — 步骤清单

将本项目移植到另一台海光 K100 DCU 机器的完整步骤：

1. **确认硬件**: 8 × K100 AI (gfx928), 单卡 64 GiB
2. **确认软件栈**: DTK 26.04, PyTorch 2.10.0+das, vLLM 0.18.1+das 已安装
3. **克隆仓库**: `git clone <repo-url> && cd vllm-hy3`
4. **编辑配置**: 修改 `deploy/env.sh` 中的 IP、NIC、路径、Docker 容器名
5. **验证 RCCL**: `python3 tools/test_rccl_single.py`
6. **验证 AITER**: `python3 -c "import patch_gfx928; from vllm.platforms.rocm import on_gfx9; assert on_gfx9()"`
7. **准备子模型**（可选，用于快速验证）：提取 4 层子模型权重放到 `submodel_debug/test4/`
8. **4 层快速测试**: `bash deploy/run_tp4_single_4l.sh`
9. **80 层单机 TP=8**: 按 5.1 节命令启动
10. **80 层双机 PP=2**（如需）: `bash deploy/run_pp2_80l.sh`

### 配置模板

复制 OpenCode 配置文件并填入你的模型路径：

```bash
cp configs/opencode.json.template configs/opencode.json
# 编辑 configs/opencode.json，将 <MODEL_PATH> 替换为实际路径
```

## 11. 项目文件结构

```
vllm-hy3/
├── README.md                     # 本文件
├── patch_gfx928.py               # gfx928 AITER monkey-patch
├── deploy/
│   ├── env.sh                    # ★ 集中式机器配置（移植时首先修改此文件）
│   ├── run_pp2_80l.sh            # 双机 PP=2 启动（dump 模式）
│   ├── run_pp2_80l_niah.sh       # 双机 PP=2 启动（NIAH 性能测试）
│   ├── run_debug_pp2.sh          # 双机 PP=2 调试（可变 max-len）
│   ├── run_tp4_single_4l.sh      # 单机 TP=4 子模型快速验证
│   ├── run_pp2_ray_4l.sh         # 单机 PP=2 Ray 调试
│   ├── run_pp_dump.sh            # PP=2 带 dump
│   └── start_vllm_pp.sh          # 通用 PP 启动器
├── configs/
│   ├── opencode.json.template    # OpenCode 配置模板
│   └── moe_configs/              # MoE kernel 调优参数
│       └── E=192,N=384,device_name=KONGMING.json
├── reference/
│   └── submodel_debug/
│       ├── sitecustomize.py      # PP=2 dump 钩子 + follower 修复
│       ├── extract_layers.py     # 子模型提取工具
│       └── ...
├── tools/
│   ├── test_rccl_single.py       # 单机 RCCL 测试
│   ├── test_rccl_multinode.py    # 双机 RCCL 测试
│   ├── test_rccl_direct.py       # 双机 RCCL 测试（mp.spawn 方式）
│   ├── monitor_gpu.py            # GPU 利用率监控
│   └── summarize_doc.py          # 文档总结测试脚本
├── verify/
│   ├── verify.py                 # 逐层对比验证（TP=4）
│   ├── verify_layers_65_79.py    # 层 65-79 手动验证
│   ├── compare_80l_full.py       # 80 层全量对比报告
│   ├── compare_pp_boundary.py    # PP=2 边界 tensor 对比
│   ├── generate_final_report.py  # 合并验证报告
│   ├── output_final_logits.py    # 输出最终 logits
│   ├── auto_verify.sh            # 自动等待服务就绪并验证
│   └── test_and_compare.sh       # 测试请求 + 对比
├── benchmark/
│   └── niah_test.py              # NIAH 性能测试
├── vllm/                         # vLLM 0.18.1 源码 snapshot + HY3 模型文件
└── docs/                         # 移植过程文档（移植记录、调优指南等）
```

## 12. 已知问题与限制

1. **gfx928 不在上游 `_ON_GFX9`**: 使用 `patch_gfx928.py` 修复
2. **CK kernel 无预编译 .co**: gfx928 上首次运行会 JIT 编译，可能不稳定；建议先 warmup
3. **NFS 权重加载慢**: 多线程加载 (`enable_multithread_load`) 可缓解
4. **enforce-eager 禁用 CUDA Graph**: gfx928 上 HIP Graph 尚不稳定，暂时禁用
5. **PP=2 需 monkey-patch**: follower 节点的 kv_cache_coordinator 修复在 `sitecustomize.py` 中
6. **MoE 调优配置**: `configs/moe_configs/` 中的 device_name=KONGMING 仅匹配海光卡；换 GPU 需重新生成
