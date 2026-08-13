# vllm-hy3: vLLM Adaptation for Tencent HY3 on Hygon DCU（cuda-graph 分支）

将 Tencent HY3 大模型（`HYV3ForCausalLM`）的 vLLM 推理框架适配到海光 DCU（K100 / gfx928）平台，支持 TP=4/8 单机推理和 PP=2 双机流水线推理。

**本分支（cuda-graph）**：在 main 分支基础上开启 CUDA graph（`-O1` PIECEWISE 模式），实测输出正确、**12.5 tok/s**（对比 enforce-eager 5.9 tok/s，快 2.2 倍）。配置方法见下文，完整问题排查记录见 [`docs/CUDA_Graph_问题与修复_完整记录_20260812.md`](docs/CUDA_Graph_问题与修复_完整记录_20260812.md)。

---

## 与无 CUDA Graph 版本相比，修改了哪些内容

### 1. 启动参数变化

| 项目 | 无 CUDA graph（main 分支） | CUDA graph（本分支） | 原因 |
|------|---------------------------|---------------------|------|
| 编译模式 | `--enforce-eager` | `-O1`（torch.compile + CUDA graph） | 开启 CUDA graph 的入口 |
| 调度模式 | 默认 async | **`--no-async-scheduling`（必需）** | async 下 PP 采样 token id 的 Gloo 广播每步串行 1.45s（问题 7） |
| `--max-model-len` | 8192 / 262144 | **8192** | graph 模式多尺寸图变体锁定显存，32k 会 OOM（问题 2） |
| 分布式超时 | 默认 600s | **`--distributed-timeout-seconds 1800`** | 首次 torch.compile 10-15 分钟超默认超时（问题 3） |
| RPC / 心跳超时 | 默认 | `VLLM_RPC_TIMEOUT=1800000`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`、`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`（env.sh） | 防止长时间编译/推理被 watchdog 误杀 |
| dump 环境变量 | 默认开启 | **移除** | dump hooks 的 `open()` 干扰 graph capture（问题 1） |

### 2. 源码修改（运行时补丁，两端节点都要生效）

| 文件（`vllm/vllm/` 下） | 修改内容 | 解决的问题 |
|--------------------------|----------|------------|
| `distributed/parallel_state.py` (~1627 行) | PP group backend 从 NCCL 改为 **Gloo** | RCCL P2P 在 graph replay 下每步 poll 32.6s（问题 5） |
| `compilation/cuda_graph.py` | 记录最后一次 replay 所在的 stream（`_last_replay_stream`），replay 前后与默认 stream 双向 join | 乱码的 stream 竞争（问题 6） |
| `v1/worker/gpu_worker.py` (~869 行) | `isend_tensor_dict` 前对 `_last_replay_stream.synchronize()` | Gloo D2H 拷贝不等待 stream 事件导致乱码（问题 6） |
| `v1/worker/gpu_model_runner.py` | 新增 `VLLM_HY3_SKIP_PP_TOKID_BCAST` 开关（仅 sync scheduling 下安全） | 每步 Gloo 广播串行（问题 7） |

> **注意**：`dist/` 下的 wheel **已包含上述全部补丁（2026-08-13 重新打包）**，`pip install dist/*.whl` 即可获得 CUDA graph 能力，详见[安装包更新说明](#安装包更新2026-08-13)。若环境是 2026-08-13 之前安装的旧 wheel，需对**运行时安装副本**（`/usr/local/lib/python3.10/dist-packages/vllm/`，node0 宿主机 + node1 Docker 容器各一份）手动打补丁或重装新 wheel。仓库中的 `vllm/` 目录是同步了同样补丁的源码快照，供参考。

### 3. 新增文件

| 文件 | 说明 |
|------|------|
| `benchmark/cudagraph_bench.py` | streaming TTFT/TPOT/吞吐测量脚本 |
| `benchmark/cudagraph_results_20260812_021100.json` | enforce-eager 基准结果 |
| `docs/CUDA_Graph_Benchmark_实验报告_20260812.md` | 6 次部署尝试的完整实验记录 |
| `docs/CUDA_Graph_问题与修复_完整记录_20260812.md` | 全部问题、根因、修复的完整记录 |

### 4. 性能对比（max_model_len=8192，PP=2 TP=4 双机）

| 模式 | 生成速度 | 说明 |
|------|---------|------|
| enforce-eager（无 graph） | 5.9 tok/s | 每步 kernel launch 开销大 |
| CUDA graph 修复前 | 0.1-0.6 tok/s | 三个 bug 叠加（问题 5/6/7） |
| **CUDA graph 修复后** | **12.5 tok/s** | 首 token 0.12s，每 token 0.08s，并发 4 请求 ~10 tok/s |

---

## CUDA Graph 配置的问题与解决

开启 CUDA graph 过程中共遇到 7 个问题，全部已解决。完整排查过程（含 trace 分析、复现实验、代码补丁）见 [`docs/CUDA_Graph_问题与修复_完整记录_20260812.md`](docs/CUDA_Graph_问题与修复_完整记录_20260812.md)，这里给出摘要：

| # | 问题 | 症状 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | 捕获阶段崩溃 | capture 时 worker 崩溃 | dump 环境变量干扰 capture | 移除 dump 变量；需 dump 时用 `--enforce-eager` |
| 2 | 长上下文 OOM | 32k 上下文显存不足 | graph 多尺寸变体锁定 activation | `--max-model-len 8192` |
| 3 | 分布式超时 | SEND/RECV 超时崩溃 | 首次编译 10-15 分钟 > 默认 600s | 四重超时全部调大 |
| 4 | 第二次请求挂起 | 首个请求成功，之后永久挂起 | RCCL P2P bug 表象（每 token 32.6s） | 由问题 5 的修复一并解决 |
| 5 | **性能退化 50x** | 0.1 tok/s，GPU 空闲，CPU 空转 | **RCCL 2.22.3 P2P 在 graph replay 下两端 kernel 同时 poll 32.6s**（2^15 ms 周期） | **PP group backend 改 Gloo** |
| 6 | **输出乱码** | 速度正常但输出随机字符 | **Gloo 的 D2H 拷贝不等待 capture stream**（上游 vLLM 在 PP 模式下本就禁用 CUDA graph，stream 同步从未被处理） | **isend 前精确同步 replay stream** |
| 7 | **每步 1.45s 串行** | 修复后仅 0.65-1.5 tok/s | PP 采样 token id 的 Gloo 广播每步执行（async 下功能必需，不能跳过） | **`--no-async-scheduling`** |

### 最终可用启动参数（双机 PP=2 TP=4）

```bash
python3 -u -m vllm.entrypoints.openai.api_server \
  --model /path/to/hy3-model \
  --pipeline-parallel-size 2 --tensor-parallel-size 4 \
  --nnodes 2 --node-rank 0 \
  --master-addr <NODE0_IP> --master-port 29517 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hy_v3 \
  --distributed-timeout-seconds 1800 \
  --no-async-scheduling -O1 --port 8000
```

（`deploy/run_pp2_80l.sh` 已固化上述参数，直接 `bash deploy/run_pp2_80l.sh` 即可。）

---

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
| **vLLM wheel** | 海光定制版 vLLM，包含 HY3 模型文件及全部修改 |

安装 vLLM wheel(本地实验的配置打包的)：

```bash
pip install dist/vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

> **注意**：`pip install` 该 wheel 后，vLLM 即安装完毕，无需再单独安装其他 vLLM 包。仓库中的 `vllm/` 目录是上游源码快照，仅用于代码参考和 HY3 模型文件查阅。

#### 安装包更新（2026-08-13）

`dist/` 下的 wheel 已更新为包含 CUDA graph 补丁的版本：

| 版本 | 构建时间 | 内容 |
|------|---------|------|
| 旧版 | 2026-08-07 | 仅 HY3 模型层适配（`hy_v3.py` 等），**不含** CUDA graph 补丁 |
| 新版 | 2026-08-13 | 旧版全部内容 + 4 个 CUDA graph 补丁（Gloo PP 通信、replay stream 同步、tokid 广播开关） |

重新打包方式：CUDA graph 的 4 个补丁全部是纯 Python 文件，C++ 扩展（`_C.abi3.so`、`_moe_C.abi3.so`、`cumem_allocator.abi3.so`）未改动，因此**无需重编译**，解包旧 wheel → 替换 4 个补丁文件 → `wheel pack` 重新打包即可（在仓库根目录执行）：

```bash
# 1. 解包旧 wheel
rm -rf /tmp/wheel_repack && mkdir -p /tmp/wheel_repack && cd /tmp/wheel_repack
unzip -q dist/vllm-0.18.1+das.dtk2604.hy3-*.whl

# 2. 用源码树中打补丁后的 4 个文件覆盖（转 LF 行尾）
for f in vllm/distributed/parallel_state.py \
         vllm/compilation/cuda_graph.py \
         vllm/v1/worker/gpu_worker.py \
         vllm/v1/worker/gpu_model_runner.py; do
    tr -d '\r' < ../vllm/$f > $f
done

# 3. 重新打包（wheel pack 自动重建 RECORD 校验和）
python3 -m wheel pack . --dest-dir dist/

# 4. 恢复原文件名（wheel pack 按 WHEEL 内 Tag 生成 linux_x86_64 名，改回 manylinux 名）
cd dist && mv vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-linux_x86_64.whl \
   vllm-0.18.1+das.dtk2604.hy3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

> **打包注意事项**：wheel 内有两个 DCU 定制文件必须保留原样，不要用上游源码快照覆盖——`vllm/version.py`（含 `__hcu_version__`，注释掉了上游的 `assert __version_tuple__[0] == 0`）和 `vllm/platforms/rocm.py`（注释掉了 `import vllm._rocm_C`）。补丁后的 4 个文件以 LF 行尾打包，其余文件与旧 wheel 逐字节一致（已校验 RECORD 全部 1986 条 sha256）。

### 2.3 模型权重

需要 HY3 INT8 W8A8 量化模型（compressed-tensors 格式），放置于所有节点可访问的路径（本地或 NFS）。

## 3. 快速开始

### 3.1 克隆仓库（本分支）

```bash
git clone -b cuda-graph https://github.com/miaozw66/hy3-vllm-dcu
cd hy3-vllm-dcu
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
| `GPU_COUNT` | 每节点 GPU 数量 | `4` |

### 3.3 验证 RCCL 通信（单机）

```bash
python3 tools/test_rccl_single.py
```

预期输出：`SUCCESS: Single-node RCCL works!`

### 3.4 启动推理服务（CUDA graph 模式）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/hy3-model \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --no-async-scheduling \
  -O1 \
  --port 8000
```

> 无 CUDA graph 的对照配置（main 分支）：`--enforce-eager` 替代 `-O1`，无需 `--no-async-scheduling`。**单机 TP=8 + CUDA graph 尚未验证**（本次验证的是双机 PP=2 TP=4），如遇问题请参考完整记录文档排查。

### 3.5 测试推理

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hy3","prompt":"中国的首都是","max_tokens":5}'
```

## 4. 部署场景

### 4.1 单机 TP=8（8 卡，完整 80 层，可能有误，没在单机8卡启动过）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --no-async-scheduling \
  -O1 \
  --port 8000
```

如需更长的上下文，逐步增大 `--max-model-len` 并调低 `--gpu-memory-utilization`（注意 CUDA graph 模式 32k 会 OOM，见问题 2）。

### 4.2 双机 PP=2（每机 4 卡，完整 80 层，已验证 CUDA graph）

```bash
# 1. 确认 env.sh 中 NODE1_IP、DOCKER_NAME、NIC 配置正确
# 2. 启动双机 PP=2
bash deploy/run_pp2_80l.sh
```

此脚本会自动：
- 通过 SSH 在 Node 1 上启动 PP follower
- 在 Node 0 上启动 PP leader
- 等待服务就绪并输出健康检查命令

**NIAH 性能测试版（256K 上下文）：**

```bash
bash deploy/run_pp2_80l_niah.sh
```

### 4.3 调试模式（渐进增大 max-model-len）

```bash
bash deploy/run_debug_pp2.sh 8192    # 小上下文快速启动
bash deploy/run_debug_pp2.sh 32768   # 中等上下文
bash deploy/run_debug_pp2.sh 131072  # 大上下文
```

## 5. 双机 PP=2 环境要求

### 5.1 网络

- 两台机器需通过 PCIe 网络互联，RCCL 使用 TCP 通信
- 需要免密 SSH（Node 0 → Node 1）
- 模型路径需要在两台机器上一致（NFS 挂载或各自存放）

### 5.2 RCCL 验证（双机）

```bash
# 在 Node 0 上:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 0

# 在 Node 1 上:
RCCL_MASTER_ADDR=<主节点IP> python3 tools/test_rccl_direct.py 1
```

## 6. 性能测试

### 6.1 GPU 监控

```bash
# 监控 600 秒，每 10 秒采样
python3 tools/monitor_gpu.py 600 10

# 双机监控（通过环境变量配置节点列表）
MONITOR_NODES='[{"host":"192.168.1.100","type":"local","docker":""},
               {"host":"192.168.1.101","type":"remote","docker":""}]' \
python3 tools/monitor_gpu.py 600 10
```

### 6.2 CUDA Graph Benchmark

```bash
python3 benchmark/cudagraph_bench.py --endpoint http://localhost:8000 --tokens 50
```

### 6.3 NIAH (Needle in a Haystack) 测试

```bash
python3 benchmark/niah_test.py --endpoint http://localhost:8000 --lengths 4096,8192,16384,32768,65536,131072,262144
```

## 7. 故障排查

### CUDA graph 相关问题

按症状查找对应问题编号（详见文首问题总表或完整记录文档）：

| 症状 | 问题编号 | 解决 |
|------|---------|------|
| capture 阶段崩溃 | 1 | 移除 dump 环境变量 |
| 启动后首个请求极慢（~50s/token） | 4 / 5 | 检查 PP group 是否改为 Gloo |
| 输出乱码但速度正常 | 6 | 检查 isend 前 stream 同步补丁是否两端生效 |
| 速度仅 ~1 tok/s、CPU 满、GPU 空闲 | 7 | 加 `--no-async-scheduling` |
| SEND/RECV 超时崩溃 | 3 | 四重超时调大（env.sh） |

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

## 8. 移植到新机器 — 步骤清单

将本项目移植到另一台海光 K100 DCU 机器的完整步骤：

1. **确认硬件**: 8 × K100 AI (gfx928), 单卡 64 GiB
2. **确认软件栈**: DTK 26.04, PyTorch 2.10.0+das, vLLM 0.18.1+das 已安装
3. **克隆仓库**: `git clone -b cuda-graph https://github.com/miaozw66/hy3-vllm-dcu && cd hy3-vllm-dcu`
4. **编辑配置**: 修改 `deploy/env.sh` 中的 IP、NIC、路径、Docker 容器名
5. **验证 RCCL**: `python3 tools/test_rccl_single.py`
6. **安装 vLLM**: `pip install dist/vllm-0.18.1+das.dtk2604.hy3-*.whl`（2026-08-13 起 wheel 已内置 CUDA graph 补丁，无需再手动打补丁）
7. **80 层单机 TP=8**: 按 4.1 节命令启动
8. **80 层双机 PP=2**（推荐，已验证）: `bash deploy/run_pp2_80l.sh`

### 配置模板

复制 OpenCode 配置文件并填入你的模型路径：

```bash
cp configs/opencode.json.template configs/opencode.json
# 编辑 configs/opencode.json，将 <MODEL_PATH> 替换为实际路径
```

## 9. 项目文件结构

```
hy3-vllm-dcu/
├── README.md                     # 本文件
├── deploy/
│   ├── env.sh                    # ★ 集中式机器配置（移植时首先修改此文件）
│   ├── run_pp2_80l.sh            # 双机 PP=2 启动（CUDA graph 模式）
│   ├── run_pp2_80l_niah.sh       # 双机 PP=2 启动（NIAH 性能测试）
│   ├── run_debug_pp2.sh          # 双机 PP=2 调试（可变 max-len）
│   ├── run_pp2_ray_4l.sh         # 单机 PP=2 Ray 调试（4 层子模型）
│   ├── run_pp_dump.sh            # 单机 PP=2 Ray 调试（带 dump）
│   ├── run_tp4_single_4l.sh      # 单机 TP=4 快速验证（4 层子模型）
│   └── start_vllm_pp.sh          # 通用 PP 启动器
├── configs/
│   ├── opencode.json.template    # OpenCode 配置模板
│   └── moe_configs/              # MoE kernel 调优参数
├── tools/
│   ├── test_rccl_single.py       # 单机 RCCL 测试
│   ├── test_rccl_multinode.py    # 双机 RCCL 测试（torchrun）
│   ├── test_rccl_direct.py       # 双机 RCCL 测试（mp.spawn）
│   ├── monitor_gpu.py            # GPU 利用率监控
│   └── summarize_doc.py          # 文档总结测试脚本
├── benchmark/
│   ├── cudagraph_bench.py        # CUDA graph streaming 性能测试
│   ├── cudagraph_results_*.json  # CUDA graph benchmark 结果
│   ├── niah_test.py              # NIAH 性能测试
│   └── niah_results_*.json       # NIAH 历史测试结果
├── docs/
│   ├── CUDA_Graph_问题与修复_完整记录_20260812.md   # ★ CUDA graph 完整排查记录
│   └── CUDA_Graph_Benchmark_实验报告_20260812.md   # CUDA graph benchmark 实验报告
├── outputs/                      # evalscope 运行结果
├── dist/                         # vLLM wheel 安装包（含 CUDA graph 补丁，2026-08-13 更新）
├── upstream/                     # HY3 上游 vLLM 集成改动
└── vllm/                         # vLLM 0.18.1 源码 snapshot（含 HY3 模型文件 + CUDA graph 补丁）
```

## 10. 已知问题与限制

1. **AITER CK kernel 暂未启用**: gfx928 不在上游 `_ON_GFX9` 列表，且 CK kernel 在 gfx928 上无预编译 `.co` 文件，当前部署默认 `VLLM_ROCM_USE_AITER=0`
2. **NFS 权重加载慢**: 多线程加载 (`enable_multithread_load`) 可缓解
3. **CUDA graph 上下文限制**: graph 模式 `--max-model-len` 限 8192（32k 会 OOM）；需要更长上下文时用 `--enforce-eager`（退回 main 分支行为）
4. **单机 TP=8 + CUDA graph 未验证**: 本次验证环境为双机 PP=2 TP=4，单机组合可能遇到未覆盖的问题
5. **CUDA graph 依赖运行时补丁**: 2026-08-13 起补丁已打包进 `dist/` wheel（旧 wheel 需手动打补丁）；重装 vLLM 包或重建容器后请重新安装 `dist/` 下的新 wheel
6. **MoE 调优配置**: `configs/moe_configs/` 中的 device_name=KONGMING 仅匹配海光卡；换 GPU 需重新生成
