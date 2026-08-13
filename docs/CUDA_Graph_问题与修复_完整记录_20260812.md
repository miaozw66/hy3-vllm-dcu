# CUDA Graph 完整问题与修复记录

> 时间范围：2026-08-11 至 2026-08-12 19:30
> 最终状态：CUDA graph 在双机 PP=2 TP=4 下完全跑通——输出正确、12.5 tok/s（单请求）、首 token 0.12s

## 最终结果一览

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 输出正确性 | 乱码 / 挂起 | ✅ 全部正确（多轮/并发/流式一致） |
| 生成速度 | 0.1-0.6 tok/s | **12.5 tok/s**（并发 4 请求 10 tok/s） |
| 首 token 延迟 | ~2s（甚至永久挂起） | **0.12s** |
| 每步间隔 | 32.7s（RCCL 卡顿期）/ 1.45s（广播期） | **0.08s** |

对比 enforce-eager（无 CUDA graph）基准 5.9 tok/s，CUDA graph 快 **2.2 倍**——kernel launch 开销消除的直接收益。

---

## 问题总表

| # | 问题 | 症状 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | 捕获阶段崩溃 | CUDA graph capture 时 worker 崩溃 | dump 环境变量（`VLLM_HY3_DUMP_DIR` 等）干扰 capture | 移除 dump 变量；需 dump 时用 `--enforce-eager` |
| 2 | 长上下文 OOM | 32k 上下文显存不足 | CUDA graph 多尺寸变体锁定 activation | `--max-model-len 8192` |
| 3 | 分布式超时 | SEND/RECV 超时崩溃 | torch.compile 首次 10-15 分钟 > 默认 600s 超时 | 四重超时全部调大（`--distributed-timeout-seconds 1800` 等） |
| 4 | 第二次请求永久挂起 | 首个请求秒级成功，之后全部挂起 | RCCL P2P graph bug 的表象（每 token 32.6s，50 token 需 27 分钟） | 由问题 5 的 Gloo 修复一并解决 |
| 5 | **性能退化 50x** | CUDA graph 0.1 tok/s，GPU 0.8% 空闲，CPU 110% | **RCCL 2.22.3 P2P 在 graph replay 下两端 kernel 同时 poll 32.6s**（2^15 ms 周期，RCCL proxy 无任何活动） | **PP group backend 从 NCCL 改为 Gloo** |
| 6 | **输出乱码** | 速度正常（4.5 tok/s）但输出随机乱码 | **Gloo isend 的 D2H 拷贝不等待 graph replay 的 capture stream**（上游 vLLM V1 在 PP 模式下明确禁用 CUDA graph——"Pipeline parallel is currently eager-only"，HY3 适配版强行开启的组合从未处理过 stream 同步） | **isend 前对 `_last_replay_stream.synchronize()`**（wait_stream 无效，必须 CPU 阻塞同步） |
| 7 | **每步 1.45s 串行** | 修复乱码后仅 0.65-1.5 tok/s | PP 采样 token id 的 Gloo broadcast 每步执行（async scheduling 下功能必需，跳过会输出重复 token） | **`--no-async-scheduling`**（sync 模式下该广播不再必需） |

---

## 问题 5：RCCL P2P graph bug（性能退化 50x）— 详细记录

### 症状与排查过程

1. **08-12 上午**：CUDA graph capture 成功（86 个图，19-29s），但 replay 阶段 GPU 几乎空闲（0.8%）、CPU 110% 空转、0.1 tok/s。enforce-eager 对照 5.9 tok/s 正常。
2. **测试矩阵**：`FULL_AND_PIECEWISE` 与仅 `FULL`（`-cc.cudagraph_mode=2`）同样退化；`-O1` 单独不解决——只要 graph 开启就退化。
3. **trace 分析**（关键转折）：torch profiler 的 `dur` 单位是**微秒**，之前被误标为毫秒——"32.6ms"实为 **32.6 秒**。慢 SendRecv 的 GPU 时长为 32.6s/65.4s/98.2s（32.6s 的整数倍），与实测 ~52s/token 吻合。
4. **4 层子模型复现**：node0 的慢 kernel 是 SendRecv（GPU 端 poll 32.58s），node1 是 hipEventSynchronize（CPU 端等 32.7s）——**两端 kernel 同时开始、同时 poll、同时完成**，RCCL proxy 无任何传输活动。`NCCL_DEBUG=TRACE` 下 32.6s 期间零日志。
5. **周期特征**：32.6s ≈ 32768ms = 2^15 ms，指向 RCCL 内部轮询超时机制。

### 根因

RCCL 2.22.3（/opt/dtk）的 P2P 通信（PP 层间 send/recv）在 CUDA graph replay 下存在 bug：capture 后惰性创建的 PP communicator 与 graph 交互异常，两端 GPU kernel 同时 poll 一个永不触发的事件，每步固定卡 32.6s。

### 修复：PP 通信改走 Gloo

PP 层间通信量很小（~8KB-64KB/步），Gloo（TCP）跨节点 ms 级延迟完全够用，可彻底绕开 RCCL 的 graph bug。

**补丁位置**：`/usr/local/lib/python3.10/dist-packages/vllm/distributed/parallel_state.py`（运行时安装副本，**node0 宿主机 + node1 Docker 容器两端都要改**）约 1627 行：

```python
# HY3/DCU workaround: RCCL 2.22.3 P2P communicators created after CUDA graph
# capture poll for ~2^15 ms per step (graph replay + lazy PP comm bug).
# PP traffic is small (~KB per step), so route it over Gloo (TCP) instead.
# （PP group 创建时 backend 由 NCCL 改为 Gloo）
```

`pynccl` 用 `cpu_group`（Gloo）做 bootstrap，不依赖 device_group 的 backend，改动安全。

### 验证

- 4 层模型：8 tokens 从 125 秒降至 **1.19 秒**（快 100 倍）
- 80 层模型：16 tokens 3.6 秒（4.5 tok/s），速度恢复

---

## 问题 6：输出乱码（Gloo D2H stream 竞争）— 详细记录

### 症状与排查过程

1. Gloo 修复后速度正常，但 80 层输出乱码（"TER!RATERRERIT..."），4 层模型也乱码。
2. **关键对照实验**：`--enforce-eager`（无 graph）+ Gloo → 输出正常（"北京"）。证明模型/权重/PP 协议无问题，乱码根因是 graph replay 的 stream 竞争。
3. **上游代码证据**：`model_runner.py` 中 "Pipeline parallel is currently eager-only" / "Skipping CUDA graph capture because pipeline parallel"——**上游 vLLM V1 在 PP 模式下明确禁用 CUDA graph**。PP + graph 是 HY3 适配版强行开启的组合，上游从未处理过 stream 同步，`replay()` 后没有任何同步点。
4. **本地复现实验**（决定性）：模拟 capture/replay/D2H 的 stream 交互，确认 **Gloo 的 D2H 拷贝不等待任何 stream 事件**——只等 CPU 侧同步：
   - `default stream` 事件等待（wait_stream）：**无效**
   - `S.wait_stream(R)`：**无效**
   - `R.synchronize()`（CPU 阻塞等 replay stream）：**有效** ✓
   - 全机 `torch.cuda.synchronize()`：有效 ✓ 但破坏流水线（速度掉到 0.6 tok/s）

### 根因

Gloo `isend` 对 CUDA tensor 的 D2H 拷贝不参与 CUDA stream 事件同步。graph replay 在 capture stream 上异步执行时，D2H 会读到未完成的数据，node1 收到半成品 → 乱码。

### 修复：isend 前精确同步 replay stream

**补丁 1**：`/usr/local/lib/python3.10/dist-packages/vllm/compilation/cuda_graph.py`（两端）

```python
_last_replay_stream = None   # 模块级全局

# CUDAGraphWrapper.__call__ 的 replay 之后：
global _last_replay_stream
_last_replay_stream = current_stream()
```

**补丁 2**：`/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_worker.py`（两端，isend 之前约 869 行）

```python
# 只等最后一次 replay 的 stream，避免整机 synchronize 破坏流水线
_lrs = _cg._last_replay_stream
if _lrs is not None:
    _lrs.synchronize()
```

注意：`from ... import _last_replay_stream as _lrs` 会在 import 时绑定值（None），必须用模块引用动态读取。

### 验证

- 输出正确："中国的首都是"→"北京…"、"世界上最高的山峰是"→"珠穆朗玛峰…"，3 次重复完全一致
- stream dump 确认：node0 四个 rank 全部 `torch=vllm=last_replay` 一致，同步精确

---

## 问题 7：PP 广播每步串行 1.45s — 详细记录

### 症状与排查过程

1. 乱码修复后速度仅 0.65-1.5 tok/s。py-spy 抓栈：两端 CPU 都卡在 gloo broadcast 上，GPU 利用率 0%。
2. 定位：node0 每步在 `sample_tokens → _pp_receive_prev_sampled_token_ids_to_input_batch → broadcast` 等待 node1 的采样 token id。PP group 改为 Gloo 后，这个每步的 GPU 通信变成了 CPU TCP 广播。
3. **第一次尝试（失败）**：加 `VLLM_HY3_SKIP_PP_TOKID_BCAST` 跳过广播 → 速度飙升 36 倍（24 tok/s），**但输出重复 token**——async scheduling 下这个广播是功能必需（node0 的 input_ids 需要 node1 的采样结果覆盖，engine 调度有一层延迟）。
4. **正确方案**：`--no-async-scheduling`（sync 模式下该广播不再必需，跳过安全）。

### 修复

启动命令加 `--no-async-scheduling`（已写入 `deploy/run_pp2_80l.sh`）。

### 验证

- 速度：0.65 → **12.5 tok/s**（20 倍）
- 正确性：3 次重复完全一致，无重复 token
- 每步时间戳：node0 replay 34ms + isend 3ms，开头 wait 0ms（isend 异步完成，两端 forward 已重叠）

---

## 其他问题（已解决的启动/环境问题）

### 1. dump 环境变量导致捕获崩溃

`VLLM_HY3_DUMP_DIR` 等 dump 变量（sitecustomize.py 的 hooks）会干扰 CUDA graph 捕获。**移除即可**；如需 dump 日志，用 `--enforce-eager` 禁用 CUDA graph。

### 2. 32k 上下文 OOM

CUDA graph 的 `FULL_AND_PIECEWISE` 模式在 capture 阶段为不同 batch size / sequence length 预录制多个图变体，每个变体锁定中间 activation。32k 下超显存，8k 正常。**固定 `--max-model-len 8192`**。

### 3. 分布式超时（600s）

`--distributed-timeout-seconds` 未设置时 PyTorch default_pg_timeout=600s，torch.compile 首次 10-15 分钟编译必然超时。**四重超时全部配置**（1800s 起）。

### 4. Node 1 僵尸显存（启动失败最常见原因）

ROCm 平台 `pkill -9` 杀 vLLM 进程**不释放 GPU 显存**。`pkill -9 -f "VLLM::"` 也匹配不到 Worker 进程（进程名在 /proc/PID/comm）。正确清理：`pkill -9 -f EngineCore; pkill -9 -f Worker_PP; pkill -9 -f vllm.entrypoints`，Node 1 显存仍需 `docker restart mmh_qwen_opt`。

### 5. NCCL IB/Socket 不一致（跨节点握手失败）

node0 配了 `NCCL_IB_DISABLE=1`（Socket）而 node1 走 NET/IB，握手 `wrong type 3 != 4`。两端环境变量必须一致。

---

## 排查过程中的坑（经验教训）

1. **trace 单位错误**：torch profiler 的 `dur` 是微秒，"32.6ms"实为 32.6 秒——浪费大量时间在错误的量级上分析。
2. **4 层子模型本身是坏的**（eager 模式也乱码），不能用于诊断 graph 机制；乱码排查必须用 80 层（eager 已验证正常）。
3. **health 200 不代表可用**：worker 可能已崩（health 只查 APIServer）。就绪检查要发真实请求验证。
4. **运行时 vllm 在安装副本** `/usr/local/lib/python3.10/dist-packages/vllm/`，不是源码目录；两端（宿主机 + 容器）都要打补丁，且两端文件基线可能不同（2115 vs 2118 行）。
5. **dump 补丁引入死锁**：在 irecv 前提前 wait 导致 node0/node1 互相等待。诊断补丁要谨慎处理通信等待点。
6. **import 绑定陷阱**：`from module import var` 是值绑定，模块内全局更新后旧引用不更新。
7. **跳过功能必需组件前先想清楚**：token id 广播在 async scheduling 下必需，直接跳过会导致重复 token。
8. **健康轮询要防误判**：旧服务临死前的日志可能被新服务的轮询误读。

---

## 最终配置（v4，2026-08-12 19:01 启动，至今运行）

**启动命令**（`deploy/run_pp2_80l.sh` 已固化）：

```bash
python3 -u -m vllm.entrypoints.openai.api_server \
  --model /data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master \
  --pipeline-parallel-size 2 --tensor-parallel-size 4 \
  --nnodes 2 --node-rank 0 \
  --master-addr 10.18.17.71 --master-port 29517 \
  --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hy_v3 \
  --distributed-timeout-seconds 1800 \
  --no-async-scheduling -O1 --port 8000
```

**三处运行时补丁**（均在 `/usr/local/lib/python3.10/dist-packages/vllm/` 安装副本，node0 宿主机 + node1 Docker 容器 `mmh_qwen_opt` 两端同步）：

| 文件 | 补丁 | 作用 |
|------|------|------|
| `distributed/parallel_state.py` (~1627 行) | PP group backend NCCL→Gloo | 绕过 RCCL P2P graph bug |
| `compilation/cuda_graph.py` | `_last_replay_stream` 全局记录 | 供 isend 前同步使用 |
| `v1/worker/gpu_worker.py` (~869 行) | isend 前 `_lrs.synchronize()` | 修复 Gloo D2H 乱码 |

**重启服务时注意**：补丁在安装副本上，重装/升级 vllm 包会丢失；Docker 容器重建也会丢失。备份补丁：见 `deploy/` 下的 patch 脚本（如 `patch_stream_dump2.py`）。

**服务地址**：API `http://10.18.17.71:8000`，Web `http://10.18.17.71:8080`（web_server.py 代理）。
