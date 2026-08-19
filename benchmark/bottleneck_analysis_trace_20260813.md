# 瓶颈分析：基于 torch1_cudagraph_20260813_2 trace（2026-08-13）

## 分析方法

- 数据源：`profiles/torch1_cudagraph_20260813_2/`，取 PP 两侧代表卡 rank0（PP0 前 40 层）与 rank4（PP1 后 40 层）的 trace + 4 个 profiler_out 汇总
- 方法：GPU kernel 事件合并区间求忙碌率；gloo 通信事件（user_annotation）与 GPU 空闲段（gap）做时序对齐
- 背景：8k benchmark TPOT 79ms（11.7-12.1 tok/s），16k TPOT 86ms（11.5 tok/s）

## 核心发现：GPU 利用率只有 ~45%

| 指标 | PP0（rank0） | PP1（rank4） |
|------|------|------|
| GPU 忙碌时间 | 1.25s | 1.19s |
| 窗口墙钟 | 2.75s | 2.74s |
| **GPU 利用率** | **45.4%** | **43.3%** |

**两张卡都有一半以上时间在空转**。decode 每步周期 ~86ms，其中 GPU 实际计算仅 ~37ms，~48ms 是空转等待——这正是 TPOT 79-86ms 的来源。

## decode 每步 ~86ms 的时间构成

| 阶段 | 耗时 | 说明 |
|------|------|------|
| PP0 计算前 40 层 | ~37ms | GPU 忙碌 |
| PP0 → PP1 传输 + 等待 | ~48ms | **PP1 GPU 完全空转** |
| PP1 计算后 40 层 | ~37ms | **PP0 GPU 完全空转** |
| token id 广播 + 同步 | ~10ms | 含调度开销 |

**关键证据（时序对齐）**：
- PP1 侧 64 次大 `gloo:recv`（平均 40.7ms，间隔 ~50ms 均匀），31 个 GPU 大空闲段（平均 52.2ms）与 recv **完全重叠**——PP1 的 GPU 在等数据期间零活动
- PP0 侧 31 个大空闲段（平均 45.7ms），期间 CPU 上只有 ~2ms 的 `gloo:send`——PP0 算完发出后就在干等
- PP0 的 send 本身极快（p50 0.08ms），**传输不是瓶颈，等待才是**

## 瓶颈排序

### 1. PP 串行化（最大，占每步 ~48ms）

PP=2 下 decode 每步是严格的串行链：PP0 算完 → 传 PP1 → PP1 算完 → 采样 token id 广播回 PP0 → 下一步。**PP1 算的时候 PP0 在等 token id，PP0 算的时候 PP1 在等 hidden states，两侧计算零重叠**。

- 结构根源：下一步 decode 的输入 token 来自 PP1 的采样结果，PP0 必须等广播；`--no-async-scheduling`（必须保留，async 模式更糟 1.45s/步）下每步还有全局同步
- **优化空间：~2 倍吞吐**。理想流水（PP0 预计算下一步 + PP1 算当前步）后每步 ~37-40ms → 25 tok/s
- 具体手段（按可行性排序）：微批多请求交错填充空转；PP0 用上一步 token 投机预计算（即 MTP/推测解码的用武之地）；调度器允许 PP0 在等广播期间预取 KV cache

### 2. Gloo PP 通信慢路径（占 recv 等待的一部分）

- `gloo:recv` 的 CUDA 侧时间 39ms/次（32 次共 1.248s，profiler 汇总）——含 GPU↔CPU 拷贝与同步等待；而 PP0 的 `gloo:send` CPU 侧仅 ~1ms
- 39ms 的 recv 里大部分是等待 PP0 完成计算（~37ms），但 **Gloo 的 CPU 中转模式**（GPU→CPU 内存→网络→CPU 内存→GPU）在 prefill 大张量时问题更严重：8k 长 prefill TTFT 稳态 5.5s、16k 12.5s 均由此而来
- **优化空间**：RCCL P2P graph bug 修复后切 NCCL 走 GPU direct（RDMA），prefill TTFT 有望 <1s；decode 每步可省下 Gloo 同步开销 ~10ms 量级

### 3. GPU 计算本身（~37ms/步，其中 PP1 侧）

profiler 汇总（窗口内 kernel 时间）：

| kernel 类别 | 时间 | 占比 |
|------|------|------|
| gloo:recv（GPU 侧） | 1.248s | 51% |
| NCCL TP allreduce（Generic_4） | 392ms | 16%（3840 次 × 102μs） |
| MoE 计算（scaled_mm 300 + fused_moe 93 + topk 90 + quant 35） | ~518ms | 21% |
| GEMM（Cijk）+ attention + 其他 | ~250ms | 10% |

计算侧无单点异常：MoE INT8 GEMM 58.7μs/次、NCCL TP 通信 102μs/次都属正常范围。**计算优化（如 TP 通信与 MoE 重叠）最多再挤 ~15%，不是主要矛盾**。

## 结论与预期收益

当前 11.5-12.1 tok/s 的效率损失链条：**PP 串行化（-50%）> Gloo 慢（-10~15%）> 计算/TP 通信（-10%）**。

| 优化项 | 手段 | 预期 TPOT | 预期吞吐 |
|------|------|------|------|
| 现状 | — | 79-86ms | 11.5-12.1 tok/s |
| 只修 Gloo→NCCL | RCCL P2P bug 修复后切换 | ~70ms | ~14 tok/s |
| 只破 PP 串行 | 微批/投机流水 | ~40ms | ~25 tok/s |
| 两者叠加 | 组合 | ~35-40ms | 25-28 tok/s |

**主攻方向：PP 流水线重叠**（微批或投机），它是 2 倍的量级；NCCL 切换是第二步（同时解决长 prefill TTFT 的 5.5s/12.5s 异常）。MTP 投机解码恰好能同时服务这两个目标（PP0 猜测 token 消除广播等待），值得优先修复 drafter PP 兼容 bug 后评估。
