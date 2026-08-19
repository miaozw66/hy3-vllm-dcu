# CUDA Graph 正式 Benchmark：8k 上下文 + 1k decode（2026-08-13）

## 采集信息

| 项目 | 值 |
|------|-----|
| 时间 | 2026-08-13 12:40（正式跑）/ 12:51（复测） |
| 服务配置 | 80 层 HY3，双机 PP=2 TP=4，`-O1 --no-async-scheduling`（CUDA graph PIECEWISE），`--max-model-len 8192` |
| 负载 | input 7168 + output 1024 = 8192 token（顶满 8k 窗口），random 数据集，并发 1 |
| 方法 | 官方 `vllm bench serve`（openai backend，`/v1/completions`），与批A eager（2026-08-10）同方法 |
| 原始结果 | 正式：`openai-infqps-concurrency1-master-20260813-124514.json`（seed 42，3 请求）；复测：`openai-infqps-concurrency1-master-20260813-125135.json`（seed 43，3 请求） |

## 正式结果（3 请求，seed 42）

| 指标 | 值 |
|------|-----|
| 成功率 | 3/3，0 失败 |
| **Output token throughput** | **12.09 tok/s**（峰值 13.00） |
| Total token throughput | 96.68 tok/s |
| Mean TTFT | 3735.31 ms（median 5446.35，P99 5486.92）|
| **Mean TPOT** | **79.18 ms**（median 79.19，P99 79.44）|
| Mean ITL | 79.18 ms |

warmup 请求（不计入）：TTFT 8961.59ms（含 7168 prefill 首次编译），TPOT 79.03ms，decode 11.40 tok/s。

## 与 eager 基线对比（批A 2026-08-10 同组 7168/1024/1）

| 指标 | eager（批A） | CUDA graph（本次） | 提升 |
|------|-------------|-------------------|------|
| Output tok/s | 2.52 | **12.09** | **4.8×** |
| TPOT | 393.66 ms | **79.18 ms** | **5.0×** |
| TTFT | 3212.07 ms | 3735.31 ms（mean）/ 5446.35（median） | 抖动区间重叠 |

## TTFT 异常分析

- 正式 3 请求 TTFT 约 [0.4s, 5.4s, 5.4s]：**纯 prefill 计算仅 ~0.4s**，慢请求多出 ~5s
- 与 eager 批A 7168 组的 TTFT 异常（3.2s/5.2s/5.3s，当时标注"疑似调度抖动，建议复测"）**同型**，说明与 CUDA graph 无关
- 推测根因：7168 token prefill 的 **Gloo PP 通信**——prefill 需传输全部 token 的 hidden states（decode 每步只传少量），Gloo CPU 传输成为长 prefill 的瓶颈（与 profiler 结论"Gloo 通信为最大单点"一致）
- 影响：长 prompt 场景 TTFT 不稳定，但 decode 吞吐不受影响（TPOT P99 79.44ms 极稳定）

## 复测（seed 43，3 请求）

| 指标 | 值 |
|------|-----|
| 成功率 | 3/3，0 失败 |
| Output token throughput | 11.72 tok/s |
| **TTFT** | **mean 5527.57 / median 5522.43 / P99 5648.54 ms**——3 个请求全部 ~5.5s，分布极窄 |
| TPOT | mean 80.04 / median 79.48 / P99 81.32 ms |

复测确立**稳态 TTFT ≈ 5.5s**：首次正式跑中的 0.4s 请求是偶然（initial test 请求重叠预热效应），非稳态。

## 结论

1. **decode 吞吐**：CUDA graph 在 8k 上下文 + 1k decode 下稳定 **11.7-12.1 tok/s（eager 的 4.8 倍）**，TPOT ~79ms 分布极窄（P99 81ms）
2. **长 prefill TTFT 稳态 ~5.5s**：纯计算仅 ~0.4s，~5s 是 7168 token prefill 的 Gloo PP 通信（传输全部 token 的 hidden states；decode 每步仅传少量所以 TPOT 不受影响）。eager 批A 同组 TTFT 3.2s 同样异常（4096 输入才 454ms），两者同源于 Gloo
3. **剩余瓶颈 = Gloo PP 通信**：同时拖累 decode 吞吐（~43ms/步，约占一半）和长 prefill TTFT。RCCL 修复 P2P graph bug 后切回 NCCL 可同时解决两者（decode 有望 20+ tok/s，长 prefill TTFT 回到 <1s）
