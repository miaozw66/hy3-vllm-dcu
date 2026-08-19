# CUDA Graph Benchmark：16k 上下文 + 1k decode（2026-08-13）

## 采集信息

| 项目 | 值 |
|------|-----|
| 时间 | 2026-08-13 ~14:04（正式跑）/ ~14:13（复测） |
| 服务配置 | 80 层 HY3，双机 PP=2 TP=4，`-O1 --no-async-scheduling`（CUDA graph PIECEWISE），`--max-model-len 16384`，**MTP 已禁用**（与 8k 配置一致，仅 max-model-len 不同） |
| 负载 | input 15360 + output 1024 = 16384 token（顶满 16k 窗口），random 数据集，并发 1 |
| 方法 | 官方 `vllm bench serve`（openai backend，`/v1/completions`），与 8k CUDA graph（同日上午）及批A eager（2026-08-10）同方法 |
| 原始结果 | 正式：`openai-infqps-concurrency1-master-20260813-140853.json`（seed 42，3 请求）；复测：`openai-infqps-concurrency1-master-20260813-141816.json`（seed 43，3 请求） |
| 服务状态 | **测试后 16k 服务保留运行**（按用户要求，未恢复 8k） |

## 正式结果（3 请求，seed 42，14:04-14:09）

| 指标 | 值 |
|------|-----|
| 成功率 | 3/3，0 失败 |
| **Output token throughput** | **10.23 tok/s**（峰值 12.00） |
| Total token throughput | 163.63 tok/s |
| **TTFT** | **mean 12588.42 / median 12535.47 / P99 12702.26 ms**（std 仅 83 ms，3 请求全部 ~12.5s） |
| **Mean TPOT** | **85.57 ms**（median 85.66，P99 85.71，std 0.16 ms） |
| Mean ITL | 85.57 ms |

## 复测（3 请求，seed 43，14:13-14:18）

| 指标 | 值 |
|------|-----|
| 成功率 | 3/3，0 失败 |
| **Output token throughput** | **11.51 tok/s**（峰值 12.00） |
| Total token throughput | 184.20 tok/s |
| **TTFT** | **mean 954.71 / median 464.20 / P99 1926.53 ms**——分布宽（约 0.3 / 0.46 / 1.9 s），median 仅 464 ms |
| **Mean TPOT** | **86.01 ms**（median 85.99，P99 86.16，std 0.12 ms） |
| Mean ITL | 86.01 ms |

## 与 8k / eager 对比

| 指标 | eager 批A（8k，07168/1024/1） | CUDA graph 8k（同日） | CUDA graph 16k（本次，复测） |
|------|------|------|------|
| Output tok/s | 2.52 | 11.72-12.09 | **10.23-11.51** |
| TPOT | 393.66 ms | 79.18-80.04 ms | **85.57-86.01 ms** |
| 长 prefill TTFT | 3212 ms（异常） | 稳态 ~5.5s（异常） | **0.3-12.7s（不稳定，见下）** |

- 相对 eager：16k 下 decode 吞吐仍为 eager 同量级窗口的 **4.1-4.6×**，TPOT **4.6×**
- 相对 8k：TPOT 从 ~79.5ms 升到 ~86ms（**+8.6%**），与 attention 计算量翻倍 + Gloo 通信固定开销的预期吻合

## TPOT 分解：8k → 16k

- 8k 稳态 TPOT ≈ 79ms ≈ 36ms 计算 + 43ms Gloo PP 通信（与 profiler 结论一致）
- 16k 稳态 TPOT ≈ 86ms ≈ 43ms 计算（attention 翻倍）+ 43ms Gloo 通信（固定）
- **Gloo 通信在 16k 下仍是每 decode 步 ~43ms 的固定开销，占 TPOT 一半**。切回 NCCL 后 16k decode 同样有望显著提速
- 正式跑 throughput（10.23）低于复测（11.51）**完全由 TTFT 差异解释**：duration 差 33.6s ≈ 3×(12.5-0.95)s，两跑 TPOT 仅差 0.4ms——**decode 本身两跑高度一致，是可靠数据**

## TTFT 异常分析

16k 长 prefill TTFT 呈现两个截然不同的稳定状态：

| 状态 | 表现 | 出现场合 |
|------|------|---------|
| 慢路径 | ~12.5s（std 83ms，极窄） | 正式跑全部 3 请求 |
| 快路径 | 0.3-0.46s（P99 1.9s） | 复测全部 3 请求 |

- **12.5s ≈ 8k 稳态 5.5s × (15360/7168) = 11.8s**（差 ~6%），与"Gloo 传输全部 prefill token 的 hidden states、耗时随 token 数线性扩展"的假设吻合
- 但复测 median 仅 464ms，接近纯计算时间（8k 纯计算 ~0.4s），说明 **Gloo 慢路径并非每次请求必然触发**——正式跑 3 请求全部命中慢路径（确定性 12.5s），10 分钟后复测 3 请求全部命中快路径
- 与 8k 对比：8k 正式 [0.4, 5.4, 5.4]s、复测全 5.5s（当时定稳态 5.5s）；16k 正式全 12.5s、复测全 <2s。两次实验的慢/快路径占比相反，**触发条件未明**（推测与 Gloo 大 buffer 懒初始化、PIECEWISE 图捕获状态或节点间连接状态有关，无法从现有数据确认）
- 影响：长 prompt 场景 TTFT 不可预测（0.3s ↔ 12.7s 之间跳变），但 **decode 吞吐完全不受影响**（TPOT P99 86.16ms 极稳定）
- 与 8k 结论一致：**TTFT 慢路径的根因仍是 Gloo PP 通信**，RCCL P2P graph bug 修复后切回 NCCL 应可同时消除

## 显存观察

- 8k 服务：约 41 GiB/卡；16k 服务：**约 58.7 GiB/卡（64 GiB 的 91%）**，运行稳定无 OOM
- 16k 已接近当前 64 GiB 卡的上限，进一步加长窗口需考虑 KV cache 压缩或换卡

## 结论

1. **decode 吞吐**：16k 上下文 + 1k decode 下 **10.2-11.5 tok/s（TPOT ~86ms，P99 86.2ms）**，为 eager 的 4.1-4.6×；较 8k 仅下降 ~8%，下降全部来自 attention 计算量翻倍，Gloo 通信开销（~43ms/步）不变
2. **长 prefill TTFT 不稳定**：16k 下在 ~0.4s（纯计算）与 ~12.5s（Gloo 线性扩展）之间跳变，两状态各自内部都极稳定；慢路径与 8k 的 5.5s 同源（Gloo 全量 hidden states 传输），触发条件未明
3. **剩余瓶颈 = Gloo PP 通信**：decode 每步 ~43ms 固定开销 + 长 prefill 慢路径均来自 Gloo。RCCL 修复 P2P graph bug 后切回 NCCL，16k decode 有望 16+ tok/s，长 prefill TTFT 有望稳定 <1s
4. **16k 服务已按要求保留运行**（MTP 禁用，max-model-len 16384）
