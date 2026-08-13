# CUDA Graph Profiler Trace 分析（2026-08-13 第 2 批，torch1_cudagraph_20260813_2）

## 采集信息

| 项目 | 值 |
|------|-----|
| 采集时间 | 2026-08-13 11:44 |
| 配置 | 80 层 HY3，双机 PP=2 TP=4，`-O1 --no-async-scheduling`（最终修复配置） |
| 推理请求 | prompt「世界上最高的山峰是」，max_tokens=32，端到端 2.8s |
| 输出正确性 | ✓「珠穆朗玛峰，海拔为8848.86米。……」 |
| profiler | torch profiler，lazy-init（`POST /start_profile` 启动，`/stop_profile` 停止） |
| trace 位置 | `profiles/torch1_cudagraph_20260813_2/`（8 rank × ~2.9MB + 主进程 async_llm trace + 4 个 profiler_out_*.txt） |

trace 文件命名：`dp0_pp{r}_tp{t}_dcp0_ep{e}_rank{n}.{ts}.pt.trace.json.gz`，rank0-3 在 node0（PP0），rank4-7 在 node1（PP1）。

## 性能画像（与第 1 批 20260813_1 一致）

### rank4（PP1，gloo 接收侧）关键算子耗时

| 算子 | 总耗时 | 调用次数 | 折算每步 |
|------|--------|---------|---------|
| **gloo:recv**（PP 通信接收） | **2.70s** | 160 | ~43ms/步（62 decode 步口径） |
| vllm::unified_attention_with_output | 0.475s | 1280 | ~7.7ms/步 |
| ncclDevKernel（TP all-reduce） | 0.392s | 3840 | ~6.3ms/步 |
| scaled_mm_kernel（INT8 GEMM） | 0.301s | 5120 | ~4.9ms/步 |
| unified_kv_cache_update | 0.269s | 1280 | ~4.3ms/步 |
| fused_moe_kernel | 0.093s | 2560 | ~1.5ms/步 |
| hipGraphLaunch（replay 提交） | 0.128s | 1312 | ~2ms/步 |

PP1 侧 execute_context 呈现 40ms / 0.2ms 交替：每 decode 步约 40ms，其中约 20ms 在等 gloo:recv 数据。

### rank0（PP0，gloo 发送侧）

- gloo:send 总计仅 0.040s（128 次，~0.3ms/次）——发送侧开销小，瓶颈在接收侧等待
- 端到端：32 token / 2.8s ≈ 87ms/token（含 prefill 与 profiler 记录开销），与生产 12.5 tok/s（80ms/token）基本一致

### 与第 1 批（torch1_cudagraph_20260813_1）对比

| 指标 | 第 1 批 | 第 2 批（本次） |
|------|---------|----------------|
| gloo:recv 占比 | ~50%（38.8ms/步） | ~43ms/步，仍为最大单点 |
| ncclDev TP 通信 | ~12ms/步 | ~6.3ms/步 |
| scaled_mm | ~9.5ms/步 | ~4.9ms/步 |
| SendRecv 32.6s bug | 0（已修复） | 0（未复现）✓ |

两批 trace 结论一致：**Gloo PP 通信仍是每步最大开销（约一半时间）**，RCCL 修复 P2P graph bug 后切回 NCCL，理论上可再提速接近一倍（12.5 → 20+ tok/s）。

## 复现方法

1. 打开 `deploy/run_pp2_80l.sh` 中注释的 `PROFILER_DIR`/`PROFILER_ARGS`（指向新目录），重启服务（权重加载 ~20 分钟）
2. 服务就绪后：warmup 请求触发 torch.compile（缓存命中则快）→ `POST /start_profile` → 推理请求 → `POST /stop_profile`
3. 收集 `profiles/<dir>/` 下 8 个 rank 的 `.pt.trace.json.gz`（可用 chrome://tracing 或 Perfetto 打开）
4. **注意**：DCU 上 stop_profile 后 engine 主循环可能死亡，收集后须重启正常服务；请求 model 名必须用完整路径
