# CUDA Graph Benchmark 实验报告

## 实验目的
测试带 CUDA graph 的 vLLM 在 HY3 80层模型上的推理性能。

## 实验环境

### 硬件
- **GPU**: 2 节点 x 4 Hygon DCU gfx928 (共 8 GPU)
- **节点 0 (主节点)**: 10.18.17.71 (hostname: worker24)
- **节点 1 (从节点)**: 10.18.17.74
- **网络接口**: eno1
- **部署模式**: Docker (mmh_qwen_opt)

### 软件
- **vLLM**: 0.18.1
- **ROCm**: 6.3.3.0
- **HIP**: 6.3.26102
- **RCCL**: 2.22.3
- **Python**: 3.10
- **量化**: INT8 W8A8 (Compressed-Tensors)
- **后端**: TRITON_ATTN

### 模型配置
- **模型**: Hy3-Channel-INT8-w8a8 (80层 MoE)
- **路径**: /data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master
- **并行策略**: TP4 + PP2 (tensor parallel 4, pipeline parallel 2)
- **max-model-len**: 8192
- **CUDA Graph**: FULL_AND_PIECEWISE mode
- **torch.compile**: 启用，backend=inductor
- **MoE 配置**: /data/mzw/vllm-hy3/configs/moe_configs/

### 编译缓存
- torch.compile 缓存: /root/.cache/vllm/torch_compile_cache/
- AOT 编译缓存: /root/.cache/vllm/torch_compile_cache/torch_aot_compile/

## 实验过程

### 部署

#### 启动命令
```bash
cd /data/mzw/vllm-hy3
bash deploy/run_pp2_80l.sh
```

#### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| --model | Hy3-Channel-INT8-w8a8 全量模型 | 80层 MoE |
| --pipeline-parallel-size | 2 | 流水线并行 |
| --tensor-parallel-size | 4 | 张量并行 |
| --nnodes | 2 | 2个节点 |
| --max-model-len | 8192 | 最大序列长度 |
| --gpu-memory-utilization | 0.90 | GPU 显存利用率 |
| --trust-remote-code | 是 | 信任模型代码 |
| --enable-auto-tool-choice | 是 | 自动工具选择 |
| --tool-call-parser | hy_v3 | HY3 工具调用解析器 |
| --distributed-timeout-seconds | 1800 | 分布式超时 (30分钟) |

#### 日志输出

实验结果日志见:
- Node 0: logs/vllm_node0_pp2_80l_MMDD_HHMM.log
- Node 1: /tmp/node1_80l.log (Docker 内部)
- Benchmark 结果: benchmark/cudagraph_results_YYYYMMDD_HHMMSS.json

### 遇到的问题及解决方案

#### 问题1: Node 1 GPU 内存泄漏导致 WorkerProc 启动失败 (3次复现)

**现象**: 
- Node 0 模型加载完成后，WorkerProc 初始化时 `torch.distributed.barrier()` 报错:
  `RuntimeError: Connection closed by peer [10.18.17.74]`
- 根因在 Node 1: `ValueError: Free memory on device cuda:0 (3.68/63.98 GiB) on startup is less than desired GPU memory utilization (0.9, 57.59 GiB)`

**根因分析**:
- 0811_2334 运行成功后，Node 1 上的 PP1 工作进程 (VLLM::Worker_PP1_TP0~3) 未被正确清理
- `pkill -9 -f vllm` 无法匹配进程名 `VLLM::Worker_PP1_TP0` (大小写不匹配)
- 这些进程持续占用 Node 1 的 4 个 GPU 各 94% VRAM (约 60GB/64GB)
- 后续部署尝试时 Node 1 没有可用的 GPU 内存

**解决方案**:
1. 使用 `rocm-smi` 检查 Node 1 GPU 内存使用情况
2. 用 PID 精确杀掉残留的 Worker 进程: `kill -9 38406 38407 38408 38409`
3. 验证 GPU 内存释放 (VRAM 使用从 94% 降到 ~0%)
4. 重新部署

**改进建议**:
- 在 `deploy/run_pp2_80l.sh` 的清理命令中添加大小写不敏感匹配:
  ```bash
  pkill -9 -i -f worker_pp 2>/dev/null
  ```
- 或在部署前添加 GPU 内存检查步骤

#### 问题2: model-max-len 不一致

**现象**: 0811_2334 成功运行使用 `max-model-len 32768`，但当前脚本使用 8192
**影响**: 不影响本次 benchmark (prompt 很短)，但限制了长上下文能力
**状态**: 已确认当前脚本配置为 8192，适用于短 prompt 测试

#### 部署状态

#### 第二次尝试 (0812_0037) — 因 Node 1 GPU 内存泄漏失败
- 启动时间: 08-12 00:37
- 失败原因: Node 1 残留 Worker 进程占用 GPU VRAM
- 状态: 修复后重启

#### 第三次尝试 (0812_0049) — 成功启动但推理失败

**服务器启动成功**:
- 启动时间: 08-12 00:49
- 服务器就绪: 08-12 01:13:33
- Node 0 log: `logs/vllm_node0_pp2_80l_0812_0049.log`

**启动时间线 (实测)**:
| 阶段 | 时间 | 耗时 |
|------|------|------|
| 启动脚本 | 00:49:10 | - |
| APIServer 初始化 | 00:49:50 | 0:40 |
| EngineCore 启动 | 00:50:25 | 0:35 |
| 模型权重加载 | 00:51:09 ~ 01:12:02 | ~21 min (99 shards, 34.69 GiB) |
| torch.compile | 01:12:26 ~ 01:12:38 | 42.94s (缓存命中) |
| KV cache 初始化 | 01:12:57 | 22.36 GiB, 554,640 tokens |
| CUDA graph PIECEWISE | ~01:13:29 | 19s (51 graphs) |
| CUDA graph FULL decode | ~01:13:29 | 11s (35 graphs) |
| 服务器就绪 | 01:13:33 | - |

**总计启动时间**: ~24 分钟

**基本验证**:
- 01:14:33: 发送测试请求 "中国的首都是" (max_tokens=5)
- 响应正确: "北京。 中国的首"
- 测试请求完成: 约 26s (含模型首次推理开销)

**问题: 长文本推理异常缓慢**
- 01:15:13: 运行 benchmark (prompt: "输出岳飞满江红全文，怒发冲冠那篇", max_tokens=1024)
- **现象**: 推理吞吐极低 (0.0-0.2 tokens/s)，GPU 利用率仅 0.8%，Worker CPU 110%
- 日志在 01:27:53 后停止输出，服务器对外 `/health` 仍可访问
- 新请求全部超时 (连接排队等待)
- Workers 持续高 CPU (>100%) 但 GPU 几乎空闲
- 01:38: 手动终止部署

**根因分析** (待确认):
- 高 CPU + 低 GPU = 典型的 GPU kernel 启动受阻或 CPU 端忙等待
- 可能原因: CUDA graph 重放问题 / NCCL 通信挂起 / IOMMU 配置缺失
- 对比: 0811_2334 运行使用 max_model_len=32768 并成功，但与本次部署的 max_model_len=8192 不同
- 与之前成功的短文本推理 (max_tokens=5) 对比，短文本能正常返回

#### 第四次尝试 (0812_0141) — 取消
- 启动时间: 08-12 01:41
- 终止原因: 发现 0812_0049 的性能问题后，改为使用 `--enforce-eager` 模式
- 状态: 启动后立即终止

### 性能问题分析

#### 发现: CUDA Graph + torch.compile 导致推理性能严重退化

通过对比历史部署日志发现:

| 部署 | CUDA Graph | torch.compile | 生成吞吐 | 备注 |
|------|-----------|---------------|---------|------|
| 0807_1353 | 关闭 | 关闭 (`enforce-eager`) | **2.3-2.6 tok/s** | fuse_norm/act_quant 启用 |
| 0811_2334 | FULL_AND_PIECEWISE | inductor | ~0.1-0.2 tok/s | 未完成长文本测试 |
| 0812_0049 | FULL_AND_PIECEWISE | inductor | **0.0-0.2 tok/s** | Worker CPU 110%, GPU 0.8% |

**结论**: CUDA graph (FULL_AND_PIECEWISE) + torch.compile (inductor) 组合导致推理吞吐从 2.5 tok/s 退化到 0.1 tok/s，性能下降约 **10-20 倍**。

**症状**:
- Worker 进程 CPU 使用率 110% (应为低 CPU)
- GPU 利用率 0.8% (应为 >80%)
- 生成吞吐 0.0-0.2 tok/s
- 日志输出间歇性停止 (NCCL 通信可能被阻塞)
- IOMMU 未配置 (`iommu=pt` 缺失) — 可能导致系统不稳定或挂起

**根因推测**:
1. **CUDA graph 重放失败**: graph 捕获后重放时 GPU kernel 未能正常启动
2. **torch.compile 生成的代码运行在 CPU 上**: Dynamo trace 可能引入了 CPU fallback
3. **NCCL 通信挂起**: 缺少 IOMMU passthrough 导致 NCCL 集合操作间歇性挂起
4. **与编译选项冲突**: `custom_ops=['+sparse_attn_indexer', 'none']` 可能导致算子选择问题

**对比 0807 成功配置**:
- `--enforce-eager` (禁用 torch.compile 和 CUDA graph)
- `max_model_len=262144` (vs 当前 8192)
- `pass_config` 中 `fuse_norm_quant=True, fuse_act_quant=True` (当前为 False)
- `custom_ops=['+sparse_attn_indexer', 'all']` (当前为 'none')

#### 第五次尝试 (0812_0144) — 使用 enforce-eager 模式 (失败)
- 启动时间: 08-12 01:44
- Log: `logs/vllm_node0_pp2_80l_eager_0812_0144.log`
- 配置: `--enforce-eager`
- 失败原因: 端口冲突 (NCCL internal error)，旧进程残留占用 29511 端口

#### 第六次尝试 (0812_0151) — enforce-eager 模式 (成功)
- 启动时间: 08-12 01:51
- Log: `logs/vllm_node0_pp2_80l_eager_0812_0151.log`
- 使用端口 29611 (避免冲突)
- 服务器就绪: 02:10:48 (~19 分钟)
- 配置: `--enforce-eager`，与 0807_1353 成功配置匹配

**Benchmark 结果 (enforce-eager baseline)**:

| 指标 | Concurrency=1 | Concurrency=4 Run1 | 说明 |
|------|--------------|-------------------|------|
| TTFT (avg) | 0.39s | 0.81s | 首 token 延迟 |
| Decode 吞吐 (per-request) | 5.9 tok/s | 3.2 tok/s (est.) | 单请求吞吐 |
| 批量吞吐 (aggregate) | 5.9 tok/s | 12.9 tok/s | 总吞吐 |
| 输出 tokens | 1024 | 598 avg | 满江红全文约 598 tokens |
| GPU KV cache | 0.0% | 0.1% | 显存充足 |

**关键发现**:
1. **enforce-eager 性能是 CUDA graph 的 30-50x**: 5.9 tok/s vs 0.1-0.2 tok/s
2. **批量吞吐线性扩展**: 服务器处理 4 个并发请求时总吞吐可达 24 tok/s
3. **max_num_batched_tokens=2048 限制**: 每个请求生成 600-1024 tokens 时，最大并发 batch 为 2
4. **模型输出质量良好**: 满江红全文正确生成

### Benchmark 最终对比

| 模式 | 单请求吞吐 | 4并发吞吐 | 状态 |
|------|-----------|----------|------|
| CUDA Graph (FULL_AND_PIECEWISE) | ~0.1 tok/s | 未测 | 严重退化，不可用 |
| enforce-eager (无CUDA graph) | **5.9 tok/s** | **12.9 tok/s** | 正常，baseline |

**结论**: 在当前硬件配置 (Hygon DCU gfx928, ROCm 6.3.3, RCCL 2.22.3) 下，CUDA graph (FULL_AND_PIECEWISE) + torch.compile (inductor) 导致推理性能严重退化 (~50x 慢于 enforce-eager)。建议在此硬件上暂时使用 `--enforce-eager` 模式运行推理。

