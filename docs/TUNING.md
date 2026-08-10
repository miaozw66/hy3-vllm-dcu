# HY3 vLLM 算子调优文档 — Hygon DCU (gfx928)

## 背景

HY3 模型（80 layers, 192 experts top-8, INT8 W8A8）在 PP=2/TP=4 双节点 Hygon DCU 上部署，初始 decode 吞吐仅约 1.7 tok/s，远低于硬件理论上限。

### 根因

`vllm/platforms/rocm.py:149` 中 `_ON_GFX9` 的定义不包括 `gfx928`，导致 `on_gfx9()` 返回 `False`，**所有 AITER CK 加速 kernel 被静默禁用**：

- INT8 GEMM → 走通用 Triton kernel 而非硬件调优的 Composable Kernel `gemm_a8w8_CK`
- MoE dispatch → 无专用 fused kernel
- Attention → 使用 `TRITON_ATTN` 而非 `ROCM_AITER_FA`（CK Flash Attention）
- RMSNorm / RoPE → 无 CK 加速

### 瓶颈排序

1. **INT8 GEMM 走 Triton 而非 CK**（最高优先级）
2. **AITER 全家桶被禁用**（根因同 1）
3. **无 KONGMING MoE tuning config**
4. **NFS 权重加载 ~21 min**（单线程，无 fastsafetensors）
5. **`--enforce-eager` 禁用 HIP Graph**（每个 decode step 重新 launch kernel）
6. **PP 跨节点通信**（1 GbE + 双 tensor 传输 + 阻塞式 CPU metadata 交换）

---

## 修改清单

### 1. Monkey-patch：`patch_gfx928.py`

**文件**：`/data/mzw/vllm-hy3/patch_gfx928.py`

**原因**：不修改 vLLM 源码，运行时将 `_ON_GFX9` 设为 `True`，启用全部 AITER CK kernel。

**原理**：与 `verify.py` 同模式 — `import` 时修改 vLLM 模块级常量。升级 vLLM 后无需重新 patch 源码。

```python
import vllm.platforms.rocm as _rocm_platform
_rocm_platform._ON_GFX9 = True
```

### 2. Wrapper：`run_api_server.py`

**文件**：`/data/mzw/vllm-hy3/run_api_server.py`

**原因**：在 vLLM 启动前自动执行 monkey-patch，参数传递与 `python3 -m vllm.entrypoints.openai.api_server` 完全等价。

```
用法: python3 run_api_server.py --model ... --port 8000 ...
     等价于: python3 -m vllm.entrypoints.openai.api_server ...
```

### 3. AITER 环境变量

在启动脚本中设置：

```bash
export VLLM_ROCM_USE_AITER=1           # 总开关
export VLLM_ROCM_USE_AITER_LINEAR=1    # INT8 GEMM → CK gemm_a8w8
export VLLM_ROCM_USE_AITER_MOE=1       # MoE → CK fused_moe
export VLLM_ROCM_USE_AITER_RMSNORM=1   # RMSNorm → CK
export VLLM_ROCM_USE_AITER_MHA=1       # Attention → ROCM_AITER_FA
```

**验证 AITER 是否启用的命令**：

```bash
python3 -c "
import sys; sys.path.insert(0,'/data/mzw/vllm-hy3')
import patch_gfx928
import os; os.environ['VLLM_ROCM_USE_AITER']='1'
from vllm._aiter_ops import rocm_aiter_ops
print('is_enabled:', rocm_aiter_ops.is_enabled())
print('  linear:', rocm_aiter_ops.is_linear_enabled())
print('  fused_moe:', rocm_aiter_ops.is_fused_moe_enabled())
print('  rmsnorm:', rocm_aiter_ops.is_rmsnorm_enabled())
print('  mha:', rocm_aiter_ops.is_mha_enabled())
"
# 预期：全部 True
```

### 4. RCCL 通信调优

```bash
export RCCL_BUFFSIZE=8388608       # 8 MiB buffer（默认 4 MiB）
export NCCL_MIN_NCHANNELS=4        # 最少通道数
export NCCL_PROTO=Simple           # 小消息用 Simple 协议（降低延迟）
export NCCL_ALGO=Ring              # 2 节点拓扑用 Ring
```

### 5. KONGMING MoE Tuning Config

**文件**：`/data/mzw/vllm-hy3/moe_configs/E=192,N=384,device_name=KONGMING.json`

**环境变量**：`VLLM_TUNED_CONFIG_FOLDER=/data/mzw/vllm-hy3/moe_configs/`

包含 18 个 batch size（1~4096）的 kernel 启动参数（BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu），基于 MI300X 参考配置调整。

**验证**：启动日志中不应出现 "Using default MoE config" 警告。

### 6. 多线程权重加载

```bash
--model-loader-extra-config '{"enable_multithread_load": true}'
```

原因：本地 SSD 仅 105 GB，模型 280 GB，无法预拷贝。多线程加载作为替代方案，利用 NFS 并发读取缓解单线程瓶颈。

---

## 受影响的文件

| 文件 | 改了什么 |
|------|----------|
| `patch_gfx928.py` | **新增** — monkey-patch 模块 |
| `run_api_server.py` | **新增** — 入口 wrapper |
| `run_pp2_80l.sh` | AITER/RCCL env vars, MoE config 路径, 多线程加载, 改用 wrapper |
| `run_pp2_80l_niah.sh` | 同上 |
| `run_debug_pp2.sh` | 同上 |
| `moe_configs/E=192,N=384,device_name=KONGMING.json` | **新增** — MoE tuning config |

---

## 实验验证方案

### Phase 1：基准对比

```bash
# 1A. 无 AITER 基准（手动禁用验证）
VLLM_ROCM_USE_AITER=0 bash run_pp2_80l.sh
# 服务器就绪后：
python3 niah_test.py --endpoint http://localhost:8000 --lengths 4096 --timeout 600

# 1B. AITER 启用（当前默认）
bash run_pp2_80l.sh
python3 niah_test.py --endpoint http://localhost:8000 --lengths 4096 --timeout 600
```

**对比指标**：
- TTFT（Time To First Token）
- TPOT（Time Per Output Token）
- Decode tok/s
- 权重加载时间
- 峰值 GPU 显存

### Phase 2：正确性验证

```bash
bash run_pp2_80l_niah.sh  # max-model-len=262144
# 服务器就绪后：
python3 niah_test.py --endpoint http://localhost:8000 \
  --lengths 4096,8192,16384,32768,65536,131072,262144 \
  --positions "0.5" --timeout 1200
```

确认 AITER CK kernel 不引入数值精度问题。

### Phase 3：HIP Graph（待实施）

从启动命令中移除 `--enforce-eager`，观察 graph capture 是否成功。

```bash
# 临时测试：手动编辑启动脚本注释掉 --enforce-eager
# 或直接运行：
python3 -u /data/mzw/vllm-hy3/run_api_server.py \
  --model <MODEL_PATH> \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --nnodes 2 --node-rank 0 \
  ...  # 不含 --enforce-eager
```

若 INT8 动态量化导致 graph capture 失败，尝试 `VLLM_GRAPH_CAPTURE_MODE=split`。

### Phase 4：高级 Profiling（按需）

```bash
# DTK hipprof
hipprof --stats python3 run_api_server.py ...

# 或 torch.profiler
export VLLM_TORCH_PROFILER_DIR=/data/mzw/vllm-hy3/traces/
python3 run_api_server.py ...
```

---

## 预期提升

| 优化项 | 预期提升 | 风险 |
|--------|----------|------|
| AITER CK INT8 GEMM | 3-5x GEMM 加速 | CK kernel 在 gfx928 需 JIT 编译，可能崩溃 |
| AITER CK MoE | 2-3x MoE 加速 | gfx928 无预编译 .co，运行时 JIT |
| RCCL 通信调优 | 5-15% decode | 无 |
| HIP Graph | 30-50% decode | 动态量化可能导致 capture 失败 |
| 多线程加载 | 20-30% 加载 | 无 |

综合预期：decode 吞吐从 ~1.7 tok/s 提升至 5-10 tok/s。

---

## 回滚

如需完全回滚到原始状态：

```bash
# 1. 还原启动脚本（git checkout 或手动还原）
# 2. 删除新增文件
rm /data/mzw/vllm-hy3/patch_gfx928.py
rm /data/mzw/vllm-hy3/run_api_server.py
rm -r /data/mzw/vllm-hy3/moe_configs/
# 3. run_api_server.py 已替代 -m vllm.entrypoints 调用，
#    如需要，在启动脚本中手动改回
```
