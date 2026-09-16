#!/bin/bash
# Runs inside zth-vllm-hy3-test: /workspace/zth_w8a8
set -x
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/usr/local/lib/:/usr/local/lib64/:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
source /opt/dtk/env.sh 2>/dev/null || true

echo "=== kernel selection probe ==="
python3 - <<'PY' 2>&1
from vllm.model_executor.kernels.linear import init_int8_linear_kernel
k = init_int8_linear_kernel(True, False, True, "probe")
print("SELECTED:", type(k).__name__)
PY

echo "=== building 8 extensions (sequential) ==="
cd /workspace/zth_w8a8
declare -A HIPS=(
  [o_proj_m16]=src/M16/o_proj.hip
  [qkv_proj_m16]=src/M16/qkv_proj.hip
  [shared_down_proj_m16]=src/M16/shared_down_proj.hip
  [shared_gate_up_proj_m16]=src/M16/shared_gate_up_proj.hip
  [o_proj_m4096]=src/M4096/o_proj.hip
  [qkv_proj_m4096]=src/M4096/qkv_proj.hip
  [shared_down_proj_m4096]=src/M4096/shared_down_proj.hip
  [shared_gate_up_proj_m4096]=src/M4096/shared_gate_up_proj.hip
)
for name in o_proj_m16 qkv_proj_m16 shared_down_proj_m16 shared_gate_up_proj_m16 o_proj_m4096 qkv_proj_m4096 shared_down_proj_m4096 shared_gate_up_proj_m4096; do
  echo "[$(date +%T)] building $name ..."
  if timeout 1800 python3 build_extension.py "$name" "${HIPS[$name]}" > logs/$name.log 2>&1; then
    echo "[$(date +%T)] $name rc=0"
  else
    echo "[$(date +%T)] $name rc=$? BUILD_FAILED"
    exit 1
  fi
done

echo "=== import check ==="
python3 - <<'PY' 2>&1
import torch
names = ["o_proj_m16","qkv_proj_m16","shared_down_proj_m16","shared_gate_up_proj_m16",
         "o_proj_m4096","qkv_proj_m4096","shared_down_proj_m4096","shared_gate_up_proj_m4096"]
for n in names:
    ns = getattr(torch.ops, f"zth_{n}")
    print("OK", n, hasattr(ns, "gemm_out"), hasattr(ns, "pack_weight"))
print("ALL_EXTENSIONS_OK")
PY
echo "BUILD_ALL_DONE"
