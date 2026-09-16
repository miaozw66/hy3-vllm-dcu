#!/usr/bin/env python3
"""Route-level test: drive the patched TritonInt8ScaledMMLinearKernel through a
mock vLLM layer in ZTH_W8A8_MODE=validate for every m in {1,2,4,8,32} x the 4
TP8 shapes. Verifies the full chain: import-hook patch -> process_weights
(pack into _zth_packed32/_zth_ws32) -> apply_weights (m<=32 arm of
KN_PACK_M32). validate mode computes the original Triton result, logs the
HIP-vs-original diff, and returns the original result; a clean diff log
means the m<=32 HIP path is numerics-equivalent to production.
Usage: python3 verify_m32_routing.py
"""
import os
import sys

os.environ.setdefault(
    "ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache")
sys.path.insert(0, "/home/hy3-vllm-dcu/算子优化/deliver/python")

import torch

import zth_w8a8_integrate as integ  # noqa: F401  (registers import hook)

# NOTE: python/sitecustomize.py is auto-imported at interpreter startup and may
# already have imported zth_w8a8_integrate with a stale _MODE; force validate.
integ._MODE = "validate"

if os.path.exists(integ._LOG_PATH):
    os.remove(integ._LOG_PATH)

from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (  # noqa: E402
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (  # noqa: E402
    TritonInt8ScaledMMLinearKernel,
)

# (label, n_out, k_in) per TP8 shape; weight is [n, k] (vLLM Linear convention).
SHAPES = [
    ("qkv_proj", 1280, 4096),
    ("o_proj", 4096, 1024),
    ("gate_up", 384, 4096),
    ("down", 4096, 192),
]
M_VALUES = [1, 2, 4, 8, 32]


class MockLayer(torch.nn.Module):
    def __init__(self, n: int, k: int):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randint(-127, 128, (n, k), dtype=torch.int8,
                          device="cuda"),
            requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            (torch.rand(n, 1, device="cuda") + 0.1).float(),
            requires_grad=False)
        self.logical_widths = [n]
        self.input_scale = None
        self.input_zero_point = None
        self.azp_adj = None


def main():
    config = Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False, is_channelwise=True, input_symmetric=True)
    kernel = TritonInt8ScaledMMLinearKernel(
        config, ["weight", "weight_scale", "input_scale", "input_zero_point",
                 "azp_adj"])

    torch.manual_seed(0)
    all_ok = True
    routed = 0
    for label, n, k in SHAPES:
        layer = MockLayer(n, k)
        kernel.process_weights_after_loading(layer)
        route = getattr(layer, "_zth_route", None)
        has_pack = getattr(layer, "_zth_packed32", None) is not None
        print(f"{label:10s} pack32={has_pack} route={route}")
        assert route is not None, f"expected route for {(k,n)} got None"
        for m in M_VALUES:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 3.0
            out = kernel.apply_weights(layer, x, None)  # validate mode: returns ref
            # out must equal the HIP result (else diff log would have gt1 rows)
            d = (out.float() - out.float()).abs().max().item()
            routed += 1
            print(f"  m={m:2d} out_shape={tuple(out.shape)} maxself={d:.3f}")
            if tuple(out.shape) != (m, n):
                all_ok = False

    # check the validate diff log
    gt1 = 0
    rows = 0
    if os.path.exists(integ._LOG_PATH):
        with open(integ._LOG_PATH) as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) == 5 and p[1] not in ("PACK_ERR", "PATCH_ERR"):
                    rows += 1
                    if int(p[4]) > 0:
                        gt1 += 1
    print(f"\ndiff log: rows={rows} gt1={gt1} routed_calls={routed}")
    status = "PASS" if (all_ok and gt1 == 0 and rows == routed) else "FAIL"
    print(f"{status}: patched routing chain for m<=32")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
