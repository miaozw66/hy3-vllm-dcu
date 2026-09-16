#!/usr/bin/env python3
"""Locate and load the 7 routed zth_<shape> extensions (.so).

Search order:
  1. ZTH_W8A8_CACHE env (default: /home/hy3-TP8/ext_cache == host
     /public/home/zz_zhoutianhao/hy3-TP8/ext_cache, survives container rebuild)
  2. fallback /root/.cache/torch_extensions/py310_cpu (inside-container build cache)

Safe to call multiple times; no CUDA work at import.
"""
import glob
import os

import torch

_CACHE = os.environ.get(
    "ZTH_W8A8_CACHE",
    "/home/hy3-TP8/ext_cache",
)
_FALLBACK = "/root/.cache/torch_extensions/py310_cpu"

# The 7 routed shapes + shared_down_proj_m4096. The down m>=4096 shape
# (4096,4096,192) measured 2.38x SLOWER than Triton and is intentionally NOT
# routed (KN_PACK_M4096 excludes (192,4096)), but its extension must be
# loaded because the m <= 32 small-M path routes (192, 4096) through its
# launch_w8a8_gemm split-k arm and pack_weight (see zth_w8a8_integrate.py
# KN_PACK_M32). The m>=128 arm is never reached for down in production.
NAMES = [
    "o_proj_m16",
    "qkv_proj_m16",
    "shared_down_proj_m16",
    "shared_gate_up_proj_m16",
    "o_proj_m4096",
    "qkv_proj_m4096",
    "shared_gate_up_proj_m4096",
    "shared_down_proj_m4096",
]
NAMES_STANDALONE = NAMES

_loaded = False
_failed = []


def _find_so(name):
    for base in (_CACHE, _FALLBACK):
        sos = sorted(glob.glob(os.path.join(base, f"zth_{name}", "*.so")))
        if sos:
            return sos[0]
    return None


def _register_abstract():
    """Register fake-tensor (meta) impls so torch.compile / dynamo can trace
    the custom ops without running the real kernels."""
    try:
        from torch.library import register_fake
    except ImportError:  # older torch
        from torch.library import impl_abstract as register_fake

    for name in NAMES:
        try:
            @register_fake(f"zth_{name}::gemm_out")
            def _gemm_out(x_q, packed, x_scale, w_scale, workspace):
                m = x_q.size(0)
                n = w_scale.size(0)
                return torch.empty((m, n), dtype=torch.bfloat16,
                                   device=x_q.device)

            @register_fake(f"zth_{name}::pack_weight")
            def _pack_weight(raw, w_scale):
                return torch.empty_like(raw), torch.empty_like(w_scale)
        except Exception as exc:  # noqa: BLE001
            print(f"abstract impl failed {name}: {exc}")


def load_all(verbose=False, include_standalone_only=False):
    """Load routed extensions. include_standalone_only=True additionally loads
    shared_down_proj_m4096 (for standalone correctness/latency tests only)."""
    global _loaded
    if _loaded:
        return _failed
    names = NAMES_STANDALONE if include_standalone_only else NAMES
    for name in names:
        so = _find_so(name)
        if so is None:
            _failed.append(name)
            continue
        try:
            torch.ops.load_library(so)
            if verbose:
                print("loaded", name, so)
        except Exception as exc:  # noqa: BLE001
            _failed.append(name)
            print(f"load failed {name}: {exc}")
    _register_abstract()
    _loaded = True
    return _failed
