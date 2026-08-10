#!/usr/bin/env python3
"""
gfx928 (Hygon K100) monkey-patch for vLLM AITER CK acceleration.

Import this BEFORE any vllm imports to enable Composable Kernel (CK)
INT8 GEMM, fused MoE, CK RMSNorm, and CK Flash Attention on the K100.

Background:
  vllm/platforms/rocm.py:149 defines _ON_GFX9 as matching only
  ["gfx90a", "gfx942", "gfx950"].  Because "gfx928" is absent,
  on_gfx9() returns False, silently disabling ALL AITER CK kernels.

  This patch forces _ON_GFX9 to True, re-enabling the full AITER
  acceleration stack.

Usage:
  import patch_gfx928  # noqa: F401  — must be FIRST import
  # ... rest of vllm imports follow
"""

import vllm.platforms.rocm as _rocm_platform

_rocm_platform._ON_GFX9 = True
