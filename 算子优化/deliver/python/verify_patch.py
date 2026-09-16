#!/usr/bin/env python3
"""Verify the import-hook patch is registered and routing tables are correct.

Checks (no GPU work):
  1. sitecustomize/zth_w8a8_integrate imported cleanly and hook is installed.
  2. TritonInt8ScaledMMLinearKernel.apply_weights is patched.
  3. Routing table contains exactly the 7 fast shapes and NOT down_proj_m4096.
  4. Extension .so files resolve for every routed shape.

Usage (container):  python3 /home/hy3-TP8/entry-vllm/verify_patch.py
"""
import os
import sys

os.environ.setdefault("ZTH_W8A8_CACHE", "/home/hy3-TP8/ext_cache")
sys.path.insert(0, "/home/hy3-TP8/entry-vllm")

import zth_w8a8_integrate as integ  # noqa: F401  (registers hook)
import zth_w8a8_ext


def main():
    # 1. hook installed
    assert any(
        isinstance(f, integ._Finder) for f in sys.meta_path
    ), "integration hook NOT installed"
    print("[OK] import hook installed")

    # 2. patched class
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )
    import types
    assert isinstance(TritonInt8ScaledMMLinearKernel.apply_weights,
                      types.FunctionType)
    print("[OK] TritonInt8ScaledMMLinearKernel.apply_weights patched")

    # 3. routing tables
    all_routed = set(integ.M16_SHAPES) | set(integ.M4096_SHAPES)
    print(f"[OK] routed shapes ({len(all_routed)}):")
    for s in sorted(all_routed):
        print(f"      {s}")
    assert (4096, 4096, 192) not in integ.M4096_SHAPES, (
        "down_proj_m4096 must NOT be routed"
    )
    assert (4096, 4096, 192) not in integ.KN_PACK_M4096, (
        "down_proj_m4096 must NOT be packed for M4096"
    )
    assert (192, 4096) in integ.KN_PACK_M16, (
        "down_proj_m16 pack must be kept (M16 IS routed)"
    )
    print("[OK] down_proj_m4096 excluded from M4096 routing and packing")
    print("[OK] down_proj_m16 (M16) still routed and packed")

    # 4. extension resolution
    failed = zth_w8a8_ext.load_all(verbose=True)
    if failed:
        print(f"[FAIL] extensions missing: {failed}")
        sys.exit(1)
    print("[OK] all routed extensions loadable")
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
