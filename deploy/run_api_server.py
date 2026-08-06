#!/usr/bin/env python3
"""Thin wrapper: monkey-patch gfx928 → run vLLM API server.

与 verify.py 模式一致 — 运行时修改 vLLM 内部状态，不修改源码。
用法: python3 run_api_server.py --model ... --port 8000 ...
      等价于: python3 -m vllm.entrypoints.openai.api_server ...
"""
import sys
import runpy

# ── Monkey-patch BEFORE vLLM fully initializes ──
# patch_gfx928 在 import 时会将 _ON_GFX9 设为 True
import patch_gfx928  # noqa: F401, E402 — side-effect import must run first

# ── Delegate to vLLM entrypoint ──
# runpy.run_module 与 python3 -m 行为等价，且会正确设置 sys.argv
if __name__ == "__main__":
    # 将当前脚本的 argv 传递给目标模块（去掉脚本名，替换为模块名）
    del sys.argv[0]  # 移除 run_api_server.py 自身
    sys.argv.insert(0, "vllm.entrypoints.openai.api_server")
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
