"""Monkey-patch: add gfx928 to _ON_GFX9 before vLLM initializes.

与 verify.py 模式一致：不修改 vLLM 源码，而是在运行时修改模块级常量。
在 vLLM import 之后、模型初始化之前执行即可确保 AITER CK kernel 被启用。

用法：
    在启动命令前执行:
        python3 -c "import patch_gfx928; ..."
    或在 wrapper 脚本中:
        import patch_gfx928
        # ... 正常启动 vLLM
"""
import vllm.platforms.rocm as _rocm_platform

_rocm_platform._ON_GFX9 = True
