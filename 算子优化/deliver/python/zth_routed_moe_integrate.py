#!/usr/bin/env python3
"""Optional HY3 TP8 routed-MoE -> sidecar HIP integration.

Importing registers an import hook only. Set ZTH_ROUTED_MOE_MODE=use to patch
fused_experts_impl after vLLM loads it. The exact decode shape is routed to
routed_w13.so; every other configuration calls the original implementation.
"""
import os
import sys
from importlib.abc import Loader, MetaPathFinder

_MODE = os.environ.get("ZTH_ROUTED_MOE_MODE", "off")
_TARGET = "vllm.model_executor.layers.fused_moe.fused_moe"
_SO = os.path.join(
    os.environ.get("ZTH_W8A8_CACHE", "/home/hy3-vllm-dcu/算子优化/deliver/so_cache"),
    "zth_routed_moe", "routed_w13.so")
_packed_ptrs = set()
_tok_cache = {}
_loaded = False


def _load():
    global _loaded
    if not _loaded:
        import torch
        torch.ops.load_library(_SO)
        _loaded = True


def _pack_inplace(weight):
    ptr = weight.data_ptr()
    if ptr in _packed_ptrs:
        return
    import torch
    packed = torch.ops.routed_w13.w1_pack(weight)
    weight.copy_(packed.view_as(weight))
    _packed_ptrs.add(ptr)


def _unpack_inplace(weight):
    ptr = weight.data_ptr()
    if ptr not in _packed_ptrs:
        return
    import torch
    raw = torch.ops.routed_w13.w1_unpack(
        weight, weight.size(0), weight.size(1), weight.size(2))
    weight.copy_(raw)
    _packed_ptrs.remove(ptr)


def _tok_ids(device, m):
    import torch
    key = (device.type, device.index, m)
    result = _tok_cache.get(key)
    if result is None:
        result = torch.arange(m, device=device, dtype=torch.int32).repeat_interleave(8)
        _tok_cache[key] = result
    return result


def _patch(module):
    if _MODE != "use" or getattr(module, "_zth_routed_moe_patched", False):
        return
    original = module.fused_experts_impl

    def wrapped(
        hidden_states, w1, w2, topk_weights, topk_ids, inplace,
        activation="silu", apply_router_weight_on_input=False,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        use_int4_w4a16=False, ocp_mx_scheme=None, per_channel_quant=False,
        global_num_experts=-1, expert_map=None, w1_scale=None, w2_scale=None,
        w1_zp=None, w2_zp=None, a1_scale=None, a2_scale=None,
        block_shape=None, w1_bias=None, w2_bias=None,
    ):
        import torch
        use_hip = (
            not inplace
            and activation == "silu"
            and not apply_router_weight_on_input
            and use_int8_w8a8
            and not use_fp8_w8a8
            and not use_int8_w8a16
            and not use_int4_w4a16
            and ocp_mx_scheme is None
            and per_channel_quant
            and expert_map is None
            and w1_zp is None and w2_zp is None
            and a1_scale is None and a2_scale is None
            and block_shape is None
            and w1_bias is None and w2_bias is None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.size(0) in (4, 16)
            and hidden_states.size(1) == 4096
            and tuple(w1.shape) == (192, 384, 4096)
            and tuple(w2.shape) == (192, 4096, 192)
            and w1.dtype == torch.int8 and w2.dtype == torch.int8
            and w1_scale is not None and w2_scale is not None
            and tuple(topk_ids.shape) == (hidden_states.size(0), 8)
            and tuple(topk_weights.shape) == tuple(topk_ids.shape)
            and topk_weights.dtype == torch.float32
        )
        if not use_hip:
            if w1.data_ptr() in _packed_ptrs or w2.data_ptr() in _packed_ptrs:
                _load()
                _unpack_inplace(w1)
                _unpack_inplace(w2)
            return original(
                hidden_states, w1, w2, topk_weights, topk_ids, inplace,
                activation, apply_router_weight_on_input, use_fp8_w8a8,
                use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                ocp_mx_scheme, per_channel_quant, global_num_experts,
                expert_map, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                a2_scale, block_shape, w1_bias, w2_bias)

        _load()
        _pack_inplace(w1)
        _pack_inplace(w2)
        from vllm.model_executor.layers.quantization.utils.int8_utils import (
            per_token_quant_int8)
        qhidden, input_scale = per_token_quant_int8(hidden_states)
        m = hidden_states.size(0)
        p = m * 8
        tok = _tok_ids(hidden_states.device, m)
        exp = topk_ids.reshape(-1).to(dtype=torch.int32)
        weights = topk_weights.reshape(-1).contiguous()
        w13_out = torch.empty((p, 384), dtype=torch.bfloat16,
                              device=hidden_states.device)
        qact = torch.empty((p, 192), dtype=torch.int8,
                           device=hidden_states.device)
        act_scale = torch.empty((p,), dtype=torch.float32,
                                device=hidden_states.device)
        pair_out = torch.empty((p, 4096), dtype=torch.bfloat16,
                               device=hidden_states.device)
        out = torch.empty_like(hidden_states)
        torch.ops.routed_w13.full_out(
            qhidden, w1, w2, input_scale.reshape(-1),
            w1_scale.squeeze(-1), w2_scale.squeeze(-1), tok, exp, weights,
            w13_out, qact, act_scale, pair_out, out)
        return out

    module.fused_experts_impl = wrapped
    module._zth_routed_moe_patched = True


class _Loader(Loader):
    def __init__(self, original):
        self._original = original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._original.exec_module(module)
        _patch(module)


class _Finder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


if _MODE == "use" and os.environ.get("EAGER") == "on":
    sys.meta_path.insert(0, _Finder())
elif _MODE == "use":
    print("[zth_routed_moe] disabled: experimental in-place pack requires EAGER=on")
