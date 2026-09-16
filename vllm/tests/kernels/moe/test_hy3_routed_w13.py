# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe import fused_moe


class _Weight:
    dtype = torch.int8
    shape = (192, 384, 4096)

    def data_ptr(self) -> int:
        return 17


def _eligible_kwargs(hidden_states: torch.Tensor) -> dict:
    return {
        "hidden_states": hidden_states,
        "w1": _Weight(),
        "topk_ids": torch.zeros((4, 8), dtype=torch.int32),
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": True,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_channel_quant": True,
        "expert_map": None,
        "w1_zp": None,
        "block_shape": None,
        "w1_bias": None,
    }


def test_routed_w13_hip_predicate_is_fail_closed(monkeypatch):
    hidden_states = torch.empty((4, 4096), dtype=torch.bfloat16)
    kwargs = _eligible_kwargs(hidden_states)
    monkeypatch.setenv("ZTH_ROUTED_W13_MODE", "use")
    monkeypatch.setattr(fused_moe, "_routed_w13_packed", {17: torch.empty(0)})
    monkeypatch.setattr(
        fused_moe,
        "_routed_w13_token_ids",
        {(hidden_states.device, 4): torch.empty(32, dtype=torch.int32)},
    )

    assert fused_moe._can_use_routed_w13_hip(**kwargs)

    kwargs["topk_ids"] = torch.zeros((4, 8), dtype=torch.int64)
    assert not fused_moe._can_use_routed_w13_hip(**kwargs)


def test_routed_w13_hip_predicate_rejects_unqualified_m(monkeypatch):
    hidden_states = torch.empty((3, 4096), dtype=torch.bfloat16)
    kwargs = _eligible_kwargs(hidden_states)
    kwargs["topk_ids"] = torch.zeros((3, 8), dtype=torch.int32)
    monkeypatch.setenv("ZTH_ROUTED_W13_MODE", "use")
    monkeypatch.setattr(fused_moe, "_routed_w13_packed", {17: torch.empty(0)})
    monkeypatch.setattr(fused_moe, "_routed_w13_token_ids", {})

    assert not fused_moe._can_use_routed_w13_hip(**kwargs)
