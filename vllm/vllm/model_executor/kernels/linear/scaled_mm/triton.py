# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import os
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    get_triton_scaled_mm_launch_config,
    triton_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.platforms import current_platform

from .cutlass import CutlassInt8ScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    Int8ScaledMMLinearLayerConfig,
)


_ABI_DUMP_SEEN: set[tuple] = set()


def _tensor_metadata(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": tensor.is_contiguous(),
    }


def _dump_scaled_mm_abi(
    layer: torch.nn.Module,
    x: torch.Tensor,
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    x_s: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    target = os.getenv("VLLM_HY3_SCALED_MM_ABI_DUMP")
    if not target or torch.compiler.is_compiling():
        return

    capturing = False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is not None:
        capturing = bool(is_capturing())
    prefix = getattr(layer, "prefix", layer.__class__.__name__)
    key = (
        prefix,
        tuple(x_q.shape),
        tuple(x_q.stride()),
        tuple(w_q.shape),
        tuple(w_q.stride()),
        tuple(x_s.shape),
        tuple(w_s.shape),
        capturing,
    )
    if key in _ABI_DUMP_SEEN:
        return
    _ABI_DUMP_SEEN.add(key)

    m, k = x_q.shape
    n = w_q.shape[1]
    tile, num_warps, num_stages = get_triton_scaled_mm_launch_config(
        m, k, n, x_q.dtype
    )
    payload = {
        "layer_prefix": prefix,
        "layer_class": layer.__class__.__name__,
        "m": m,
        "k": k,
        "n": n,
        "capturing": capturing,
        "input": _tensor_metadata(x),
        "input_q": _tensor_metadata(x_q),
        "weight_q": _tensor_metadata(w_q),
        "input_scale": _tensor_metadata(x_s),
        "weight_scale": _tensor_metadata(w_s),
        "bias": _tensor_metadata(bias),
        "output_dtype": str(x.dtype),
        "tile": list(tile),
        "num_warps": num_warps,
        "num_stages": num_stages,
        "grid_programs": ((m + tile[0] - 1) // tile[0])
        * ((n + tile[1] - 1) // tile[1]),
        "pid": os.getpid(),
        "rank": os.getenv("RANK"),
        "local_rank": os.getenv("LOCAL_RANK"),
    }
    output_path = Path(
        target.format(
            pid=os.getpid(),
            rank=os.getenv("RANK", "unknown"),
            local_rank=os.getenv("LOCAL_RANK", "unknown"),
        )
    )
    with output_path.open("a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


class TritonInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.is_cuda_alike():
            return True, None
        return False, "requires ROCm or CUDA."

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "supports symmetric input only."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q, _, i_s, _, _ = self._get_layer_params(layer)
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names

        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(w_q.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Triton kernel supports only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not self.config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if self.config.is_static_input_scheme:
            assert i_s is not None
            replace_parameter(
                layer,
                i_s_name,
                torch.nn.Parameter(i_s.max(), requires_grad=False),
            )
            setattr(layer, i_zp_name, None)
        else:
            setattr(layer, i_s_name, None)
            setattr(layer, i_zp_name, None)

        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, _ = self._get_layer_params(layer)

        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=True
        )

        assert x_zp is None, "Triton kernel only supports symmetric quantization"

        _dump_scaled_mm_abi(layer, x, x_q, w_q, x_s, w_s, bias)

        return triton_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )
