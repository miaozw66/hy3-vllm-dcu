#!/usr/bin/env python3
"""Correctness test for the fixed-shape HY3 W8A8 HIP MMAC custom op."""

import argparse
from pathlib import Path

import torch


def reference(
    activation_q: torch.Tensor,
    weight_kn: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    acc = activation_q.cpu().to(torch.int32) @ weight_kn.cpu().to(torch.int32)
    value = acc.float() * activation_scale.cpu().reshape(1, 1)
    value = value * weight_scale.cpu().reshape(1, -1)
    return value.to(torch.bfloat16)


def make_inputs(mode: str, device: str):
    if mode == "ones":
        activation = torch.ones((1, 4096), device=device, dtype=torch.int8)
        weight = torch.ones((4096, 384), device=device, dtype=torch.int8)
    elif mode == "extreme":
        activation = torch.full((1, 4096), 127, device=device, dtype=torch.int8)
        weight = torch.full((4096, 384), -128, device=device, dtype=torch.int8)
    elif mode == "k32-pattern":
        activation = torch.zeros((1, 4096), device=device, dtype=torch.int8)
        weight = torch.zeros((4096, 384), device=device, dtype=torch.int8)
        activation[0, ::32] = 1
        weight[::32, :] = torch.arange(384, device=device, dtype=torch.int16).to(torch.int8)
    else:
        activation = torch.randint(-128, 128, (1, 4096), device=device, dtype=torch.int8)
        weight = torch.randint(-128, 128, (4096, 384), device=device, dtype=torch.int8)
    activation_scale = torch.tensor([[0.03125]], device=device, dtype=torch.float32)
    weight_scale = torch.linspace(0.001, 0.02, 384, device=device, dtype=torch.float32).reshape(-1, 1)
    return activation, weight, activation_scale, weight_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--split-k", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--a-resident", action="store_true")
    parser.add_argument("--persistent-workspace", action="store_true")
    args = parser.parse_args()
    if args.a_resident and args.split_k != 16:
        raise ValueError("--a-resident currently requires --split-k 16")
    if args.persistent_workspace and not args.a_resident:
        raise ValueError("--persistent-workspace currently requires --a-resident")

    torch.cuda.set_device(args.gpu)
    torch.ops.load_library(str(args.extension))
    op = torch.ops._rocm_C
    scaled_mm = op.hy3_w8a8_scaled_mm_splitk16_aresident if args.a_resident else {
        1: op.hy3_w8a8_scaled_mm,
        4: op.hy3_w8a8_scaled_mm_splitk4,
        8: op.hy3_w8a8_scaled_mm_splitk8,
        16: op.hy3_w8a8_scaled_mm_splitk16,
    }[args.split_k]
    modes = ("ones", "extreme", "k32-pattern", "random")
    for mode in modes:
        activation, weight, activation_scale, weight_scale = make_inputs(mode, "cuda")
        packed = op.hy3_w8a8_pack(weight)
        partial = torch.empty((16, 384), device="cuda", dtype=torch.int32)
        tickets = torch.zeros((24,), device="cuda", dtype=torch.int32)
        output = torch.empty((1, 384), device="cuda", dtype=torch.bfloat16)

        def call():
            if args.persistent_workspace:
                op.hy3_w8a8_scaled_mm_splitk16_aresident_out(
                    activation, packed, activation_scale, weight_scale,
                    partial, tickets, output
                )
                return output
            return scaled_mm(activation, packed, activation_scale, weight_scale)

        actual = call()
        expected = reference(activation, weight, activation_scale, weight_scale).cuda()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(args.repetitions):
            repeated = call()
            torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
        print(f"{mode}: exact BF16 correctness passed")


if __name__ == "__main__":
    main()
