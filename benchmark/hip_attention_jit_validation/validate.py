#!/usr/bin/env python3
"""Direct HIP JIT validation for the HY3 MTP paged-attention kernel."""

import gc
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
KERNEL = ROOT / "vllm/csrc/rocm/hy3_mtp_paged_attention.cu"
BINDING = HERE / "binding.cpp"
OP = None
SCALE = 1.0 / (128.0**0.5)


def reference(q, key_cache, value_cache, query_start, seq_lens, block_tables):
    """Compute the NHD, Hkv=1, paged causal result in independent FP32."""
    result = torch.empty_like(q, dtype=torch.float32)
    batch = seq_lens.numel()
    for request in range(batch):
        start = int(query_start[request].item())
        seq_len = int(seq_lens[request].item())
        for query_pos in range(4):
            token = start + query_pos
            causal_end = seq_len - 4 + query_pos
            positions = torch.arange(causal_end + 1, device=q.device)
            logical_blocks = torch.div(positions, 16, rounding_mode="floor")
            physical_blocks = block_tables[request, logical_blocks].long()
            offsets = positions.remainder(16)
            keys = key_cache[physical_blocks, offsets, 0].float()
            values = value_cache[physical_blocks, offsets, 0].float()
            for head in range(8):
                scores = keys @ q[token, head].float() * SCALE
                result[token, head] = torch.softmax(scores, dim=0) @ values
    return result


def make_case(dtype, seq_lens):
    batch = len(seq_lens)
    max_blocks = max((length + 15) // 16 for length in seq_lens)
    # Each row is a random permutation, deliberately decoupled from logical IDs.
    num_cache_blocks = batch * max_blocks + 17
    block_tables = torch.stack(
        [torch.randperm(num_cache_blocks, device="cuda", dtype=torch.int32)[:max_blocks]
         for _ in range(batch)]
    )
    q = torch.randn((batch * 4, 8, 128), device="cuda", dtype=dtype)
    key_cache = torch.randn((num_cache_blocks, 16, 1, 128), device="cuda", dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    out = torch.empty_like(q)
    query_start = torch.arange(0, batch * 4 + 1, 4, device="cuda", dtype=torch.int32)
    seq_lens_tensor = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables


def run_case(dtype, seq_lens):
    tensors = make_case(dtype, seq_lens)
    out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables = tensors
    OP(out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables, SCALE)
    torch.cuda.synchronize()
    ref = reference(q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables)
    diff = (out.float() - ref).abs()
    print(
        f"dtype={str(dtype).replace('torch.', '')} B={len(seq_lens)} "
        f"seq_lens={seq_lens} max_error={diff.max().item():.7g} "
        f"mean_error={diff.mean().item():.7g}"
    )
    del ref, tensors, out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables
    gc.collect()
    torch.cuda.empty_cache()


def check_invalid_physical_block():
    tensors = make_case(torch.bfloat16, [2048])
    out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables = tensors
    out.fill_(float("nan"))
    # This logical block is read by all four queries. The kernel must return
    # before forming a KV pointer from this out-of-range physical block ID.
    block_tables[:, 0] = key_cache.size(0)
    OP(out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables, SCALE)
    torch.cuda.synchronize()
    if not torch.isnan(out).all().item():
        raise AssertionError("invalid physical block metadata wrote output")
    print("invalid_metadata=PASS (out-of-range physical block performed no output write)")
    del tensors, out, q, key_cache, value_cache, query_start, seq_lens_tensor, block_tables
    gc.collect()
    torch.cuda.empty_cache()


def main():
    global OP
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("this validation requires a PyTorch ROCm GPU runtime")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)
    load(
        name="hy3_mtp_paged_attention_jit_validation",
        sources=[str(BINDING), str(KERNEL)],
        build_directory=str(BUILD),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    OP = torch.ops.hy3_mtp_jit_validation.paged_attention
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    print(f"device={torch.cuda.get_device_name(0)} hip={torch.version.hip}")
    for dtype in (torch.bfloat16, torch.float16):
        run_case(dtype, [2048])
        run_case(dtype, [2048, 2047])
    check_invalid_physical_block()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
