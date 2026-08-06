#!/usr/bin/env python3
"""
Full 80-layer HY3 vLLM PP=2 Dump vs MetaInfer Golden Dump Comparison

Compares vLLM PP=2 inference dumps against MetaInfer golden dumps for all 80 layers,
computing cosine similarity for each of the 7 dump points per layer.

Usage:
    python3 compare_80l_full.py [--dump-dir DUMP_DIR] [--golden-dir GOLDEN_DIR]
                                [--output-report REPORT_FILE]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F


# ── Default paths ──────────────────────────────────────────────
DEFAULT_GOLDEN_DIR = "/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/golden_dump"
DEFAULT_DUMP_DIR = "/data/mzw/vllm-hy3/dumps/pp2_80l"

DUMP_POINTS = [
    "00_input",
    "01_input_layernorm",
    "02_attention_out",
    "03_attention_residual",
    "04_post_attention_layernorm",
    "05_mlp_out",
    "06_output",
]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-12)


def load_tensor(path: str) -> torch.Tensor | None:
    """Load a tensor from file, returning None if not found or on error."""
    if not os.path.exists(path):
        return None
    try:
        t = torch.load(path, map_location="cpu")
        return t.to(torch.bfloat16)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def compare_embedding(dump_dir: str, golden_dir: str) -> dict:
    """Compare embedding (L0 input) between vLLM and golden."""
    result = {}
    vllm_path = os.path.join(dump_dir, "layer_000", "00_input.pt")
    golden_path = os.path.join(golden_dir, "layer_000", "00_input.pt")

    vllm_t = load_tensor(vllm_path)
    golden_t = load_tensor(golden_path)

    if vllm_t is not None and golden_t is not None:
        result["cos_sim"] = cosine_similarity(vllm_t, golden_t).item()
        # Check for bitwise match
        result["bitwise_match"] = bool(torch.equal(vllm_t, golden_t))

        # Additional diagnostic: compare max absolute difference
        diff = (vllm_t.float() - golden_t.float()).abs()
        result["max_abs_diff"] = diff.max().item()
        result["mean_abs_diff"] = diff.mean().item()
        result["shapes_match"] = vllm_t.shape == golden_t.shape
        result["vllm_shape"] = list(vllm_t.shape)
        result["golden_shape"] = list(golden_t.shape)
        result["vllm_norm"] = vllm_t.float().norm().item()
        result["golden_norm"] = golden_t.float().norm().item()
    else:
        result["error"] = "Missing embedding files"

    return result


def compare_layers(dump_dir: str, golden_dir: str, num_layers: int = 80) -> dict:
    """Compare per-layer dump points between vLLM and golden.

    Returns dict with per-layer and per-point cosine similarities.
    """
    all_results = {}
    missing_vllm = []
    missing_golden = []
    point_stats = {p: [] for p in DUMP_POINTS}
    layer_means = {}

    for layer_idx in range(num_layers):
        layer_tag = f"layer_{layer_idx:03d}"
        layer_vllm_dir = os.path.join(dump_dir, layer_tag)
        layer_golden_dir = os.path.join(golden_dir, layer_tag)

        layer_results = {}
        for point in DUMP_POINTS:
            vllm_path = os.path.join(layer_vllm_dir, f"{point}.pt")
            golden_path = os.path.join(layer_golden_dir, f"{point}.pt")

            vllm_t = load_tensor(vllm_path)
            golden_t = load_tensor(golden_path)

            if vllm_t is None:
                missing_vllm.append(f"{layer_tag}/{point}")
                continue
            if golden_t is None:
                missing_golden.append(f"{layer_tag}/{point}")
                continue

            cos_sim = cosine_similarity(vllm_t, golden_t).item()
            layer_results[point] = cos_sim
            point_stats[point].append(cos_sim)

        if layer_results:
            all_results[layer_idx] = layer_results
            values = list(layer_results.values())
            layer_means[layer_idx] = {
                "mean": sum(values) / len(values),
                "min_point": min(layer_results, key=layer_results.get),
                "min_val": min(values),
                "max_point": max(layer_results, key=layer_results.get),
                "max_val": max(values),
            }

    return {
        "layers": all_results,
        "missing_vllm": missing_vllm,
        "missing_golden": missing_golden,
        "point_stats": point_stats,
        "layer_means": layer_means,
    }


def compare_final_norm(dump_dir: str, golden_dir: str) -> dict:
    """Compare vLLM final_norm against golden layer_079/06_output (normed).

    The vLLM dumps final_norm = RMSNorm(layer_079_hidden + residual_079).
    The golden layer_079/06_output = layer_079_hidden + residual_079 (unnormalized).

    We validate by: comparing vLLM final_norm shape/norm, and
    checking that cosine similarity of the two is close to 1.
    This is an approximation since golden doesn't have final_norm directly.
    """
    result = {}
    vllm_path = os.path.join(dump_dir, "layer_080", "final_norm.pt")
    golden_path = os.path.join(golden_dir, "layer_079", "06_output.pt")

    vllm_t = load_tensor(vllm_path)
    golden_unnormed = load_tensor(golden_path)

    if vllm_t is None:
        result["error"] = "Missing vLLM final_norm dump"
    elif golden_unnormed is None:
        result["error"] = "Missing golden layer_079/06_output"
    else:
        result["vllm_shape"] = list(vllm_t.shape)
        result["vllm_norm"] = vllm_t.float().norm().item()
        result["golden_079_output_shape"] = list(golden_unnormed.shape)
        result["golden_079_output_norm"] = golden_unnormed.float().norm().item()

        # Apply RMSNorm manually to golden output
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        eps = 1e-5
        x_f32 = golden_unnormed.float()
        rms = torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
        # Without the weight, we just normalize
        golden_normed_approx = (x_f32 / rms).bfloat16()
        result["cos_sim_approx"] = cosine_similarity(
            vllm_t, golden_normed_approx
        ).item()

    return result


def compare_logits(dump_dir: str, golden_dir: str) -> dict:
    """Check vLLM logits existence and shape.

    Golden dump does not contain logits, so we report structure only.
    """
    result = {}
    logits_path = os.path.join(dump_dir, "layer_080", "logits.pt")
    vllm_logits = load_tensor(logits_path)

    if vllm_logits is not None:
        result["shape"] = list(vllm_logits.shape)
        result["norm"] = vllm_logits.float().norm().item()
        # Top-5 token IDs
        if vllm_logits.dim() >= 2:
            last_token = vllm_logits[-1] if vllm_logits.dim() == 2 else vllm_logits[0, -1]
            top5_vals, top5_ids = torch.topk(last_token.float(), 5)
            result["top5_ids"] = top5_ids.tolist()
            result["top5_vals"] = [f"{v:.4f}" for v in top5_vals.tolist()]
    else:
        result["error"] = "Missing vLLM logits dump"

    return result


def print_report(embedding_result, layers_result, final_norm_result, logits_result,
                 num_layers=80):
    """Print a comprehensive verification report."""
    layer_data = layers_result["layers"]
    layer_means = layers_result["layer_means"]
    point_stats = layers_result["point_stats"]

    print("=" * 80)
    print("  HY3 Full 80-Layer vLLM PP=2 vs MetaInfer Golden Dump — Verification Report")
    print("=" * 80)
    print()

    # ── Summary ──
    print("─" * 80)
    print("  SUMMARY")
    print("─" * 80)
    covered_layers = sorted(layer_data.keys())
    total_points = sum(len(v) for v in layer_data.values())
    all_vals = [v for layer_vals in layer_data.values() for v in layer_vals.values()]
    if all_vals:
        print(f"  Layers compared:  {min(covered_layers)} – {max(covered_layers)} "
              f"({len(covered_layers)}/{num_layers} layers, {total_points} points)")
        print(f"  Mean cos_sim:     {sum(all_vals)/len(all_vals):.6f}")
        print(f"  Min cos_sim:      {min(all_vals):.6f}")
        print(f"  Max cos_sim:      {max(all_vals):.6f}")
        degraded = [(layer_idx, layer_means[layer_idx]["min_val"])
                    for layer_idx in covered_layers
                    if layer_means[layer_idx]["min_val"] < 0.99]
        if degraded:
            print(f"  Degraded layers:  {len(degraded)} (< 0.99)")
            for lidx, mv in degraded:
                pts = layer_data[lidx]
                min_pt = min(pts, key=pts.get)
                print(f"    Layer {lidx}: min={mv:.6f} @ {min_pt}")
        else:
            print("  Degraded layers:  0 (all layers >= 0.99)")
    print()

    # ── Embedding ──
    print("─" * 80)
    print("  EMBEDDING (L0 00_input)")
    print("─" * 80)
    if "error" in embedding_result:
        print(f"  ERROR: {embedding_result['error']}")
    else:
        print(f"  Shapes match:  {embedding_result['shapes_match']}")
        print(f"  Bitwise match: {embedding_result['bitwise_match']}")
        print(f"  Cos sim:       {embedding_result['cos_sim']:.8f}")
        print(f"  Max abs diff:  {embedding_result['max_abs_diff']:.2e}")
        print(f"  Mean abs diff: {embedding_result['mean_abs_diff']:.2e}")
        print(f"  vLLM norm:     {embedding_result['vllm_norm']:.4f}")
        print(f"  Golden norm:   {embedding_result['golden_norm']:.4f}")
    print()

    # ── Missing files ──
    missing_v = layers_result["missing_vllm"]
    missing_g = layers_result["missing_golden"]
    if missing_v:
        print(f"  [MISSING] vLLM dumps ({len(missing_v)} files):")
        for m in missing_v[:20]:
            print(f"    {m}")
        if len(missing_v) > 20:
            print(f"    ... and {len(missing_v) - 20} more")
        print()
    if missing_g:
        print(f"  [MISSING] Golden dumps ({len(missing_g)} files):")
        for m in missing_g[:20]:
            print(f"    {m}")
        if len(missing_g) > 20:
            print(f"    ... and {len(missing_g) - 20} more")
        print()

    # ── Per dump-point stats ──
    print("─" * 80)
    print("  PER DUMP-POINT SUMMARY")
    print("─" * 80)
    print(f"  {'Point':30s}  {'Mean':>10s}  {'Min':>10s}  {'Max':>10s}  {'Count':>6s}")
    print("  " + "-" * 72)
    for p in DUMP_POINTS:
        vals = point_stats[p]
        if vals:
            print(f"  {p:30s}  {sum(vals)/len(vals):10.6f}  {min(vals):10.6f}  "
                  f"{max(vals):10.6f}  {len(vals):6d}")
        else:
            print(f"  {p:30s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}  {0:6d}")
    print()

    # ── Per-layer table ──
    print("─" * 80)
    print("  PER-LAYER MEANS")
    print("─" * 80)
    print(f"  {'Lyr':>4s} | {'input':>8s} {'in_ln':>8s} {'attn':>8s} {'att_res':>8s} "
          f"{'pst_ln':>8s} {'mlp':>8s} {'output':>8s} | {'Mean':>8s} {'Min':>8s}")
    print("  " + "-" * 98)
    for layer_idx, means in sorted(layer_means.items()):
        pts = layer_data[layer_idx]
        v00 = pts.get("00_input", float("nan"))
        v01 = pts.get("01_input_layernorm", float("nan"))
        v02 = pts.get("02_attention_out", float("nan"))
        v03 = pts.get("03_attention_residual", float("nan"))
        v04 = pts.get("04_post_attention_layernorm", float("nan"))
        v05 = pts.get("05_mlp_out", float("nan"))
        v06 = pts.get("06_output", float("nan"))
        print(f"  {layer_idx:03d}  | {v00:8.6f} {v01:8.6f} {v02:8.6f} {v03:8.6f} "
              f"{v04:8.6f} {v05:8.6f} {v06:8.6f} | {means['mean']:8.6f} "
              f"{means['min_val']:8.6f}")
    print()

    # ── Final Norm ──
    print("─" * 80)
    print("  FINAL NORM")
    print("─" * 80)
    if "error" in final_norm_result:
        print(f"  ERROR: {final_norm_result['error']}")
    else:
        print(f"  vLLM final_norm shape:  {final_norm_result['vllm_shape']}")
        print(f"  vLLM final_norm norm:   {final_norm_result['vllm_norm']:.4f}")
        print(f"  Golden 079 output shape: {final_norm_result['golden_079_output_shape']}")
        print(f"  Golden 079 output norm:  {final_norm_result['golden_079_output_norm']:.4f}")
        print(f"  Cos sim (approx normed): {final_norm_result['cos_sim_approx']:.8f}")
    print()

    # ── Logits ──
    print("─" * 80)
    print("  LOGITS")
    print("─" * 80)
    if "error" in logits_result:
        print(f"  ERROR: {logits_result['error']}")
    else:
        print(f"  Logits shape:  {logits_result['shape']}")
        print(f"  Logits norm:   {logits_result['norm']:.4f}")
        if "top5_ids" in logits_result:
            print(f"  Top-5 tokens:")
            for i, (tid, val) in enumerate(zip(logits_result["top5_ids"],
                                                logits_result["top5_vals"])):
                print(f"    {i+1}. token_id={tid}, logit={val}")
    print()

    # ── Conclusion ──
    print("=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    if all_vals:
        mean_cos = sum(all_vals) / len(all_vals)
        min_cos = min(all_vals)
        if mean_cos >= 0.999 and min_cos >= 0.99:
            print(f"  ✓ VERIFIED — Full 80-layer vLLM PP=2 inference matches MetaInfer.")
            print(f"    Mean cosine similarity: {mean_cos:.6f}")
        elif mean_cos >= 0.99:
            print(f"  ~ PARTIALLY VERIFIED — Minor numerical discrepancies detected.")
            print(f"    Mean cosine similarity: {mean_cos:.6f}")
        else:
            print(f"  ✗ DISCREPANCY DETECTED — Significant numerical differences.")
            print(f"    Mean cosine similarity: {mean_cos:.6f}")
        print(f"    Overall min: {min_cos:.6f}")
    else:
        print("  ✗ NO DATA — No comparison points available. Check dump directories.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="HY3 Full 80-Layer vLLM vs MetaInfer Golden Dump Comparison"
    )
    parser.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR,
                        help="Directory containing vLLM dumps")
    parser.add_argument("--golden-dir", default=DEFAULT_GOLDEN_DIR,
                        help="Directory containing MetaInfer golden dumps")
    parser.add_argument("--num-layers", type=int, default=80,
                        help="Number of layers to compare")
    parser.add_argument("--output-report", default=None,
                        help="Path to save the report (default: stdout only)")
    args = parser.parse_args()

    if not os.path.isdir(args.dump_dir):
        print(f"[ERROR] vLLM dump directory not found: {args.dump_dir}")
        print("Run the PP=2 inference with VLLM_HY3_DUMP_DIR set first.")
        sys.exit(1)
    if not os.path.isdir(args.golden_dir):
        print(f"[ERROR] Golden dump directory not found: {args.golden_dir}")
        sys.exit(1)

    print(f"vLLM dump dir:  {args.dump_dir}")
    print(f"Golden dump dir: {args.golden_dir}")
    print(f"Layers:         0–{args.num_layers - 1}")
    print()

    # Run comparisons
    print("Loading and comparing...")
    embedding_result = compare_embedding(args.dump_dir, args.golden_dir)
    layers_result = compare_layers(args.dump_dir, args.golden_dir, args.num_layers)
    final_norm_result = compare_final_norm(args.dump_dir, args.golden_dir)
    logits_result = compare_logits(args.dump_dir, args.golden_dir)

    # Print report
    print_report(embedding_result, layers_result, final_norm_result, logits_result,
                 args.num_layers)

    # Optionally save to file
    if args.output_report:
        import contextlib
        with open(args.output_report, "w") as f:
            with contextlib.redirect_stdout(f):
                print_report(embedding_result, layers_result, final_norm_result,
                            logits_result, args.num_layers)
        print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
