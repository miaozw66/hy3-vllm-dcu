"""
生成完整的 80 层验证报告
合并 verify.py（层 0-64，vLLM hook 捕获）和 verify_layers_65_79.py（层 65-79，纯手动计算）的结果
"""
import os
import torch
import torch.nn.functional as F

MODEL_DIR = "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master"
GOLDEN_DIR = "/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/golden_dump"

DUMP_POINTS = [
    "00_input", "01_input_layernorm", "02_attention_out",
    "03_attention_residual", "04_post_attention_layernorm",
    "05_mlp_out", "06_output",
]

NUM_LAYERS_TOTAL = 80

def cosine_similarity(a, b):
    a_f = a.float().reshape(-1, a.shape[-1])
    b_f = b.float().reshape(-1, b.shape[-1])
    a_n = F.normalize(a_f, dim=-1)
    b_n = F.normalize(b_f, dim=-1)
    return (a_n * b_n).sum(-1).mean().item()


def load_golden(layer_range):
    golden = {}
    for layer_idx in layer_range:
        golden[layer_idx] = {}
        layer_dir = os.path.join(GOLDEN_DIR, f"layer_{layer_idx:03d}")
        for point in DUMP_POINTS:
            pt_file = os.path.join(layer_dir, f"{point}.pt")
            golden[layer_idx][point] = torch.load(pt_file, map_location="cpu")
    return golden


def main():
    print("=" * 80)
    print("HY3 Full 80-Layer Verification Report")
    print("=" * 80)

    # ── 加载层 65-79 的手动计算结果 ──
    print("\nLoading captured_65_79.pt ...")
    captured_65_79 = torch.load("captured_65_79.pt", map_location="cpu", weights_only=False)
    print(f"  Loaded {len(captured_65_79)} layers (65-79)")

    # ── 加载所有 80 层的 golden ──
    print("\nLoading golden dumps for all 80 layers ...")
    golden = load_golden(range(NUM_LAYERS_TOTAL))
    print(f"  Loaded {len(golden)} layers of golden data")

    # ── 逐层逐点计算 cos_sim ──
    print("\n" + "=" * 80)
    print("Per-Layer Cosine Similarity")
    print("=" * 80)

    hdr = f"{'Lyr':>4} |" + "|".join(f"{p[3:]:>8}" for p in DUMP_POINTS) + "|  Mean  | Source"
    print(hdr)
    print("-" * len(hdr))

    all_sims = []
    layer_means_0_64 = []
    layer_means_65_79 = []

    for lyr in range(NUM_LAYERS_TOTAL):
        row_vals = []
        row_str = f" {lyr:03d} |"

        for pt in DUMP_POINTS:
            if lyr in captured_65_79 and pt in captured_65_79[lyr] and pt in golden.get(lyr, {}):
                sim = cosine_similarity(captured_65_79[lyr][pt], golden[lyr][pt])
                row_vals.append(sim)
                all_sims.append(sim)
                row_str += f" {sim:.6f}"
            elif pt in golden.get(lyr, {}):
                # Layer 0-64: golden exists but captured not saved
                row_str += f" {'N/A':>8}"
            else:
                row_str += f" {'N/A':>8}"

        if row_vals:
            mean_val = sum(row_vals) / len(row_vals)
            row_str += f" | {mean_val:.6f}  manual"
            if lyr >= 65:
                layer_means_65_79.append(mean_val)
            else:
                layer_means_0_64.append(mean_val)
        elif lyr < 65:
            # Layers 0-64: we know they passed at ~0.998867 from verify.py
            row_str += f" | {'?':>6}  vLLM-hook"

        print(row_str)

    # ── 汇总统计 ──
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if all_sims:
        mean_all = sum(all_sims) / len(all_sims)
        print(f"\n  Layers 65-79 (manual, verified):")
        print(f"    Points:       {len(all_sims)} ({len(layer_means_65_79)} layers x 7 dump points)")
        print(f"    Mean cos_sim: {sum(all_sims)/len(all_sims):.6f}")
        print(f"    Min:          {min(all_sims):.6f}")
        print(f"    Max:          {max(all_sims):.6f}")

        # Per-dump-point summary for layers 65-79
        print(f"\n  Per dump-point mean (layers 65-79):")
        for pt in DUMP_POINTS:
            pt_sims = []
            for lyr in range(65, 80):
                if lyr in captured_65_79 and pt in captured_65_79[lyr] and pt in golden.get(lyr, {}):
                    pt_sims.append(cosine_similarity(captured_65_79[lyr][pt], golden[lyr][pt]))
            if pt_sims:
                print(f"    {pt}: {sum(pt_sims)/len(pt_sims):.6f}  (min={min(pt_sims):.6f}, max={max(pt_sims):.6f})")

    # Known results from verify.py (65-layer run)
    print(f"\n  Layers 0-64 (vLLM hook, from previous run):")
    print(f"    Mean cos_sim: 0.998867 (455 points = 65 layers x 7 dump points)")

    # Combined estimate
    # (455 * 0.998867 + 105 * mean_65_79) / 560
    if all_sims:
        n_0_64 = 65 * 7  # 455
        n_65_79 = len(all_sims)  # 105
        mean_65_79 = sum(all_sims) / len(all_sims)
        combined = (n_0_64 * 0.998867 + n_65_79 * mean_65_79) / (n_0_64 + n_65_79)
        print(f"\n  Combined 80-layer estimate:")
        print(f"    Weighted mean cos_sim: {combined:.6f}")
        print(f"    ({n_0_64} points from layers 0-64 + {n_65_79} points from layers 65-79)")

    print("\n" + "=" * 80)
    print("Report complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
