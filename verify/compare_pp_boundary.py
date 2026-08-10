"""
对比 PP=2 边界 tensor 与 golden dump

用法:
  python compare_pp_boundary.py

对比项:
  1. Node 0 send_combined (hidden_states + residual) vs golden layer_039/06_output
  2. 如果存在 recv 文件, 对比 send vs recv (传输保真度)
"""
import os
import torch
import torch.nn.functional as F

GOLDEN_DIR = os.environ.get(
    "VLLM_GOLDEN_DUMP_DIR",
    "/data/mzw/MetaInfer/nodes/worker24/.metainfer/tasks/hy3-test2-44db1034/code/004/golden_dump")
DUMP_DIR = os.environ.get("VLLM_PP_BOUNDARY_DIR", "/tmp/pp_boundary")


def cosine_similarity(a, b):
    """计算两个 tensor 的余弦相似度"""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos = (a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-12)
    return cos.item()


def compare():
    print("=" * 70)
    print("PP=2 边界 tensor 对比 golden dump")
    print("=" * 70)

    # 加载 golden dump: layer_039/06_output.pt
    golden_path = os.path.join(GOLDEN_DIR, "layer_039", "06_output.pt")
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden dump not found: {golden_path}")
        return
    golden = torch.load(golden_path, map_location="cpu")

    # 同时加载 layer_040/00_input.pt (应该与 layer_039/06_output 相同)
    golden_040_input = os.path.join(GOLDEN_DIR, "layer_040", "00_input.pt")
    if os.path.exists(golden_040_input):
        golden_040 = torch.load(golden_040_input, map_location="cpu")
        assert torch.equal(golden, golden_040), \
            "golden layer_039/06_output != layer_040/00_input (unexpected!)"
        print("✓ golden 连续性验证: layer_039/06_output == layer_040/00_input (bitwise)")

    print(f"\ngolden layer_039/06_output: shape={list(golden.shape)}, "
          f"dtype={golden.dtype}, norm={golden.float().norm():.4f}")

    # ── 1. Node 0 send_combined vs golden ──
    send_combined_path = os.path.join(DUMP_DIR, "send_combined.pt")
    send_hidden_path = os.path.join(DUMP_DIR, "send_hidden_states.pt")
    send_residual_path = os.path.join(DUMP_DIR, "send_residual.pt")

    print(f"\n{'─' * 70}")
    print("1. Node 0 (PP rank 0) 发送端对比")

    if os.path.exists(send_combined_path):
        send_combined = torch.load(send_combined_path, map_location="cpu")
        cos = cosine_similarity(send_combined, golden)
        max_diff = (send_combined.float() - golden.float()).abs().max().item()
        l2 = (send_combined.float() - golden.float()).norm().item()
        print(f"  send_combined vs golden layer_039/06_output:")
        print(f"    cos_sim  = {cos:.8f}")
        print(f"    max_diff = {max_diff:.6e}")
        print(f"    L2_diff  = {l2:.6e}")
        if cos > 0.999:
            print(f"  ✓ PASS — Node 0 PP 边界输出与 golden 一致")
        else:
            print(f"  ✗ FAIL — Node 0 PP 边界输出与 golden 不一致!")
    else:
        print(f"  [SKIP] 文件不存在: {send_combined_path}")

    if os.path.exists(send_hidden_path) and os.path.exists(send_residual_path):
        hs = torch.load(send_hidden_path, map_location="cpu")
        res = torch.load(send_residual_path, map_location="cpu")
        print(f"  send_hidden_states: shape={list(hs.shape)}, norm={hs.float().norm():.4f}")
        print(f"  send_residual:      shape={list(res.shape)}, norm={res.float().norm():.4f}")
        print(f"  注意: golden dump 中无独立的 hidden_states/residual 分离值，"
              f"仅能对比 combined")

    # ── 2. Node 1 recv vs golden ──
    recv_combined_path = os.path.join(DUMP_DIR, "recv_combined.pt")
    recv_hidden_path = os.path.join(DUMP_DIR, "recv_hidden_states.pt")
    recv_residual_path = os.path.join(DUMP_DIR, "recv_residual.pt")

    print(f"\n{'─' * 70}")
    print("2. Node 1 (PP rank 1) 接收端对比")

    has_recv = os.path.exists(recv_combined_path)
    if not has_recv:
        print(f"  [SKIP] 文件不存在: {recv_combined_path}")
        print(f"  → 需要从 Node 1 复制: scp node1:/tmp/pp_boundary/recv_*.pt /tmp/pp_boundary/")
    else:
        recv_combined = torch.load(recv_combined_path, map_location="cpu")
        cos = cosine_similarity(recv_combined, golden)
        max_diff = (recv_combined.float() - golden.float()).abs().max().item()
        print(f"  recv_combined vs golden layer_039/06_output:")
        print(f"    cos_sim  = {cos:.8f}")
        print(f"    max_diff = {max_diff:.6e}")

    # ── 3. send vs recv (传输保真度) ──
    print(f"\n{'─' * 70}")
    print("3. 传输保真度 (send vs recv)")

    if os.path.exists(send_combined_path) and os.path.exists(recv_combined_path):
        send_c = torch.load(send_combined_path, map_location="cpu")
        recv_c = torch.load(recv_combined_path, map_location="cpu")
        if torch.equal(send_c, recv_c):
            print(f"  send_combined == recv_combined: ✓ bitwise 一致")
        else:
            cos = cosine_similarity(send_c, recv_c)
            max_diff = (send_c.float() - recv_c.float()).abs().max().item()
            print(f"  send_combined vs recv_combined:")
            print(f"    cos_sim  = {cos:.8f}")
            print(f"    max_diff = {max_diff:.6e}")
            if cos > 0.999999:
                print(f"  ✓ PASS — 传输基本保真 (cos > 0.999999)")
            else:
                print(f"  ✗ FAIL — 传输数据有显著差异!")

    if os.path.exists(send_hidden_path) and os.path.exists(recv_hidden_path):
        sh = torch.load(send_hidden_path, map_location="cpu")
        rh = torch.load(recv_hidden_path, map_location="cpu")
        if torch.equal(sh, rh):
            print(f"  send_hidden_states == recv_hidden_states: ✓ bitwise 一致")
        else:
            cos = cosine_similarity(sh, rh)
            max_diff = (sh.float() - rh.float()).abs().max().item()
            print(f"  send_hidden_states vs recv_hidden_states:")
            print(f"    cos_sim  = {cos:.8f}, max_diff = {max_diff:.6e}")
            if max_diff > 1e-3:
                print(f"  ✗ FAIL — hidden_states 传输严重损坏!")

    if os.path.exists(send_residual_path) and os.path.exists(recv_residual_path):
        sr = torch.load(send_residual_path, map_location="cpu")
        rr = torch.load(recv_residual_path, map_location="cpu")
        if torch.equal(sr, rr):
            print(f"  send_residual == recv_residual: ✓ bitwise 一致")
        else:
            cos = cosine_similarity(sr, rr)
            max_diff = (sr.float() - rr.float()).abs().max().item()
            print(f"  send_residual vs recv_residual:")
            print(f"    cos_sim  = {cos:.8f}, max_diff = {max_diff:.6e}")
            if max_diff > 1e-3:
                print(f"  ✗ FAIL — residual 传输严重损坏!")

    # ── 4. 总结 ──
    print(f"\n{'=' * 70}")
    print("结论")
    print("=" * 70)
    all_ok = True

    if os.path.exists(send_combined_path):
        cos = cosine_similarity(torch.load(send_combined_path, map_location="cpu"), golden)
        if cos > 0.999:
            print("  ✓ Node 0 边界计算正确 (前 40 层 OK)")
        else:
            print(f"  ✗ Node 0 边界计算错误 (cos={cos:.6f}), 问题在前 40 层")
            all_ok = False

    if os.path.exists(send_combined_path) and os.path.exists(recv_combined_path):
        send_c = torch.load(send_combined_path, map_location="cpu")
        recv_c = torch.load(recv_combined_path, map_location="cpu")
        if torch.equal(send_c, recv_c):
            print("  ✓ PP 传输保真 (send == recv bitwise)")
        else:
            cos = cosine_similarity(send_c, recv_c)
            if cos > 0.999999:
                print(f"  ✓ PP 传输基本保真 (cos={cos:.8f})")
            else:
                print(f"  ✗ PP 传输损坏 (cos={cos:.6f})")
                all_ok = False

    if all_ok:
        print("\n  如果边界数据正确但推理仍然乱码，问题在后半段 (layers 40-79)")
        print("  或 final norm / lm_head")

    print()


if __name__ == "__main__":
    compare()
