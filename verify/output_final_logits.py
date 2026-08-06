"""
加载 80 层完整计算结果，输出最终的 logits 和生成 token
"""
import glob
import os
import safetensors
import torch
import torch.nn.functional as F

MODEL_DIR = "/data/model/hygon/Hy3-Channel-INT8-w8a8/models/hygon--Hy3-Channel-INT8-w8a8/snapshots/master"
PROMPT = "中国的首都是"
HIDDEN_SIZE = 4096
VOCAB_SIZE = 128256  # Hy3 vocab size


def rms_norm(x, weight, eps=1e-5):
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + eps)
    return (x_f / rms * weight.float()).to(x.dtype)


def main():
    device = torch.device("cuda:0")

    # ── 加载层 79 的最终输出 ──
    print("Loading final output from captured_65_79.pt ...")
    captured = torch.load("captured_65_79.pt", map_location="cpu", weights_only=False)
    final_hidden = captured[79]["06_output"]  # layer 79 output = after 80 layers
    print(f"  Layer 79 06_output shape: {final_hidden.shape}")
    print(f"  Norm: {final_hidden.float().norm():.4f}")

    # ── 加载 final norm (model.norm) ──
    print("\nLoading final model norm ...")
    norm_weight = None
    files = sorted(glob.glob(f"{MODEL_DIR}/model-*-of-*.safetensors"))
    for fname in files:
        f = safetensors.safe_open(fname, framework='pt')
        if 'model.norm.weight' in f.keys():
            norm_weight = f.get_tensor('model.norm.weight')
            print(f"  Found model.norm.weight in {os.path.basename(fname)}")
            break
    print(f"  Norm weight shape: {norm_weight.shape}")

    # ── 加载 lm_head ──
    print("\nLoading lm_head ...")
    lm_head_w = None
    for fname in files:
        f = safetensors.safe_open(fname, framework='pt')
        if 'lm_head.weight' in f.keys():
            lm_head_w = f.get_tensor('lm_head.weight')
            print(f"  Found lm_head.weight in {os.path.basename(fname)}")
            break
    print(f"  lm_head weight: shape={lm_head_w.shape}, dtype={lm_head_w.dtype}")

    # ── 应用 final norm + lm_head ──
    print("\nApplying final norm + lm_head ...")
    final_hidden_gpu = final_hidden.to(device)
    norm_weight_gpu = norm_weight.to(device)
    lm_head_gpu = lm_head_w.to(device)

    normed = rms_norm(final_hidden_gpu, norm_weight_gpu)
    print(f"  After final norm: shape={normed.shape}, norm={normed.float().norm():.4f}")

    logits = torch.mm(normed.bfloat16(), lm_head_gpu.T)
    logits_f32 = logits.float()
    print(f"  Logits shape: {logits_f32.shape}")

    # ── Softmax + top-k tokens ──
    probs = F.softmax(logits_f32, dim=-1)
    top_probs, top_ids = torch.topk(probs, 10, dim=-1)

    # ── Tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")[0]
    print(f"\n  Input tokens ({len(input_ids)}): {input_ids.tolist()}")
    print(f"  Input decoded: {tokenizer.decode(input_ids.tolist())}")

    print("\n" + "=" * 60)
    print("Final Output: Top-10 Predictions per Position")
    print("=" * 60)
    for pos in range(logits_f32.shape[0]):
        print(f"\n  Position {pos} (token: '{tokenizer.decode([input_ids[pos].item()])}'):")
        for k in range(10):
            tid = top_ids[pos, k].item()
            prob = top_probs[pos, k].item()
            token_str = tokenizer.decode([tid])
            print(f"    {k+1:2d}. id={tid:6d}  prob={prob:.6f}  → '{token_str}'")

    # ── Greedy next token prediction ──
    print("\n" + "=" * 60)
    print("Greedy Next Token")
    print("=" * 60)
    last_logits = logits_f32[-1]  # logits for last position
    next_id = torch.argmax(last_logits).item()
    next_prob = F.softmax(last_logits, dim=-1)[next_id].item()
    next_token = tokenizer.decode([next_id])
    full_text = tokenizer.decode(input_ids.tolist() + [next_id])
    print(f"  Next token: id={next_id} ('{next_token}'), prob={next_prob:.6f}")
    print(f"  Full text: '{full_text}'")

    # ── Top-5 greedy continuations ──
    print(f"\n  Top-5 next tokens:")
    vals, ids = torch.topk(last_logits, 5)
    for k in range(5):
        tid = ids[k].item()
        prob = F.softmax(last_logits, dim=-1)[tid].item()
        print(f"    {k+1}. id={tid:6d}  prob={prob:.6f}  → '{tokenizer.decode([tid])}'")


if __name__ == "__main__":
    main()
