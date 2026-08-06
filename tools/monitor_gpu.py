#!/usr/bin/env python3
"""每 10 秒采样两台机器 GPU 利用率与显存，输出到 stdout。
用法: monitor_gpu.py <时长秒> <间隔秒>"""
import subprocess
import sys
import time

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 600
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NODES = ["10.18.17.71", "10.18.17.74"]

def sample(host):
    try:
        if host == "10.18.17.71":
            out = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=8).stdout
        else:
            out = subprocess.run(
                ["ssh", host, "docker", "exec", "mmh_qwen_opt",
                 "rocm-smi", "--showuse", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=15).stdout
        uses = [l.split(":")[-1].strip() for l in out.splitlines()
                if "use (%" in l]
        mems = [l.split(":")[-1].strip() for l in out.splitlines()
                if "vram" in l.lower() and ("total" in l.lower() or "used" in l.lower())]
        return "uses=" + ",".join(uses) + " | " + "mem=" + " ".join(mems[:8])
    except Exception as e:
        return f"ERR {e}"

t0 = time.time()
while time.time() - t0 < DURATION:
    ts = time.strftime("%H:%M:%S")
    parts = [f"[{ts}]"]
    for host in NODES:
        parts.append(f"{host}: {sample(host)}")
    print(" ".join(parts), flush=True)
    time.sleep(INTERVAL)
