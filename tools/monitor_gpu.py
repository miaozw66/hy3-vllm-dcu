#!/usr/bin/env python3
"""每 10 秒采样 GPU 利用率与显存，输出到 stdout。

用法:
  python3 tools/monitor_gpu.py <时长秒> <间隔秒>

环境变量:
  MONITOR_NODES  — JSON 格式的节点配置列表，每个节点包含 host, type, docker
                   例如: [{"host":"10.18.17.71","type":"local","docker":""},
                          {"host":"10.18.17.74","type":"remote","docker":"mmh_qwen_opt"}]
                   本地单机默认: [{"host":"localhost","type":"local","docker":""}]
"""
import json
import os
import subprocess
import sys
import time

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 600
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10

_NODES_ENV = os.environ.get("MONITOR_NODES")
if _NODES_ENV:
    NODES = json.loads(_NODES_ENV)
else:
    NODES = [{"host": "10.18.17.71", "type": "local", "docker": ""},
             {"host": "10.18.17.74", "type": "remote", "docker": "mmh_qwen_opt"}]


def sample(node):
    host = node["host"]
    docker = node.get("docker", "")
    try:
        if node.get("type") == "local" or not docker:
            out = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=8).stdout
        else:
            out = subprocess.run(
                ["ssh", host, "docker", "exec", docker,
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
    for node in NODES:
        parts.append(f"{node['host']}: {sample(node)}")
    print(" ".join(parts), flush=True)
    time.sleep(INTERVAL)
