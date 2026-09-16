#!/usr/bin/env python3
"""Sample DCU hardware utilization from rocm-smi during a benchmark request."""
import argparse
import json
import re
import statistics
import subprocess
import time

USE_RE = re.compile(r"DCU\[(\d+)\]\s*: DCU use \(%\):\s*([0-9.]+)")
MEM_RE = re.compile(r"DCU\[(\d+)\]\s*: DCU memory use \(%\):\s*([0-9.]+)")
SCLK_RE = re.compile(r"DCU\[(\d+)\]\s*: sclk clock level:.*?\(([0-9.]+)Mhz\)")
MCLK_RE = re.compile(r"DCU\[(\d+)\]\s*: mclk clock level:.*?\(([0-9.]+)Mhz\)")


def collect() -> dict:
    out = subprocess.check_output(
        ["rocm-smi", "--showuse", "--showmemuse", "--showclocks"], text=True)
    fields = {"use": USE_RE, "mem": MEM_RE, "sclk_mhz": SCLK_RE, "mclk_mhz": MCLK_RE}
    sample = {"t": time.time()}
    for field, pattern in fields.items():
        for gpu, value in pattern.findall(out):
            sample.setdefault(int(gpu), {})[field] = float(value)
    return sample


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            samples.append(collect())
        except subprocess.CalledProcessError:
            pass
        time.sleep(args.interval)

    summary = {}
    for gpu in range(8):
        rows = [s[gpu] for s in samples if gpu in s]
        if not rows:
            continue
        item = {"samples": len(rows)}
        for field in ("use", "mem", "sclk_mhz", "mclk_mhz"):
            vals = [r[field] for r in rows if field in r]
            if vals:
                item[field] = {
                    "mean": round(statistics.mean(vals), 2),
                    "median": round(statistics.median(vals), 2),
                    "p95": round(pct(vals, 0.95), 2),
                    "max": round(max(vals), 2),
                }
        use = [r["use"] for r in rows if "use" in r]
        if use:
            item["use_over_90_pct"] = round(100 * sum(x >= 90 for x in use) / len(use), 1)
        summary[str(gpu)] = item

    result = {"samples": samples, "summary": summary}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
