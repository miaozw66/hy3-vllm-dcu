#!/usr/bin/env python3
"""Fail unless every local HCU has enough free VRAM for vLLM startup."""

import argparse
import csv
import io
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=60000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = subprocess.run(
        ["hy-smi", "--showmemavailable", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    free_by_device = {
        row["device"]: int(row["Available memory size (MiB)"])
        for row in rows
        if row.get("device")
    }

    if len(free_by_device) != args.expected_gpus:
        print(
            f"GPU preflight failed: found {len(free_by_device)} devices, "
            f"expected {args.expected_gpus}",
            file=sys.stderr,
        )
        return 1

    failed = False
    for device, free_mib in sorted(free_by_device.items()):
        print(f"{device}: free={free_mib} MiB, required>={args.min_free_mib} MiB")
        failed |= free_mib < args.min_free_mib

    if failed:
        print("GPU preflight failed: insufficient free VRAM", file=sys.stderr)
        return 1

    print(f"GPU preflight passed: {len(free_by_device)}/{args.expected_gpus} devices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
