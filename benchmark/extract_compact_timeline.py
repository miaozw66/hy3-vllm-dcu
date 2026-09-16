#!/usr/bin/env python3
"""Extract a compact per-rank kernel/event timeline from a torch-profiler
.pt.trace.json.gz, so that subagents can analyse the Sep-3 c1/MTP trace without
re-decompressing 45-56 MiB gz blobs every time.

Output (one JSON per rank, sorted by ts):
  {
    "trace": "...",
    "generations": [ {i, name, ts, dur} ... ],           # decode-step boundaries
    "devices":   [ {ts, dur, tid, cat, name} ... ],      # GPU kernels + memcpy (device-side)
    "cpu_events": [ {ts, dur, tid, cat, name} ... ],     # hipGraphLaunch / hipStreamWaitEvent /
                                                         # hipEventRecord / MTP-EAGLE-hy_v3_mtp CPU,
                                                         # Memcpy HtoD/HtoD host-side
  }
Kernel events that fall inside a generation annotation interval get tagged with
the 0-based step index (the first generation that covers event.ts), enabling
per-step rollups. Events before/after all generations are tagged -1.

tid for device events is the CUDA/HIP stream id (torch profiler encodes stream
as tid on the 'GPU kernels'/'GPU memcpy' process). pid is dropped.
"""
import argparse
import gzip
import json
import os
import sys

GPU_CATS = {"kernel", "gpu_memcpy"}

# Host-side events worth keeping: graph replay / stream sync / HtoD + speculative CPU.
CPU_MARKERS = (
    "hipGraphLaunch",
    "hipStreamWaitEvent",
    "hipEventRecord",
    "hipEventQuery",
    "Memcpy HtoD",
    "mtp",
    "MTP",
    "EAGLE",
    "hy_v3_mtp",
    "spec_decode",
    "execute_",
    "sampler",
    "cudaGraphLaunch",
    "cudaStreamWaitEvent",
    "cudaEventRecord",
)


def load_trace(path: str) -> list:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            d = json.load(f)
    else:
        with open(path) as f:
            d = json.load(f)
    return d.get("traceEvents", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cpu-dur-us", type=float, default=0.0)
    args = ap.parse_args()

    evs = load_trace(args.trace)
    gens = [
        e for e in evs
        if e.get("ph") == "X" and "generation" in e.get("name", "") and e.get("cat") == "gpu_user_annotation"
    ]
    gens = sorted(gens, key=lambda e: e["ts"])

    devices = []
    cpu_events = []
    for e in evs:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        dur = e.get("dur", 0.0)
        if not name or dur is None:
            continue
        ts = e["ts"]
        if cat in GPU_CATS:
            devices.append({"ts": ts, "dur": dur, "tid": e.get("tid", 0), "cat": cat, "name": name})
        elif any(m in name for m in CPU_MARKERS) and cat not in GPU_CATS:
            cpu_events.append({"ts": ts, "dur": dur, "tid": e.get("tid", 0), "cat": cat, "name": name})

    devices.sort(key=lambda e: e["ts"])
    cpu_events.sort(key=lambda e: e["ts"])

    # tag device events with step index
    gi = 0
    ngen = len(gens)
    for de in devices:
        ts = de["ts"]
        while gi < ngen and ts >= gens[gi]["ts"] + gens[gi]["dur"]:
            gi += 1
        de["step"] = gi if (gi < ngen and gens[gi]["ts"] <= ts < gens[gi]["ts"] + gens[gi]["dur"]) else -1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "trace": args.trace,
                "generations": [
                    {"i": i, "name": g["name"], "ts": g["ts"], "dur": g["dur"]}
                    for i, g in enumerate(gens)
                ],
                "n_devices": len(devices),
                "n_cpu_events": len(cpu_events),
                "devices": devices,
                "cpu_events": cpu_events,
            },
            f,
        )
    print(f"{args.trace}")
    print(f"  generations={len(gens)}  device_events={len(devices)}  cpu_events={len(cpu_events)}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
